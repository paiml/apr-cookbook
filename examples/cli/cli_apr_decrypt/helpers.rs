#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use clap::Parser;
use proptest::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Parser)]
#[command(
    name = "apr-decrypt",
    about = "Decrypt .apr model weights (APR-SPEC encryption envelope)"
)]
pub struct DecryptConfig {
    /// Input encrypted file
    #[arg(value_name = "ENCRYPTED.apr")]
    pub input: Option<String>,

    /// Output path for decrypted model
    #[arg(short, long)]
    pub output: Option<String>,

    /// Decryption password
    #[arg(short, long)]
    pub password: Option<String>,

    /// Run with demo encrypted payload
    #[arg(long)]
    pub demo: bool,
}

// ---------------------------------------------------------------------------
// Key derivation and crypto primitives
// ---------------------------------------------------------------------------

/// BLAKE3 context string for key derivation (APR-SPEC encryption).
pub const KEY_CONTEXT: &str = "apr-encrypt-v1";

/// Size of the MAC appended to ciphertext.
pub const MAC_SIZE: usize = 32;

// Derive a 256-bit key from a password using BLAKE3 key derivation.
//
// Uses the `apr-encrypt-v1` context string so keys are domain-separated
/// from other BLAKE3 uses in the APR toolchain.
pub fn derive_key(password: &str) -> [u8; 32] {
    blake3::derive_key(KEY_CONTEXT, password.as_bytes())
}

// Generate a deterministic keystream block for the given key and counter.
//
/// Each call produces 32 bytes of keystream by hashing `key || counter`.
pub fn keystream_block(key: &[u8; 32], counter: u64) -> [u8; 32] {
    let mut input = Vec::with_capacity(40);
    input.extend_from_slice(key);
    input.extend_from_slice(&counter.to_le_bytes());
    *blake3::hash(&input).as_bytes()
}

/// Compute a MAC (message authentication code) over data using a BLAKE3 keyed hash.
pub fn compute_mac(key: &[u8; 32], data: &[u8]) -> [u8; 32] {
    *blake3::keyed_hash(key, data).as_bytes()
}

// XOR `data` against the BLAKE3 keystream derived from `key`.
//
// This function is its own inverse: encrypting and decrypting are the same
/// XOR operation with the same keystream.
pub fn xor_keystream(data: &[u8], key: &[u8; 32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    let mut counter: u64 = 0;
    let mut block = [0u8; 32];
    let mut block_offset = 32; // force generation on first byte

    for &byte in data {
        if block_offset >= 32 {
            block = keystream_block(key, counter);
            counter = counter.wrapping_add(1);
            block_offset = 0;
        }
        out.push(byte ^ block[block_offset]);
        block_offset += 1;
    }

    out
}

// ---------------------------------------------------------------------------
// Encrypt / Decrypt payloads
// ---------------------------------------------------------------------------

// Encrypt plaintext into the APR encryption envelope: `[ciphertext || MAC]`.
//
// 1. XOR plaintext with BLAKE3 keystream to produce ciphertext.
// 2. Compute a BLAKE3 keyed-hash MAC over the ciphertext.
/// 3. Append the 32-byte MAC.
pub fn encrypt_payload(plaintext: &[u8], key: &[u8; 32]) -> Vec<u8> {
    let ciphertext = xor_keystream(plaintext, key);
    let mac = compute_mac(key, &ciphertext);
    let mut envelope = ciphertext;
    envelope.extend_from_slice(&mac);
    envelope
}

// Decrypt an APR encryption envelope, verifying the MAC first.
//
// Returns `Err` if the envelope is too short or the MAC does not match
/// (wrong password or tampered data).
pub fn decrypt_payload(envelope: &[u8], key: &[u8; 32]) -> Result<Vec<u8>> {
    if envelope.len() < MAC_SIZE {
        return Err(CookbookError::invalid_format(
            "Encrypted payload too short: missing MAC",
        ));
    }

    let split = envelope.len() - MAC_SIZE;
    let ciphertext = &envelope[..split];
    let stored_mac = &envelope[split..];

    let expected_mac = compute_mac(key, ciphertext);
    if stored_mac != expected_mac {
        return Err(CookbookError::invalid_format(
            "MAC verification failed: wrong password or corrupted data",
        ));
    }

    Ok(xor_keystream(ciphertext, key))
}

// ---------------------------------------------------------------------------
// Demo mode
// ---------------------------------------------------------------------------

/// Generate deterministic model bytes using `DefaultHasher` (no external RNG).
pub fn demo_model_bytes(seed: u64, size: usize) -> Vec<u8> {
    (0..size)
        .map(|i| {
            let mut h = DefaultHasher::new();
            seed.hash(&mut h);
            i.hash(&mut h);
            (h.finish() & 0xFF) as u8
        })
        .collect()
}

pub fn run_demo() -> Result<()> {
    let mut ctx = RecipeContext::new("cli_apr_decrypt")?;
    let password = "demo-secret-key-2026";

    println!("APR Decrypt - Demo Mode");
    println!("=======================");
    println!();

    // 1. Generate deterministic model payload
    let seed = hash_name_to_seed("decrypt-demo-model");
    let plaintext = demo_model_bytes(seed, 2048);
    println!("Original model payload: {} bytes", plaintext.len());

    // 2. Derive key
    let key = derive_key(password);
    println!(
        "Key derived from password (BLAKE3): {:02x}{:02x}{:02x}{:02x}...{:02x}{:02x}{:02x}{:02x}",
        key[0], key[1], key[2], key[3], key[28], key[29], key[30], key[31]
    );

    // 3. Encrypt
    let encrypted = encrypt_payload(&plaintext, &key);
    println!(
        "Encrypted envelope:     {} bytes (ciphertext {} + MAC {})",
        encrypted.len(),
        encrypted.len() - MAC_SIZE,
        MAC_SIZE,
    );

    // 4. Verify ciphertext differs from plaintext
    let ct_differs = encrypted[..plaintext.len()] != plaintext[..];
    println!(
        "Ciphertext differs:     {}",
        if ct_differs { "YES" } else { "NO" }
    );

    // 5. Decrypt
    let decrypted = decrypt_payload(&encrypted, &key)?;
    println!("Decrypted payload:      {} bytes", decrypted.len());

    // 6. Roundtrip verification
    let roundtrip_ok = decrypted == plaintext;
    println!(
        "Roundtrip match:        {}",
        if roundtrip_ok { "PASS" } else { "FAIL" }
    );

    // 7. Verify wrong password fails
    let wrong_key = derive_key("wrong-password");
    let wrong_result = decrypt_payload(&encrypted, &wrong_key);
    println!(
        "Wrong-password reject:  {}",
        if wrong_result.is_err() {
            "PASS"
        } else {
            "FAIL"
        }
    );

    // 8. Write artefacts to temp dir for inspection
    let enc_path = ctx.path("model.apr.enc");
    let dec_path = ctx.path("model.apr");
    std::fs::write(&enc_path, &encrypted)?;
    std::fs::write(&dec_path, &decrypted)?;

    println!();
    println!("Artefacts (temp dir):");
    println!("  encrypted: {}", enc_path.display());
    println!("  decrypted: {}", dec_path.display());

    // Record metrics
    ctx.record_metric("plaintext_size", plaintext.len() as i64);
    ctx.record_metric("encrypted_size", encrypted.len() as i64);
    ctx.record_metric("decrypted_size", decrypted.len() as i64);
    ctx.record_metric("mac_size", MAC_SIZE as i64);
    ctx.record_string_metric("mac_verified", if roundtrip_ok { "pass" } else { "fail" });

    println!();
    ctx.report()?;

    Ok(())
}

// ---------------------------------------------------------------------------
// File-based decrypt
// ---------------------------------------------------------------------------

pub fn run_decrypt(config: &DecryptConfig) -> Result<()> {
    let Some(input_path) = config.input.clone() else {
        return Err(CookbookError::invalid_format(
            "Input file required: provide <ENCRYPTED.apr> path",
        ));
    };

    let Some(password) = config.password.clone() else {
        return Err(CookbookError::invalid_format(
            "Password required: use --password <PASS>",
        ));
    };

    let mut ctx = RecipeContext::new("cli_apr_decrypt")?;

    let encrypted = std::fs::read(&input_path).map_err(|e| {
        CookbookError::invalid_format(format!("Failed to read {}: {}", input_path, e))
    })?;

    println!("APR Decrypt");
    println!("===========");
    println!("Input:          {}", input_path);
    println!("Input size:     {} bytes", encrypted.len());

    let key = derive_key(&password);
    let decrypted = decrypt_payload(&encrypted, &key)?;

    println!("Decrypted size: {} bytes", decrypted.len());
    println!("MAC verified:   PASS");

    let output_path = config.output.clone().unwrap_or_else(|| {
        if std::path::Path::new(&input_path)
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("enc"))
        {
            input_path[..input_path.len() - 4].to_string()
        } else {
            format!("{}.dec", input_path)
        }
    });

    std::fs::write(&output_path, &decrypted).map_err(|e| {
        CookbookError::invalid_format(format!("Failed to write {}: {}", output_path, e))
    })?;

    println!("Output:         {}", output_path);

    ctx.record_metric("encrypted_size", encrypted.len() as i64);
    ctx.record_metric("decrypted_size", decrypted.len() as i64);
    ctx.record_string_metric("mac_verified", "pass");

    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================
