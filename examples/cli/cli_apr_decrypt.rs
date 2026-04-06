//! # Recipe: APR Model Decrypt CLI
//!
//! **Category**: CLI Tools
//! **CLI Equivalent**: `apr decrypt`
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/aes256-gcm-decrypt-v1.yaml
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Demonstrate the `apr decrypt` workflow: decrypt model weights that were
//! encrypted with `apr encrypt` using a BLAKE3-derived keystream and MAC.
//! Shows the APR-SPEC encryption envelope: `[ciphertext || MAC-32]`.
//!
//! ## Run Command
//! ```bash
//! cargo run --example cli_apr_decrypt
//! cargo run --example cli_apr_decrypt -- --demo
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr inspect model.apr          # APR native format
//! apr inspect model.gguf         # GGUF (llama.cpp compatible)
//! apr inspect model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Amershi, S. et al. (2019). *Software Engineering for Machine Learning: A Case Study*. ICSE. DOI: 10.1109/ICSE-SEIP.2019.00042

use apr_cookbook::prelude::*;
use clap::Parser;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn main() -> Result<()> {
    let config = DecryptConfig::parse();

    if config.demo || config.input.is_none() {
        return run_demo();
    }

    run_decrypt(&config)
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Decrypt APR model weights (APR-SPEC encryption envelope)
#[derive(Debug, Clone, Parser)]
#[command(
    name = "apr-decrypt",
    about = "Decrypt .apr model weights (APR-SPEC encryption envelope)"
)]
struct DecryptConfig {
    /// Input encrypted file
    #[arg(value_name = "ENCRYPTED.apr")]
    input: Option<String>,

    /// Output path for decrypted model
    #[arg(short, long)]
    output: Option<String>,

    /// Decryption password
    #[arg(short, long)]
    password: Option<String>,

    /// Run with demo encrypted payload
    #[arg(long)]
    demo: bool,
}

// ---------------------------------------------------------------------------
// Key derivation and crypto primitives
// ---------------------------------------------------------------------------

/// BLAKE3 context string for key derivation (APR-SPEC encryption).
const KEY_CONTEXT: &str = "apr-encrypt-v1";

/// Size of the MAC appended to ciphertext.
const MAC_SIZE: usize = 32;

/// Derive a 256-bit key from a password using BLAKE3 key derivation.
///
/// Uses the `apr-encrypt-v1` context string so keys are domain-separated
/// from other BLAKE3 uses in the APR toolchain.
fn derive_key(password: &str) -> [u8; 32] {
    blake3::derive_key(KEY_CONTEXT, password.as_bytes())
}

/// Generate a deterministic keystream block for the given key and counter.
///
/// Each call produces 32 bytes of keystream by hashing `key || counter`.
fn keystream_block(key: &[u8; 32], counter: u64) -> [u8; 32] {
    let mut input = Vec::with_capacity(40);
    input.extend_from_slice(key);
    input.extend_from_slice(&counter.to_le_bytes());
    *blake3::hash(&input).as_bytes()
}

/// Compute a MAC (message authentication code) over data using a BLAKE3 keyed hash.
fn compute_mac(key: &[u8; 32], data: &[u8]) -> [u8; 32] {
    *blake3::keyed_hash(key, data).as_bytes()
}

/// XOR `data` against the BLAKE3 keystream derived from `key`.
///
/// This function is its own inverse: encrypting and decrypting are the same
/// XOR operation with the same keystream.
fn xor_keystream(data: &[u8], key: &[u8; 32]) -> Vec<u8> {
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

/// Encrypt plaintext into the APR encryption envelope: `[ciphertext || MAC]`.
///
/// 1. XOR plaintext with BLAKE3 keystream to produce ciphertext.
/// 2. Compute a BLAKE3 keyed-hash MAC over the ciphertext.
/// 3. Append the 32-byte MAC.
fn encrypt_payload(plaintext: &[u8], key: &[u8; 32]) -> Vec<u8> {
    let ciphertext = xor_keystream(plaintext, key);
    let mac = compute_mac(key, &ciphertext);
    let mut envelope = ciphertext;
    envelope.extend_from_slice(&mac);
    envelope
}

/// Decrypt an APR encryption envelope, verifying the MAC first.
///
/// Returns `Err` if the envelope is too short or the MAC does not match
/// (wrong password or tampered data).
fn decrypt_payload(envelope: &[u8], key: &[u8; 32]) -> Result<Vec<u8>> {
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
fn demo_model_bytes(seed: u64, size: usize) -> Vec<u8> {
    (0..size)
        .map(|i| {
            let mut h = DefaultHasher::new();
            seed.hash(&mut h);
            i.hash(&mut h);
            (h.finish() & 0xFF) as u8
        })
        .collect()
}

fn run_demo() -> Result<()> {
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

fn run_decrypt(config: &DecryptConfig) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_key_deterministic() {
        let k1 = derive_key("password123");
        let k2 = derive_key("password123");
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_derive_key_different_passwords() {
        let k1 = derive_key("alpha");
        let k2 = derive_key("bravo");
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let key = derive_key("test-secret");
        let plaintext = b"Hello, APR model weights!";
        let encrypted = encrypt_payload(plaintext, &key);
        let decrypted = decrypt_payload(&encrypted, &key).expect("decrypt should succeed");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_wrong_password_fails() {
        let key = derive_key("correct-password");
        let wrong = derive_key("wrong-password");
        let plaintext = b"secret model data";
        let encrypted = encrypt_payload(plaintext, &key);
        let result = decrypt_payload(&encrypted, &wrong);
        assert!(result.is_err());
    }

    #[test]
    fn test_envelope_too_short() {
        let key = derive_key("key");
        let short = vec![0u8; MAC_SIZE - 1];
        let result = decrypt_payload(&short, &key);
        assert!(result.is_err());
    }

    #[test]
    fn test_tampered_ciphertext() {
        let key = derive_key("integrity-check");
        let plaintext = b"tamper test payload";
        let mut encrypted = encrypt_payload(plaintext, &key);
        // Flip a bit in the ciphertext (not the MAC)
        if !encrypted.is_empty() {
            encrypted[0] ^= 0x01;
        }
        let result = decrypt_payload(&encrypted, &key);
        assert!(result.is_err());
    }

    #[test]
    fn test_ciphertext_differs_from_plaintext() {
        let key = derive_key("diff-check");
        let plaintext = vec![0xAA; 256];
        let encrypted = encrypt_payload(&plaintext, &key);
        // Ciphertext portion (everything except trailing MAC)
        let ciphertext = &encrypted[..encrypted.len() - MAC_SIZE];
        assert_ne!(ciphertext, &plaintext[..]);
    }

    #[test]
    fn test_empty_payload() {
        let key = derive_key("empty");
        let plaintext: &[u8] = b"";
        let encrypted = encrypt_payload(plaintext, &key);
        assert_eq!(encrypted.len(), MAC_SIZE);
        let decrypted = decrypt_payload(&encrypted, &key).expect("empty decrypt");
        assert!(decrypted.is_empty());
    }

    #[test]
    fn test_clap_parse_demo() {
        let config = DecryptConfig::parse_from(["apr-decrypt", "--demo"]);
        assert!(config.demo);
    }

    #[test]
    fn test_clap_parse_full() {
        let config = DecryptConfig::parse_from([
            "apr-decrypt",
            "model.apr.enc",
            "-p",
            "secret",
            "-o",
            "model.apr",
        ]);
        assert_eq!(config.input, Some("model.apr.enc".to_string()));
        assert_eq!(config.password, Some("secret".to_string()));
        assert_eq!(config.output, Some("model.apr".to_string()));
    }

    #[test]
    fn test_clap_parse_unknown_flag() {
        let result = DecryptConfig::try_parse_from(["apr-decrypt", "--bogus"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_demo_model_bytes_deterministic() {
        let a = demo_model_bytes(42, 128);
        let b = demo_model_bytes(42, 128);
        assert_eq!(a, b);
    }

    #[test]
    fn test_demo_model_bytes_different_seeds() {
        let a = demo_model_bytes(1, 128);
        let b = demo_model_bytes(2, 128);
        assert_ne!(a, b);
    }

    #[test]
    fn test_keystream_block_deterministic() {
        let key = derive_key("ks-test");
        let b1 = keystream_block(&key, 0);
        let b2 = keystream_block(&key, 0);
        assert_eq!(b1, b2);
    }

    #[test]
    fn test_keystream_block_varies_with_counter() {
        let key = derive_key("ks-test");
        let b0 = keystream_block(&key, 0);
        let b1 = keystream_block(&key, 1);
        assert_ne!(b0, b1);
    }

    #[test]
    fn test_mac_deterministic() {
        let key = derive_key("mac-test");
        let data = b"some data";
        let m1 = compute_mac(&key, data);
        let m2 = compute_mac(&key, data);
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_large_payload_roundtrip() {
        let key = derive_key("big-model");
        let plaintext: Vec<u8> = (0u32..10_000).map(|i| (i % 256) as u8).collect();
        let encrypted = encrypt_payload(&plaintext, &key);
        assert_eq!(encrypted.len(), plaintext.len() + MAC_SIZE);
        let decrypted = decrypt_payload(&encrypted, &key).expect("large decrypt");
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn test_run_demo() {
        assert!(run_demo().is_ok());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_encrypt_decrypt_roundtrip(
            plaintext in proptest::collection::vec(any::<u8>(), 0..4096),
            password in "[a-zA-Z0-9]{1,32}",
        ) {
            let key = derive_key(&password);
            let encrypted = encrypt_payload(&plaintext, &key);
            prop_assert_eq!(encrypted.len(), plaintext.len() + MAC_SIZE);
            let decrypted = decrypt_payload(&encrypted, &key).expect("roundtrip");
            prop_assert_eq!(decrypted, plaintext);
        }

        #[test]
        fn prop_wrong_password_rejects(
            plaintext in proptest::collection::vec(any::<u8>(), 1..512),
            pw1 in "[a-z]{4,16}",
            pw2 in "[A-Z]{4,16}",
        ) {
            // pw1 is all-lowercase, pw2 is all-uppercase, so they always differ
            let k1 = derive_key(&pw1);
            let k2 = derive_key(&pw2);
            let encrypted = encrypt_payload(&plaintext, &k1);
            let result = decrypt_payload(&encrypted, &k2);
            prop_assert!(result.is_err());
        }

        #[test]
        fn prop_tamper_detected(
            plaintext in proptest::collection::vec(any::<u8>(), 1..512),
            password in "[a-zA-Z0-9]{4,16}",
            flip_pos in 0usize..512,
        ) {
            let key = derive_key(&password);
            let mut encrypted = encrypt_payload(&plaintext, &key);
            let idx = flip_pos % encrypted.len();
            encrypted[idx] ^= 0x01;
            // Tampered data should either fail MAC or produce wrong plaintext
            // (in practice, MAC check catches it)
            let result = decrypt_payload(&encrypted, &key);
            prop_assert!(result.is_err());
        }

        #[test]
        fn prop_ciphertext_differs(
            plaintext in proptest::collection::vec(any::<u8>(), 32..256),
            password in "[a-zA-Z0-9]{4,16}",
        ) {
            let key = derive_key(&password);
            let encrypted = encrypt_payload(&plaintext, &key);
            let ciphertext = &encrypted[..plaintext.len()];
            // XOR keystream should change the data (except in astronomically
            // unlikely case of all-zero keystream)
            prop_assert_ne!(ciphertext, &plaintext[..]);
        }
    }
}
