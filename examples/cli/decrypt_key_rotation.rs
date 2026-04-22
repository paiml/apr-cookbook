//! # Recipe: Key Rotation During Decrypt
//!
//! **Category**: cli
//! **CLI Equivalent**: `apr decrypt model.apr --old-key k1 --new-key k2 --rotate`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example decrypt_key_rotation` exits 0
//! 2. [x] `cargo test --example decrypt_key_rotation` passes
//! 3. [x] Deterministic output (same seed -> same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr decrypt` key rotation in-process (no shell-out)
//! 10. [x] Unit tests cover correct key, wrong key, ciphertext differ, roundtrip
//!
//! ## Learning Objective
//! Simulates decrypt-then-re-encrypt key rotation using a pedagogical XOR-based
//! stream cipher with blake3 for KDF. Demonstrates the rotation invariant:
//! plaintext is identical across old and new keys, but the ciphertexts differ.
//! Models the control flow `apr decrypt --rotate` follows.
//!
//! ## Run Command
//! ```bash
//! cargo run --example decrypt_key_rotation
//! ```
//!
//! ## References
//! - Percival, C. (2009). *Stronger Key Derivation via Sequential Memory-Hard Functions*. BSDCan.

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

// ---------------------------------------------------------------------------
// Crypto (pedagogical: blake3 KDF + XOR keystream)
// ---------------------------------------------------------------------------

fn derive_stream(key: &[u8], nonce: &[u8], out_len: usize) -> Vec<u8> {
    // Expand the key+nonce into a keystream of desired length via repeated
    // blake3 hashing over a counter. This is NOT a real AEAD -- it is a pure-
    // Rust demo using only deps already in Cargo.toml.
    let mut out = Vec::with_capacity(out_len);
    let mut counter: u64 = 0;
    while out.len() < out_len {
        let mut hasher = blake3::Hasher::new();
        hasher.update(key);
        hasher.update(nonce);
        hasher.update(&counter.to_le_bytes());
        let block = hasher.finalize();
        out.extend_from_slice(block.as_bytes());
        counter += 1;
    }
    out.truncate(out_len);
    out
}

fn xor_cipher(data: &[u8], key: &[u8], nonce: &[u8]) -> Vec<u8> {
    let keystream = derive_stream(key, nonce, data.len());
    data.iter()
        .zip(keystream.iter())
        .map(|(d, k)| d ^ k)
        .collect()
}

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct RotationReport {
    plaintext_len: usize,
    old_cipher_len: usize,
    new_cipher_len: usize,
    ciphertexts_differ: bool,
    plaintexts_match: bool,
}

// ---------------------------------------------------------------------------
// Logic
// ---------------------------------------------------------------------------

fn rotate_key(
    old_cipher: &[u8],
    old_key: &[u8],
    new_key: &[u8],
    old_nonce: &[u8],
    new_nonce: &[u8],
) -> Result<(Vec<u8>, Vec<u8>)> {
    if old_cipher.is_empty() {
        return Err(CookbookError::invalid_format("empty ciphertext"));
    }
    // Decrypt with old key.
    let plaintext = xor_cipher(old_cipher, old_key, old_nonce);
    // Encrypt with new key under new nonce.
    let new_cipher = xor_cipher(&plaintext, new_key, new_nonce);
    Ok((plaintext, new_cipher))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("decrypt_key_rotation")?;
    println!("=== Recipe: {} ===", ctx.name());

    let seed = hash_name_to_seed("decrypt-key-rotation");
    let model_bytes = generate_model_payload(seed, 128);

    let old_key = b"original-master-key-0123";
    let new_key = b"rotated-master-key-000456";
    let old_nonce = b"nonce-old-001";
    let new_nonce = b"nonce-new-002";

    // Encrypt initially.
    let old_cipher = xor_cipher(&model_bytes, old_key, old_nonce);
    let old_path = ctx.path("model.encrypted.v1");
    std::fs::write(&old_path, &old_cipher)?;
    println!("Encrypted under old key: {} bytes", old_cipher.len());

    // Rotate.
    let (plaintext, new_cipher) = rotate_key(&old_cipher, old_key, new_key, old_nonce, new_nonce)?;
    let new_path = ctx.path("model.encrypted.v2");
    std::fs::write(&new_path, &new_cipher)?;
    println!("Re-encrypted under new key: {} bytes", new_cipher.len());

    // Sanity: decrypt new ciphertext and compare to original plaintext.
    let decrypted_again = xor_cipher(&new_cipher, new_key, new_nonce);
    let plaintexts_match = decrypted_again == model_bytes;
    let ciphertexts_differ = old_cipher != new_cipher;

    let report = RotationReport {
        plaintext_len: plaintext.len(),
        old_cipher_len: old_cipher.len(),
        new_cipher_len: new_cipher.len(),
        ciphertexts_differ,
        plaintexts_match,
    };

    println!("\n--- Rotation Report ---");
    println!("Plaintext len:        {}", report.plaintext_len);
    println!("Old ciphertext len:   {}", report.old_cipher_len);
    println!("New ciphertext len:   {}", report.new_cipher_len);
    println!("Ciphertexts differ:   {}", report.ciphertexts_differ);
    println!("Plaintexts match:     {}", report.plaintexts_match);

    assert!(report.plaintexts_match, "rotation must preserve plaintext");
    assert!(report.ciphertexts_differ, "new cipher must differ from old");

    let out = json!({
        "recipe": ctx.name(),
        "plaintext_len": report.plaintext_len,
        "old_cipher_len": report.old_cipher_len,
        "new_cipher_len": report.new_cipher_len,
        "ciphertexts_differ": report.ciphertexts_differ,
        "plaintexts_match": report.plaintexts_match,
    });
    let out_path = ctx.path("rotation.json");
    let out_bytes =
        serde_json::to_vec_pretty(&out).map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out_path, out_bytes)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_roundtrips_with_same_key() {
        let plaintext = b"hello world".to_vec();
        let key = b"key";
        let nonce = b"nonce";
        let cipher = xor_cipher(&plaintext, key, nonce);
        let back = xor_cipher(&cipher, key, nonce);
        assert_eq!(back, plaintext);
    }

    #[test]
    fn test_wrong_key_does_not_roundtrip() {
        let plaintext = b"secret".to_vec();
        let cipher = xor_cipher(&plaintext, b"k1", b"n");
        let back = xor_cipher(&cipher, b"k2", b"n");
        assert_ne!(back, plaintext);
    }

    #[test]
    fn test_rotate_preserves_plaintext() {
        let plaintext = b"hello rotation".to_vec();
        let old_c = xor_cipher(&plaintext, b"a", b"n1");
        let (pt, new_c) = rotate_key(&old_c, b"a", b"b", b"n1", b"n2").expect("rotate");
        assert_eq!(pt, plaintext);
        let back = xor_cipher(&new_c, b"b", b"n2");
        assert_eq!(back, plaintext);
    }

    #[test]
    fn test_rotate_ciphertexts_differ() {
        let plaintext = b"content".to_vec();
        let old_c = xor_cipher(&plaintext, b"a", b"n1");
        let (_, new_c) = rotate_key(&old_c, b"a", b"b", b"n1", b"n2").expect("rotate");
        assert_ne!(old_c, new_c);
    }

    #[test]
    fn test_rotate_empty_errors() {
        let err = rotate_key(&[], b"a", b"b", b"n1", b"n2");
        assert!(err.is_err());
    }

    #[test]
    fn test_keystream_deterministic() {
        let s1 = derive_stream(b"key", b"nonce", 64);
        let s2 = derive_stream(b"key", b"nonce", 64);
        assert_eq!(s1, s2);
    }
}
