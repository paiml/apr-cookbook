//! # Recipe: Encrypted + Signed Bundle
//!
//! **Category**: bundling
//! **CLI Equivalent**: `apr encrypt model.apr --sign --key k --out model.enc.sig`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example encrypt_signed` exits 0
//! 2. [x] `cargo test --example encrypt_signed` passes
//! 3. [x] Deterministic output (same seed -> same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr encrypt --sign` in-process (no shell-out)
//! 10. [x] Unit tests cover valid sig, tamper detection, decrypt fail, order
//!
//! ## Learning Objective
//! Combines symmetric encryption with an ed25519 signature so the consumer can
//! both verify authenticity (signature over ciphertext) and recover the
//! plaintext (XOR-derived keystream). Demonstrates the sign-over-ciphertext
//! convention that prevents attackers from swapping encrypted payloads.
//!
//! ## Run Command
//! ```bash
//! cargo run --example encrypt_signed
//! ```
//!
//! ## References
//! - Percival, C. (2009). *Stronger Key Derivation via Sequential Memory-Hard Functions*. BSDCan.

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey};
use rand::{RngCore, SeedableRng};
use serde_json::json;

// ---------------------------------------------------------------------------
// Crypto (pedagogical XOR + ed25519 signature)
// ---------------------------------------------------------------------------

fn derive_stream(key: &[u8], nonce: &[u8], out_len: usize) -> Vec<u8> {
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
    let stream = derive_stream(key, nonce, data.len());
    data.iter().zip(stream.iter()).map(|(d, k)| d ^ k).collect()
}

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SignedCipher {
    ciphertext: Vec<u8>,
    signature: Vec<u8>,
    public_key: VerifyingKey,
}

// ---------------------------------------------------------------------------
// Logic
// ---------------------------------------------------------------------------

fn encrypt_and_sign(
    plaintext: &[u8],
    enc_key: &[u8],
    nonce: &[u8],
    signing: &SigningKey,
) -> SignedCipher {
    let ciphertext = xor_cipher(plaintext, enc_key, nonce);
    let signature = signing.sign(&ciphertext);
    SignedCipher {
        ciphertext,
        signature: signature.to_bytes().to_vec(),
        public_key: signing.verifying_key(),
    }
}

fn verify_and_decrypt(sc: &SignedCipher, enc_key: &[u8], nonce: &[u8]) -> Result<Vec<u8>> {
    if sc.signature.len() != 64 {
        return Err(CookbookError::invalid_format(
            "signature length must be 64 bytes",
        ));
    }
    let mut sig_arr = [0_u8; 64];
    sig_arr.copy_from_slice(&sc.signature);
    let signature = ed25519_dalek::Signature::from_bytes(&sig_arr);

    sc.public_key
        .verify(&sc.ciphertext, &signature)
        .map_err(|e| {
            CookbookError::invalid_format(format!("signature verification failed: {e}"))
        })?;

    Ok(xor_cipher(&sc.ciphertext, enc_key, nonce))
}

fn build_signing_key(seed: u64) -> SigningKey {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut bytes = [0_u8; 32];
    rng.fill_bytes(&mut bytes);
    SigningKey::from_bytes(&bytes)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("encrypt_signed")?;
    println!("=== Recipe: {} ===", ctx.name());

    let seed = hash_name_to_seed("encrypt-signed");
    let plaintext = generate_model_payload(seed, 64);
    let enc_key = b"encryption-key-0001";
    let nonce = b"nonce-0001";
    let signing_key = build_signing_key(seed);

    // Happy path: encrypt + sign then verify + decrypt.
    let sc = encrypt_and_sign(&plaintext, enc_key, nonce, &signing_key);
    let recovered = verify_and_decrypt(&sc, enc_key, nonce)?;
    assert_eq!(recovered, plaintext);
    println!("OK: sign+encrypt roundtrip, {} bytes", recovered.len());

    // Write artifact.
    let artifact_path = ctx.path("model.enc.sig");
    let mut bytes = Vec::with_capacity(sc.ciphertext.len() + sc.signature.len() + 32);
    bytes.extend_from_slice(&sc.signature); // first 64 bytes
    bytes.extend_from_slice(sc.public_key.as_bytes()); // next 32 bytes
    bytes.extend_from_slice(&sc.ciphertext);
    std::fs::write(&artifact_path, &bytes)?;
    println!("Wrote {} ({} bytes)", artifact_path.display(), bytes.len());

    // Tamper detection.
    let mut tampered = sc.clone();
    tampered.ciphertext[0] ^= 0xFF;
    let tamper_result = verify_and_decrypt(&tampered, enc_key, nonce);
    assert!(
        tamper_result.is_err(),
        "tampered ciphertext must fail verification"
    );
    println!("OK: tampered ciphertext rejected");

    let out = json!({
        "recipe": ctx.name(),
        "plaintext_bytes": plaintext.len(),
        "ciphertext_bytes": sc.ciphertext.len(),
        "signature_bytes": sc.signature.len(),
        "pubkey_bytes": 32,
        "artifact_bytes": bytes.len(),
        "tamper_rejected": true,
    });
    let out_path = ctx.path("encrypt-signed.json");
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

    fn make_key(seed: u64) -> SigningKey {
        build_signing_key(seed)
    }

    #[test]
    fn test_happy_roundtrip() {
        let sk = make_key(1);
        let sc = encrypt_and_sign(b"hello", b"k", b"n", &sk);
        let pt = verify_and_decrypt(&sc, b"k", b"n").expect("ok");
        assert_eq!(pt, b"hello");
    }

    #[test]
    fn test_tamper_detected() {
        let sk = make_key(2);
        let mut sc = encrypt_and_sign(b"hello", b"k", b"n", &sk);
        sc.ciphertext[0] ^= 0x01;
        assert!(verify_and_decrypt(&sc, b"k", b"n").is_err());
    }

    #[test]
    fn test_wrong_enc_key_yields_garbage_but_passes_sig() {
        // Signature verifies because it's over ciphertext, so decrypt runs but
        // returns garbage (no MAC on plaintext by design here).
        let sk = make_key(3);
        let sc = encrypt_and_sign(b"hello world", b"good-key", b"n", &sk);
        let bad = verify_and_decrypt(&sc, b"wrong-key", b"n").expect("sig ok");
        assert_ne!(bad, b"hello world");
    }

    #[test]
    fn test_signature_length_enforced() {
        let sk = make_key(4);
        let mut sc = encrypt_and_sign(b"hi", b"k", b"n", &sk);
        sc.signature.truncate(32);
        assert!(verify_and_decrypt(&sc, b"k", b"n").is_err());
    }

    #[test]
    fn test_signature_is_64_bytes() {
        let sk = make_key(5);
        let sc = encrypt_and_sign(b"hi", b"k", b"n", &sk);
        assert_eq!(sc.signature.len(), 64);
    }

    #[test]
    fn test_deterministic_signing_key() {
        let sk1 = build_signing_key(42);
        let sk2 = build_signing_key(42);
        assert_eq!(sk1.to_bytes(), sk2.to_bytes());
    }
}
