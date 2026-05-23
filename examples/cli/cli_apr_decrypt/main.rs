#![allow(unused_imports)]
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

mod helpers;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;

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
