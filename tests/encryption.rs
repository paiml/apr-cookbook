//! AES-256-GCM end-to-end tests.
//!
//! Witnesses the cookbook's binding of `aes256-gcm-decrypt-v1.yaml` obligations:
//! - `roundtrip` — `save_encrypted` followed by `load_encrypted` recovers
//!   the original model byte-for-byte.
//! - `authenticity` — tampering with a single byte of the encrypted payload
//!   causes `load_encrypted` to fail; AES-256-GCM's GHASH tag verification
//!   rejects any non-identical ciphertext.
//!
//! These tests are gated behind the `encryption` feature (the same gate the
//! `bundle_encrypted_model` recipe uses). CI runs `cargo test --all-features`,
//! so they execute on every PR.
//!
//! ## What this does NOT prove
//! - The `decrypt_latency` < 5ms claim remains `pending` because the cookbook's
//!   end-to-end `load_encrypted` path includes Argon2id KDF (intentionally slow
//!   ~100ms). Measuring just the post-KDF AES-GCM step would require bench-
//!   slicing inside aprender-core.
//! - Formal proof of AES-256-GCM's IND-CCA2 security. That lives in the crypto
//!   literature, not the cookbook.

#![cfg(feature = "encryption")]

use aprender::format::{load_encrypted, save_encrypted, ModelType, SaveOptions};
use serde::{Deserialize, Serialize};
use std::fs;
use tempfile::tempdir;

/// Minimal serializable model for the test.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct TinyModel {
    name: String,
    weights: Vec<f32>,
    bias: f32,
}

fn sample_model() -> TinyModel {
    TinyModel {
        name: "tamper-test".into(),
        weights: vec![0.1, -0.2, 0.3, -0.4, 0.5],
        bias: 0.01,
    }
}

/// Witnesses `aes256-gcm-decrypt-v1.yaml::roundtrip`.
#[test]
fn aes_gcm_roundtrip_preserves_model() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("roundtrip.apr.enc");
    let original = sample_model();
    let password = "correct_horse_battery_staple";

    save_encrypted(
        &original,
        ModelType::Custom,
        &path,
        SaveOptions::default().with_name("tamper-test"),
        password,
    )
    .expect("save_encrypted");

    let loaded: TinyModel =
        load_encrypted(&path, ModelType::Custom, password).expect("load_encrypted");

    assert_eq!(loaded, original);
}

/// Witnesses `aes256-gcm-decrypt-v1.yaml::authenticity`.
///
/// Flips one byte in the middle of the encrypted payload. AES-256-GCM's GHASH
/// authentication tag is computed over the entire ciphertext; any single-byte
/// modification causes tag verification to fail and `load_encrypted` to error.
#[test]
fn aes_gcm_tamper_detection_single_byte_flip() {
    let dir = tempdir().expect("tempdir");
    let good_path = dir.path().join("good.apr.enc");
    let bad_path = dir.path().join("tampered.apr.enc");
    let password = "correct_horse_battery_staple";

    save_encrypted(
        &sample_model(),
        ModelType::Custom,
        &good_path,
        SaveOptions::default().with_name("tamper-test"),
        password,
    )
    .expect("save_encrypted");

    // Flip one byte near the middle of the file — deep in the ciphertext,
    // past the APR header and metadata, where GHASH will definitely catch it.
    let mut bytes = fs::read(&good_path).expect("read good");
    let n = bytes.len();
    assert!(
        n > 128,
        "encrypted payload too small for a meaningful tamper test"
    );
    let tamper_offset = n / 2;
    bytes[tamper_offset] ^= 0xAA;
    fs::write(&bad_path, &bytes).expect("write tampered");

    let result: Result<TinyModel, _> = load_encrypted(&bad_path, ModelType::Custom, password);

    assert!(
        result.is_err(),
        "AES-GCM tag verification should reject tampered ciphertext; \
         load_encrypted returned Ok — possible authentication bypass!"
    );
}

/// Witnesses `aes256-gcm-decrypt-v1.yaml::authenticity` from the wrong-key
/// direction. Incorrect password → wrong Argon2id-derived key → AES-GCM
/// decrypt fails with authentication error.
#[test]
fn aes_gcm_wrong_password_rejected() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("wrong-pw.apr.enc");

    save_encrypted(
        &sample_model(),
        ModelType::Custom,
        &path,
        SaveOptions::default().with_name("tamper-test"),
        "correct-password",
    )
    .expect("save_encrypted");

    let result: Result<TinyModel, _> = load_encrypted(&path, ModelType::Custom, "wrong-password");

    assert!(
        result.is_err(),
        "wrong password should yield decrypt failure, not successful decode"
    );
}
