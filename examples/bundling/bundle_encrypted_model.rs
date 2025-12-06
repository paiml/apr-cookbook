//! Encrypted model bundling example.
//!
//! This example demonstrates loading encrypted APR models with password-based
//! decryption using Argon2id key derivation and AES-256-GCM encryption.
//!
//! # Run
//!
//! ```bash
//! cargo run --example bundle_encrypted_model --features encryption
//! ```
//!
//! # Security Features
//!
//! - **AES-256-GCM**: Authenticated encryption with associated data (AEAD)
//! - **Argon2id**: Memory-hard key derivation (prevents GPU brute-force)
//! - **Random nonce**: Unique per encryption (prevents IV reuse attacks)
//!
//! # Use Cases
//!
//! - Protecting proprietary models in distribution
//! - Compliance with data protection regulations
//! - Secure model deployment in untrusted environments

use apr_cookbook::Result;
#[cfg(feature = "encryption")]
use aprender::format::{
    load_encrypted, load_from_bytes_encrypted, save_encrypted, ModelType, SaveOptions,
};
use serde::{Deserialize, Serialize};

/// Example model for encryption demonstration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct SentimentClassifier {
    /// Vocabulary size
    vocab_size: usize,
    /// Embedding dimension
    embed_dim: usize,
    /// Word embeddings (flattened)
    embeddings: Vec<f32>,
    /// Classification weights
    weights: Vec<f32>,
    /// Classification bias
    bias: f32,
}

impl SentimentClassifier {
    /// Create a mock classifier for demonstration
    fn mock() -> Self {
        let vocab_size = 1000;
        let embed_dim = 64;

        // Generate reproducible random weights
        let mut seed: u64 = 12345;
        let mut next_random = || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5
        };

        let embeddings: Vec<f32> = (0..vocab_size * embed_dim).map(|_| next_random()).collect();
        let weights: Vec<f32> = (0..embed_dim).map(|_| next_random()).collect();
        let bias = next_random();

        Self {
            vocab_size,
            embed_dim,
            embeddings,
            weights,
            bias,
        }
    }
}

#[cfg(feature = "encryption")]
mod demo {
    #[allow(clippy::wildcard_imports)]
    use super::*;
    use std::path::Path;

    pub(super) fn print_model_info(model: &SentimentClassifier) {
        println!("Created sentiment classifier:");
        println!("  Vocabulary size: {}", model.vocab_size);
        println!("  Embedding dimension: {}", model.embed_dim);
        println!(
            "  Total parameters: {}",
            model.embeddings.len() + model.weights.len() + 1
        );
    }

    pub(super) fn print_size_comparison(encrypted_path: &Path, unencrypted_path: &Path) {
        let encrypted_size = std::fs::metadata(encrypted_path)
            .map(|m| m.len())
            .unwrap_or(0);
        let unencrypted_size = std::fs::metadata(unencrypted_path)
            .map(|m| m.len())
            .unwrap_or(0);

        println!("File sizes:");
        println!("  Unencrypted: {} bytes", unencrypted_size);
        println!(
            "  Encrypted:   {} bytes (+{} bytes overhead)",
            encrypted_size,
            encrypted_size.saturating_sub(unencrypted_size)
        );
    }

    pub(super) fn print_wrong_password_result(
        result: std::result::Result<SentimentClassifier, aprender::AprenderError>,
    ) {
        match result {
            Ok(_) => println!("  ✗ Unexpected success with wrong password!"),
            Err(e) => {
                let err_msg = e.to_string();
                if err_msg.contains("ecrypt") || err_msg.contains("auth") {
                    println!("  ✓ Correctly rejected wrong password");
                } else {
                    println!("  ✓ Decryption failed as expected: {}", err_msg);
                }
            }
        }
    }

    pub(super) fn print_usage_example() {
        println!("\n=== Production Usage ===");
        println!("```rust");
        println!("// Embed encrypted model at compile time");
        println!("const MODEL: &[u8] = include_bytes!(\"model.apr.enc\");");
        println!();
        println!("fn load_model(password: &str) -> Result<MyModel> {{");
        println!("    load_from_bytes_encrypted(MODEL, ModelType::Custom, password)");
        println!("}}");
        println!("```");
    }
}

#[cfg(feature = "encryption")]
fn main() -> Result<()> {
    use tempfile::tempdir;

    println!("=== APR Cookbook: Encrypted Model Bundling ===\n");

    let model = SentimentClassifier::mock();
    demo::print_model_info(&model);

    let dir = tempdir().map_err(apr_cookbook::CookbookError::Io)?;
    let encrypted_path = dir.path().join("sentiment.apr.enc");
    let unencrypted_path = dir.path().join("sentiment.apr");
    let password = "demo_password_123!";

    // Save models
    println!("\nSaving encrypted model...");
    save_encrypted(
        &model,
        ModelType::Custom,
        &encrypted_path,
        SaveOptions::default()
            .with_name("sentiment-classifier")
            .with_description("Encrypted sentiment classification model"),
        password,
    )
    .map_err(|e| apr_cookbook::CookbookError::Aprender(e.to_string()))?;

    aprender::format::save(
        &model,
        ModelType::Custom,
        &unencrypted_path,
        SaveOptions::default().with_name("sentiment-classifier"),
    )
    .map_err(|e| apr_cookbook::CookbookError::Aprender(e.to_string()))?;

    demo::print_size_comparison(&encrypted_path, &unencrypted_path);

    // Inspect
    println!("\nInspecting encrypted model...");
    let info = aprender::format::inspect(&encrypted_path)
        .map_err(|e| apr_cookbook::CookbookError::Aprender(e.to_string()))?;
    println!("  Name: {:?}", info.metadata.model_name);
    println!("  Encrypted: {}", info.encrypted);
    println!("  Signed: {}", info.signed);

    // Load and verify
    println!("\nLoading encrypted model with correct password...");
    let loaded: SentimentClassifier = load_encrypted(&encrypted_path, ModelType::Custom, password)
        .map_err(|e| apr_cookbook::CookbookError::Aprender(e.to_string()))?;
    assert_eq!(model, loaded, "Model mismatch after decryption!");
    println!("  ✓ Model loaded successfully");
    println!("  ✓ Decryption verified (model matches original)");

    // From bytes
    println!("\nDemonstrating include_bytes!() pattern...");
    let encrypted_bytes =
        std::fs::read(&encrypted_path).map_err(apr_cookbook::CookbookError::Io)?;
    println!(
        "  Read {} bytes (simulating include_bytes!)",
        encrypted_bytes.len()
    );

    let from_bytes: SentimentClassifier =
        load_from_bytes_encrypted(&encrypted_bytes, ModelType::Custom, password)
            .map_err(|e| apr_cookbook::CookbookError::Aprender(e.to_string()))?;
    assert_eq!(model, from_bytes, "Model mismatch from bytes!");
    println!("  ✓ Loaded from bytes successfully");

    // Wrong password
    println!("\nTesting wrong password...");
    let wrong_result = load_encrypted(&encrypted_path, ModelType::Custom, "wrong_password");
    demo::print_wrong_password_result(wrong_result);

    println!("\n[SUCCESS] Encrypted model demonstration complete!");
    demo::print_usage_example();

    Ok(())
}

#[cfg(not(feature = "encryption"))]
fn main() {
    println!("=== APR Cookbook: Encrypted Model Bundling ===\n");
    println!("This example requires the 'encryption' feature.");
    println!();
    println!("Run with:");
    println!("  cargo run --example bundle_encrypted_model --features encryption");
    println!();
    println!("The encryption feature enables:");
    println!("  - AES-256-GCM authenticated encryption");
    println!("  - Argon2id key derivation");
    println!("  - X25519 recipient-based encryption");
}

#[cfg(all(test, feature = "encryption"))]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_encrypted_roundtrip() {
        let model = SentimentClassifier::mock();
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_encrypted.apr");
        let password = "test_password";

        save_encrypted(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            password,
        )
        .unwrap();

        let loaded: SentimentClassifier =
            load_encrypted(&path, ModelType::Custom, password).unwrap();

        assert_eq!(model, loaded);
    }

    #[test]
    fn test_encrypted_from_bytes() {
        let model = SentimentClassifier::mock();
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_bytes.apr");
        let password = "byte_password";

        save_encrypted(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            password,
        )
        .unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let loaded: SentimentClassifier =
            load_from_bytes_encrypted(&bytes, ModelType::Custom, password).unwrap();

        assert_eq!(model, loaded);
    }

    #[test]
    fn test_wrong_password_fails() {
        let model = SentimentClassifier::mock();
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_wrong_pw.apr");
        let password = "correct_password";

        save_encrypted(
            &model,
            ModelType::Custom,
            &path,
            SaveOptions::default(),
            password,
        )
        .unwrap();

        let result: std::result::Result<SentimentClassifier, _> =
            load_encrypted(&path, ModelType::Custom, "wrong_password");

        assert!(result.is_err());
    }
}
