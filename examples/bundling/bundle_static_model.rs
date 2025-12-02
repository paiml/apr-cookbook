//! Statically embedded model inference.
//!
//! This example demonstrates how to embed an ML model directly into
//! a Rust binary using `include_bytes!()`, enabling zero-dependency
//! deployment.
//!
//! # Run
//!
//! ```bash
//! cargo run --example bundle_static_model
//! ```
//!
//! # Philosophy (Muda Elimination)
//!
//! By embedding the model at compile time, we eliminate:
//! - External file dependencies
//! - Runtime file I/O errors
//! - Deployment complexity

use apr_cookbook::bundle::{BundledModel, ModelBundle};
use apr_cookbook::Result;

/// Create a sample model for demonstration.
///
/// In production, you would use:
/// ```ignore
/// const MODEL_BYTES: &[u8] = include_bytes!("../models/sentiment.apr");
/// ```
fn create_sample_model() -> Vec<u8> {
    ModelBundle::new()
        .with_name("sentiment-classifier")
        .with_description("Demo sentiment classifier for cookbook")
        .with_payload(vec![0u8; 1024]) // Simulated weights
        .build()
}

fn main() -> Result<()> {
    println!("=== APR Cookbook: Static Model Bundling ===\n");

    // In production: include_bytes!("../models/sentiment.apr")
    let model_bytes = create_sample_model();

    // Load the bundled model
    let model = BundledModel::from_bytes(&model_bytes)?;

    // Display model information
    println!("Model Information:");
    println!("  Name: {}", model.name());
    println!("  Size: {} bytes", model.size());
    println!("  Version: {}.{}", model.version().0, model.version().1);
    println!("  Compressed: {}", model.is_compressed());
    println!("  Encrypted: {}", model.is_encrypted());
    println!("  Signed: {}", model.is_signed());

    println!("\n[SUCCESS] Model loaded from embedded bytes.");
    println!("          Zero external files required!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_model_creation() {
        let model_bytes = create_sample_model();
        assert!(!model_bytes.is_empty());
        assert!(model_bytes.len() >= 32); // Minimum header size
    }

    #[test]
    fn test_sample_model_loads() {
        let model_bytes = create_sample_model();
        let model = BundledModel::from_bytes(&model_bytes);
        assert!(model.is_ok());
    }
}
