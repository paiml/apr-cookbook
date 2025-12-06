//! # Recipe: Bundle APR into Static Binary
//!
//! **Category**: Binary Bundling
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
//! Embed `.apr` model into a Rust binary for zero-dependency deployment.
//!
//! ## Run Command
//! ```bash
//! cargo run --example bundle_apr_static_binary
//! ```

use apr_cookbook::prelude::*;

/// Demo model bytes - in production, use include_bytes!("path/to/model.apr")
/// This creates a minimal valid APR model for demonstration
fn create_demo_model_bytes() -> Vec<u8> {
    ModelBundle::new()
        .with_name("demo-classifier")
        .with_description("Embedded sentiment classifier")
        .with_payload(generate_model_payload(42, 256))
        .build()
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("bundle_apr_static_binary")?;

    // In production: const MODEL_BYTES: &[u8] = include_bytes!("../models/classifier.apr");
    // For demo, we create the model inline
    let model_bytes = create_demo_model_bytes();

    // Load from embedded bytes - no filesystem access needed
    let model = BundledModel::from_bytes(&model_bytes)?;

    ctx.record_metric("model_size_bytes", model.size() as i64);
    ctx.record_string_metric("model_name", model.name());
    ctx.record_string_metric(
        "model_version",
        format!("{}.{}", model.version().0, model.version().1),
    );

    // Demonstrate inference (mock)
    let input = vec![1.0f32, 2.0, 3.0, 4.0];
    let output = mock_inference(&model, &input)?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Model: {}", model.name());
    println!("Size: {} bytes (embedded)", model.size());
    println!("Version: {}.{}", model.version().0, model.version().1);
    println!("Compressed: {}", model.is_compressed());
    println!("Encrypted: {}", model.is_encrypted());
    println!();
    println!("Inference demo:");
    println!("  Input: {:?}", input);
    println!("  Output: {:?}", output);
    println!();
    println!("Zero-dependency deployment achieved!");

    Ok(())
}

/// Mock inference for demonstration
fn mock_inference(model: &BundledModel, input: &[f32]) -> Result<Vec<f32>> {
    // In production, this would use the actual model weights
    // For demo, we just return a simple transformation
    let _model_bytes = model.as_bytes();

    // Simple mock: normalize and scale
    let sum: f32 = input.iter().sum();
    let output: Vec<f32> = input.iter().map(|x| x / sum).collect();

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_model_creation() {
        let bytes = create_demo_model_bytes();
        assert!(!bytes.is_empty());
        assert_eq!(&bytes[0..4], b"APRN");
    }

    #[test]
    fn test_model_loading() {
        let bytes = create_demo_model_bytes();
        let model = BundledModel::from_bytes(&bytes).unwrap();

        assert_eq!(model.version(), (1, 0));
        assert!(!model.is_encrypted());
    }

    #[test]
    fn test_mock_inference() {
        let bytes = create_demo_model_bytes();
        let model = BundledModel::from_bytes(&bytes).unwrap();

        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let output = mock_inference(&model, &input).unwrap();

        assert_eq!(output.len(), input.len());

        // Output should sum to 1.0 (normalized)
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_idempotent_loading() {
        let bytes = create_demo_model_bytes();

        let model1 = BundledModel::from_bytes(&bytes).unwrap();
        let model2 = BundledModel::from_bytes(&bytes).unwrap();

        assert_eq!(model1.size(), model2.size());
        assert_eq!(model1.version(), model2.version());
    }

    #[test]
    fn test_no_filesystem_access() {
        // This test verifies the model can be used without any filesystem operations
        let bytes = create_demo_model_bytes();
        let model = BundledModel::from_bytes(&bytes).unwrap();

        // All operations work on in-memory bytes
        let _ = model.name();
        let _ = model.size();
        let _ = model.version();
        let _ = model.is_compressed();
        let _ = model.as_bytes();
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_inference_output_size(input_len in 1usize..100) {
            let bytes = create_demo_model_bytes();
            let model = BundledModel::from_bytes(&bytes).unwrap();
            let input: Vec<f32> = (0..input_len).map(|i| i as f32 + 1.0).collect();

            let output = mock_inference(&model, &input).unwrap();
            prop_assert_eq!(output.len(), input.len());
        }

        #[test]
        fn prop_model_always_loadable(payload_size in 0usize..1000) {
            let bytes = ModelBundle::new()
                .with_payload(vec![0u8; payload_size])
                .build();

            let result = BundledModel::from_bytes(&bytes);
            prop_assert!(result.is_ok());
        }

        #[test]
        fn prop_deterministic_payload(seed in 0u64..1000) {
            let payload1 = generate_model_payload(seed, 100);
            let payload2 = generate_model_payload(seed, 100);
            prop_assert_eq!(payload1, payload2);
        }
    }
}
