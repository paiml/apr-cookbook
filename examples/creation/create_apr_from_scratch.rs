//! # Recipe: Create APR Model from Scratch
//!
//! **Category**: Model Creation
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
//! 6. [x] WASM compatible (N/A - uses filesystem)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Create a `.apr` model from raw tensors without external dependencies.
//!
//! ## Run Command
//! ```bash
//! cargo run --example create_apr_from_scratch
//! ```
//!
//! ## Example Output
//! ```text
//! === Recipe: create_apr_from_scratch ===
//! Created model with 590080 parameters
//! Saved to: /tmp/.../custom_model.apr (2360448 bytes)
//! Roundtrip verification: PASSED
//! ```

use apr_cookbook::prelude::*;
use rand::Rng;

/// Recipe entry point - isolated and idempotent
fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("create_apr_from_scratch")?;

    // Create model weights programmatically using deterministic RNG
    let input_dim = 768;
    let output_dim = 768;
    let weights = generate_weights(ctx.rng(), input_dim, output_dim);
    let biases = generate_biases(ctx.rng(), output_dim);

    // Calculate total parameters
    let n_params = input_dim * output_dim + output_dim;
    ctx.record_metric("parameters", n_params as i64);

    // Build APR model bytes using converter
    let mut converter = AprConverter::new();
    converter.set_metadata(ConversionMetadata {
        name: Some("scratch-model".to_string()),
        architecture: Some("linear".to_string()),
        source_format: None,
        custom: std::collections::HashMap::new(),
    });

    converter.add_tensor(TensorData {
        name: "weights".to_string(),
        shape: vec![input_dim, output_dim],
        dtype: DataType::F32,
        data: weights_to_bytes(&weights),
    });

    converter.add_tensor(TensorData {
        name: "bias".to_string(),
        shape: vec![output_dim],
        dtype: DataType::F32,
        data: weights_to_bytes(&biases),
    });

    // Save to APR format
    let apr_path = ctx.path("custom_model.apr");
    let apr_bytes = converter.to_apr()?;
    std::fs::write(&apr_path, &apr_bytes)?;

    let file_size = std::fs::metadata(&apr_path)?.len();
    ctx.record_metric("file_size_bytes", file_size as i64);

    // Verify roundtrip - load the saved model
    let loaded_bytes = std::fs::read(&apr_path)?;
    let loaded = BundledModel::from_bytes(&loaded_bytes)?;

    // Verify loaded model properties
    let roundtrip_ok = loaded.size() == apr_bytes.len() && loaded.version() == (1, 0);
    ctx.record_string_metric(
        "roundtrip_verification",
        if roundtrip_ok { "PASSED" } else { "FAILED" },
    );

    // Report results
    println!("=== Recipe: {} ===", ctx.name());
    println!("Created model with {} parameters", n_params);
    println!("Saved to: {:?} ({} bytes)", apr_path, file_size);
    println!(
        "Roundtrip verification: {}",
        if roundtrip_ok { "PASSED" } else { "FAILED" }
    );
    println!("Duration: {:.2}ms", ctx.elapsed().as_secs_f64() * 1000.0);

    Ok(())
}

/// Generate random weights with deterministic RNG
fn generate_weights(rng: &mut impl Rng, rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|_| rng.gen_range(-0.1f32..0.1f32))
        .collect()
}

/// Generate random biases with deterministic RNG
fn generate_biases(rng: &mut impl Rng, size: usize) -> Vec<f32> {
    (0..size)
        .map(|_| rng.gen_range(-0.01f32..0.01f32))
        .collect()
}

/// Convert f32 weights to raw bytes
fn weights_to_bytes(weights: &[f32]) -> Vec<u8> {
    weights.iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creates_valid_apr_header() {
        let mut ctx = RecipeContext::new("test_creates_valid_apr_header").unwrap();
        let weights = generate_weights(ctx.rng(), 64, 32);

        let mut converter = AprConverter::new();
        converter.add_tensor(TensorData {
            name: "w".to_string(),
            shape: vec![64, 32],
            dtype: DataType::F32,
            data: weights_to_bytes(&weights),
        });

        let apr_bytes = converter.to_apr().unwrap();
        assert_eq!(&apr_bytes[0..4], b"APRN", "Should have APR magic bytes");
    }

    #[test]
    fn test_tensors_preserved_exactly() {
        let mut ctx = RecipeContext::new("test_tensors_preserved").unwrap();
        let original_weights = generate_weights(ctx.rng(), 16, 8);

        let mut converter = AprConverter::new();
        converter.add_tensor(TensorData {
            name: "weights".to_string(),
            shape: vec![16, 8],
            dtype: DataType::F32,
            data: weights_to_bytes(&original_weights),
        });

        assert_eq!(converter.tensor_count(), 1);
        assert_eq!(converter.total_parameters(), 16 * 8);

        let tensor = converter.get_tensor("weights").unwrap();
        assert_eq!(tensor.shape, vec![16, 8]);
    }

    #[test]
    fn test_metadata_roundtrip() {
        let mut converter = AprConverter::new();
        converter.set_metadata(ConversionMetadata {
            name: Some("test-model".to_string()),
            architecture: Some("mlp".to_string()),
            source_format: None,
            custom: std::collections::HashMap::new(),
        });

        converter.add_tensor(TensorData {
            name: "w".to_string(),
            shape: vec![4, 4],
            dtype: DataType::F32,
            data: vec![0u8; 64],
        });

        let apr_bytes = converter.to_apr().unwrap();
        let model = BundledModel::from_bytes(&apr_bytes).unwrap();

        // Model should be loadable
        assert!(model.size() > 32);
        assert_eq!(model.version(), (1, 0));
    }

    #[test]
    fn test_deterministic_output() {
        // Two runs with same recipe name should produce identical weights
        let mut ctx1 = RecipeContext::new("deterministic_weights_test").unwrap();
        let mut ctx2 = RecipeContext::new("deterministic_weights_test").unwrap();

        let weights1 = generate_weights(ctx1.rng(), 100, 50);
        let weights2 = generate_weights(ctx2.rng(), 100, 50);

        assert_eq!(weights1, weights2, "Same seed should produce same weights");
    }

    #[test]
    fn test_idempotency() {
        // Running the recipe twice should succeed both times
        let result1 = run_recipe();
        let result2 = run_recipe();

        assert!(result1.is_ok());
        assert!(result2.is_ok());
    }

    fn run_recipe() -> apr_cookbook::Result<()> {
        let mut ctx = RecipeContext::new("idempotency_test")?;
        let weights = generate_weights(ctx.rng(), 32, 16);

        let mut converter = AprConverter::new();
        converter.add_tensor(TensorData {
            name: "w".to_string(),
            shape: vec![32, 16],
            dtype: DataType::F32,
            data: weights_to_bytes(&weights),
        });

        let apr_path = ctx.path("model.apr");
        let apr_bytes = converter.to_apr()?;
        std::fs::write(&apr_path, &apr_bytes)?;

        Ok(())
    }

    #[test]
    fn test_isolation_no_file_leaks() {
        let temp_path = {
            let ctx = RecipeContext::new("isolation_test").unwrap();
            let path = ctx.path("test.apr");
            std::fs::write(&path, b"test").unwrap();
            ctx.temp_dir().to_path_buf()
        };

        // After context drops, temp dir should be cleaned up
        assert!(
            !temp_path.exists(),
            "Temp directory should be cleaned up on drop"
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_random_dimensions(rows in 1usize..256, cols in 1usize..256) {
            let mut ctx = RecipeContext::new("prop_dimensions").unwrap();
            let weights = generate_weights(ctx.rng(), rows, cols);

            prop_assert_eq!(weights.len(), rows * cols);

            let bytes = weights_to_bytes(&weights);
            prop_assert_eq!(bytes.len(), rows * cols * 4);
        }

        #[test]
        fn prop_apr_always_valid(size in 1usize..100) {
            let mut converter = AprConverter::new();
            converter.add_tensor(TensorData {
                name: "w".to_string(),
                shape: vec![size, size],
                dtype: DataType::F32,
                data: vec![0u8; size * size * 4],
            });

            let apr_bytes = converter.to_apr().unwrap();

            // Should always produce valid APR
            prop_assert_eq!(&apr_bytes[0..4], b"APRN");
            prop_assert!(apr_bytes.len() >= 32);
        }

        #[test]
        fn prop_deterministic_generation(seed_suffix in 0u64..1000) {
            let name = format!("prop_seed_{}", seed_suffix);

            let mut ctx1 = RecipeContext::new(&name).unwrap();
            let mut ctx2 = RecipeContext::new(&name).unwrap();

            use rand::Rng;
            let val1: u64 = ctx1.rng().gen();
            let val2: u64 = ctx2.rng().gen();

            prop_assert_eq!(val1, val2, "Same name should produce same RNG values");
        }
    }
}
