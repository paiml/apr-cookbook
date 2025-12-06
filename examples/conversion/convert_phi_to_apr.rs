//! # Recipe: Convert Microsoft Phi to APR
//!
//! **Category**: Format Conversion
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
//! Convert Microsoft Phi-3 Mini (mock) to `.apr` format.
//!
//! ## Run Command
//! ```bash
//! cargo run --example convert_phi_to_apr
//! ```

use apr_cookbook::prelude::*;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("convert_phi_to_apr")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Converting Microsoft Phi-3 Mini (mock) to .apr format");
    println!();

    // Create mock Phi-3 tensor structure
    // Real Phi-3-Mini has 3.8B parameters, we simulate the structure
    let hidden_size = 3072;
    let num_layers = 32;
    let vocab_size = 32064;
    let _head_dim = 96;
    let num_heads = 32;

    let mock_seed = hash_name_to_seed("phi3_mock");

    // Build converter with Phi architecture
    let mut converter = AprConverter::new();
    converter.set_metadata(ConversionMetadata {
        name: Some("phi-3-mini-mock".to_string()),
        architecture: Some("phi3".to_string()),
        source_format: Some(ConversionFormat::SafeTensors),
        custom: [
            ("hidden_size".to_string(), hidden_size.to_string()),
            ("num_layers".to_string(), num_layers.to_string()),
            ("vocab_size".to_string(), vocab_size.to_string()),
            ("num_heads".to_string(), num_heads.to_string()),
        ]
        .into_iter()
        .collect(),
    });

    // Add embedding layer (mock - smaller for demo)
    let embed_dim = 256; // Reduced for demo
    let embed_vocab = 1000; // Reduced for demo
    let embed_data = generate_tensor_data(mock_seed, embed_vocab * embed_dim);
    converter.add_tensor(TensorData {
        name: "model.embed_tokens.weight".to_string(),
        shape: vec![embed_vocab, embed_dim],
        dtype: DataType::F16,
        data: embed_data,
    });

    // Add a few attention layers (mock)
    for layer_idx in 0..2 {
        // Reduced layers for demo
        let layer_seed = mock_seed.wrapping_add(layer_idx as u64 * 1000);

        // Q, K, V projections
        let qkv_size = 128 * 128; // Reduced for demo
        converter.add_tensor(TensorData {
            name: format!("model.layers.{}.self_attn.q_proj.weight", layer_idx),
            shape: vec![128, 128],
            dtype: DataType::F16,
            data: generate_tensor_data(layer_seed, qkv_size),
        });

        converter.add_tensor(TensorData {
            name: format!("model.layers.{}.self_attn.k_proj.weight", layer_idx),
            shape: vec![128, 128],
            dtype: DataType::F16,
            data: generate_tensor_data(layer_seed.wrapping_add(1), qkv_size),
        });

        converter.add_tensor(TensorData {
            name: format!("model.layers.{}.self_attn.v_proj.weight", layer_idx),
            shape: vec![128, 128],
            dtype: DataType::F16,
            data: generate_tensor_data(layer_seed.wrapping_add(2), qkv_size),
        });

        // Output projection
        converter.add_tensor(TensorData {
            name: format!("model.layers.{}.self_attn.o_proj.weight", layer_idx),
            shape: vec![128, 128],
            dtype: DataType::F16,
            data: generate_tensor_data(layer_seed.wrapping_add(3), qkv_size),
        });
    }

    // Calculate stats
    let total_params = converter.total_parameters();
    ctx.record_metric("total_parameters", total_params as i64);
    ctx.record_metric("tensor_count", converter.tensor_count() as i64);

    // Convert to APR
    let apr_bytes = converter.to_apr()?;
    let apr_path = ctx.path("phi-3-mini-mock.apr");
    std::fs::write(&apr_path, &apr_bytes)?;

    ctx.record_metric("apr_size_bytes", apr_bytes.len() as i64);

    // Verify loadable
    let loaded = BundledModel::from_bytes(&apr_bytes)?;

    println!("Conversion complete:");
    println!("  Source format: SafeTensors (mock)");
    println!("  Target format: APR");
    println!();
    println!("Model architecture (mock):");
    println!("  Hidden size: {}", hidden_size);
    println!("  Num layers: {} (full), 2 (demo)", num_layers);
    println!(
        "  Vocab size: {} (full), {} (demo)",
        vocab_size, embed_vocab
    );
    println!("  Num heads: {}", num_heads);
    println!();
    println!("Conversion stats:");
    println!("  Tensors: {}", converter.tensor_count());
    println!("  Parameters: {}", total_params);
    println!("  APR size: {} bytes", apr_bytes.len());
    println!("  Verified loadable: {}", loaded.size() > 0);
    println!();
    println!("Saved to: {:?}", apr_path);

    Ok(())
}

/// Generate deterministic tensor data
fn generate_tensor_data(seed: u64, n_elements: usize) -> Vec<u8> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Generate F16 data (2 bytes per element)
    let mut data = Vec::with_capacity(n_elements * 2);
    for _ in 0..n_elements {
        let val: f32 = rng.gen_range(-0.1f32..0.1f32);
        // Convert to f16 representation (simplified: just truncate f32)
        let f16_bits = f32_to_f16_bits(val);
        data.extend_from_slice(&f16_bits.to_le_bytes());
    }
    data
}

/// Convert f32 to f16 bits (simplified)
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let frac = ((bits >> 13) & 0x3FF) as u16;

    if exp <= 0 {
        // Subnormal or zero
        (sign << 15) | frac
    } else if exp >= 31 {
        // Infinity or NaN
        (sign << 15) | 0x7C00
    } else {
        (sign << 15) | ((exp as u16) << 10) | frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_data_generation() {
        let data = generate_tensor_data(42, 100);
        assert_eq!(data.len(), 200); // 100 elements * 2 bytes (f16)
    }

    #[test]
    fn test_deterministic_generation() {
        let data1 = generate_tensor_data(42, 100);
        let data2 = generate_tensor_data(42, 100);
        assert_eq!(data1, data2);
    }

    #[test]
    fn test_f16_conversion() {
        let zero = f32_to_f16_bits(0.0);
        assert_eq!(zero & 0x7FFF, 0); // Zero has zero exp and frac

        let one = f32_to_f16_bits(1.0);
        assert_ne!(one, 0); // One is not zero
    }

    #[test]
    fn test_converter_setup() {
        let mut converter = AprConverter::new();
        converter.add_tensor(TensorData {
            name: "test".to_string(),
            shape: vec![10, 10],
            dtype: DataType::F16,
            data: vec![0u8; 200],
        });

        assert_eq!(converter.tensor_count(), 1);
        assert_eq!(converter.total_parameters(), 100);
    }

    #[test]
    fn test_apr_output_valid() {
        let mut converter = AprConverter::new();
        converter.set_metadata(ConversionMetadata {
            name: Some("test".to_string()),
            ..Default::default()
        });
        converter.add_tensor(TensorData {
            name: "w".to_string(),
            shape: vec![10],
            dtype: DataType::F16,
            data: vec![0u8; 20],
        });

        let apr_bytes = converter.to_apr().unwrap();
        assert_eq!(&apr_bytes[0..4], b"APRN");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_tensor_size(n_elements in 1usize..1000) {
            let data = generate_tensor_data(42, n_elements);
            prop_assert_eq!(data.len(), n_elements * 2);
        }

        #[test]
        fn prop_f16_finite(val in -1000.0f32..1000.0) {
            let f16 = f32_to_f16_bits(val);
            // Should not produce NaN (0x7C01-0x7FFF or 0xFC01-0xFFFF)
            let exp = (f16 >> 10) & 0x1F;
            let frac = f16 & 0x3FF;
            prop_assert!(!(exp == 31 && frac != 0));
        }
    }
}
