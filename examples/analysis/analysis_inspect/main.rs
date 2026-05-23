#![allow(unused_imports)]
//! # APR Model Inspection
//!
//! CLI equivalent: `apr inspect model.apr`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Inspects an APR model file to extract metadata, architecture details,
//! tensor listing, and size breakdown. Essential for understanding model
//! structure before inference or conversion.
//!
//!
//! ## Format Variants
//! ```bash
//! apr inspect model.apr          # APR native format
//! apr inspect model.gguf         # GGUF (llama.cpp compatible)
//! apr inspect model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use std::collections::HashMap;
use std::fmt;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    let ctx = RecipeContext::new("analysis_inspect")?;

    // --- Section 1: Create a multi-tensor test model ---
    println!("=== APR Model Inspector ===\n");

    let embed_dim = 128;
    let hidden_dim = 256;
    let vocab_size = 1000;

    let seed = hash_name_to_seed("inspect-model");
    let embed_bytes = generate_model_payload(seed, vocab_size * embed_dim);
    let attn_bytes = generate_model_payload(seed + 1, hidden_dim * hidden_dim);
    let ffn_bytes = generate_model_payload(seed + 2, hidden_dim * hidden_dim);
    let norm_bytes = generate_model_payload(seed + 3, hidden_dim);
    let output_bytes = generate_model_payload(seed + 4, vocab_size * hidden_dim);

    let bundle = ModelBundleV2::new()
        .with_name("transformer-tiny")
        .with_description("Tiny transformer for inspection demo")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("embed.weight", vec![vocab_size, embed_dim], embed_bytes)
        .add_tensor("attn.qkv", vec![hidden_dim, hidden_dim], attn_bytes)
        .add_tensor("ffn.up", vec![hidden_dim, hidden_dim], ffn_bytes)
        .add_tensor("norm.weight", vec![hidden_dim], norm_bytes)
        .add_tensor("output.proj", vec![vocab_size, hidden_dim], output_bytes)
        .build();

    let model_path = ctx.path("transformer-tiny.apr");
    std::fs::write(&model_path, &bundle)?;
    println!(
        "Created test model: {} ({} bytes)\n",
        model_path.display(),
        bundle.len()
    );

    // --- Section 2: Model overview ---
    println!("--- Model Overview ---");
    let result = inspect_apr(&bundle).map_err(CookbookError::invalid_format)?;
    println!("{result}");

    // --- Section 3: Tensor listing table ---
    println!("--- Tensor Listing ---");
    println!(
        "{:<20} {:<20} {:<8} {:<12} {:<10}",
        "Name", "Shape", "DType", "Params", "Size"
    );
    println!("{}", "-".repeat(70));
    for t in &result.tensors {
        let shape_str = t
            .shape
            .iter()
            .map(|d: &usize| d.to_string())
            .collect::<Vec<_>>()
            .join("x");
        println!(
            "{:<20} {:<20} {:<8} {:<12} {:<10}",
            t.name,
            shape_str,
            t.dtype,
            t.param_count(),
            format_size(t.size_bytes)
        );
    }
    println!();

    // --- Section 4: Size breakdown by category ---
    println!("--- Size Breakdown ---");
    let breakdown = size_breakdown(&result.tensors);
    let total: usize = breakdown.values().sum();
    let mut sorted: Vec<_> = breakdown.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    for (category, size) in &sorted {
        let pct = if total > 0 {
            (**size as f64 / total as f64) * 100.0
        } else {
            0.0
        };
        println!(
            "  {:<20} {:>10}  ({:.1}%)",
            category,
            format_size(**size),
            pct
        );
    }
    println!("  {:<20} {:>10}", "TOTAL", format_size(total));
    println!();

    // --- Section 5: Compression statistics ---
    println!("--- Compression Stats ---");
    let raw_size = result.total_bytes;
    let file_size = bundle.len();
    let ratio = if raw_size > 0 {
        file_size as f64 / raw_size as f64
    } else {
        1.0
    };
    println!("  Raw tensor size:    {}", format_size(raw_size));
    println!("  File size on disk:  {}", format_size(file_size));
    println!("  Compression ratio:  {:.2}x", 1.0 / ratio);
    println!("  Compression method: {}", result.compression);

    // Verify magic bytes
    assert_eq!(&bundle[0..4], b"APR2", "APR v2 magic bytes must be present");
    println!("\nMagic bytes verified: APR2");

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_bundle(name: &str, tensors: Vec<(&str, Vec<usize>)>) -> Vec<u8> {
        let seed = hash_name_to_seed(name);
        let mut builder = ModelBundleV2::new()
            .with_name(name)
            .with_description("test model")
            .with_compression(Compression::Lz4)
            .with_quantization(Quantization::FP32);

        for (i, (tname, shape)) in tensors.iter().enumerate() {
            let num_elements: usize = shape.iter().product();
            let payload = generate_model_payload(seed + i as u64, num_elements);
            builder = builder.add_tensor(*tname, shape.clone(), payload);
        }
        builder.build()
    }

    #[test]
    fn test_magic_bytes_valid() {
        let bundle = make_test_bundle("test", vec![("w", vec![4, 4])]);
        assert_eq!(&bundle[0..4], b"APR2");
    }

    #[test]
    fn test_magic_bytes_invalid() {
        let mut bundle = make_test_bundle("test", vec![("w", vec![4, 4])]);
        bundle[0] = b'X';
        let result = inspect_apr(&bundle);
        assert!(result.is_err());
    }

    #[test]
    fn test_file_too_small() {
        let result = inspect_apr(&[0x41, 0x50]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too small"));
    }

    #[test]
    fn test_tensor_count_single() {
        let bundle = make_test_bundle("single", vec![("weight", vec![10, 5])]);
        let result = inspect_apr(&bundle).unwrap();
        assert!(result.num_tensors >= 1);
    }

    #[test]
    fn test_param_count_calculation() {
        let info = TensorInfo {
            name: "w".to_string(),
            shape: vec![10, 20, 3],
            dtype: "f32".to_string(),
            size_bytes: 2400,
        };
        assert_eq!(info.param_count(), 600);
    }

    #[test]
    fn test_size_bytes_calculation() {
        let info = TensorInfo {
            name: "w".to_string(),
            shape: vec![100, 100],
            dtype: "f32".to_string(),
            size_bytes: 40000,
        };
        assert_eq!(info.size_bytes, 40000);
        assert_eq!(info.param_count(), 10000);
    }

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(500), "500 B");
    }

    #[test]
    fn test_format_size_kb() {
        let s = format_size(2048);
        assert!(s.contains("KB"));
    }

    #[test]
    fn test_format_size_mb() {
        let s = format_size(5_242_880);
        assert!(s.contains("MB"));
    }

    #[test]
    fn test_format_size_gb() {
        let s = format_size(2_147_483_648);
        assert!(s.contains("GB"));
    }

    #[test]
    fn test_size_breakdown_categories() {
        let tensors = vec![
            TensorInfo {
                name: "embed.weight".to_string(),
                shape: vec![100],
                dtype: "f32".to_string(),
                size_bytes: 400,
            },
            TensorInfo {
                name: "attn.qkv".to_string(),
                shape: vec![64],
                dtype: "f32".to_string(),
                size_bytes: 256,
            },
            TensorInfo {
                name: "ffn.up".to_string(),
                shape: vec![128],
                dtype: "f32".to_string(),
                size_bytes: 512,
            },
        ];
        let breakdown = size_breakdown(&tensors);
        assert!(breakdown.contains_key("embedding"));
        assert!(breakdown.contains_key("attention"));
        assert!(breakdown.contains_key("feed-forward"));
    }

    #[test]
    fn test_detect_compression_none() {
        // Simple APR2 header with no compression markers
        let mut data = b"APR2".to_vec();
        data.extend_from_slice(&[0u8; 100]);
        let comp = detect_compression(&data);
        assert_eq!(comp, "None");
    }

    #[test]
    fn test_inspect_result_display() {
        let result = InspectResult {
            name: "test".to_string(),
            description: "demo".to_string(),
            format_version: 2,
            num_tensors: 3,
            total_params: 1000,
            total_bytes: 4000,
            compression: "LZ4".to_string(),
            tensors: vec![],
        };
        let display = format!("{result}");
        assert!(display.contains("test"));
        assert!(display.contains("APR v2"));
        assert!(display.contains("1000"));
    }
}
