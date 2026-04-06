#![allow(unused_imports)]
//! # APR Tensor Listing
//!
//! CLI equivalent: `apr tensors model.apr [--stats]`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Lists all tensors in a model file with shape, dtype, size, and optional
//! statistics (mean, std, min, max, NaN count, sparsity). Prints a compact
//! table sorted by size (largest first) with a total summary and dtype
//! breakdown.
//!
//!
//! ## Format Variants
//! ```bash
//! apr tensors model.apr          # APR native format
//! apr tensors model.gguf         # GGUF (llama.cpp compatible)
//! apr tensors model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::fmt;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("analysis_tensors")?;

    println!("=== APR Tensor Listing ===\n");

    // --- Section 1: Build synthetic 2-layer transformer ---
    let mut tensors = build_model(ctx.rng());
    println!(
        "Built synthetic 2-layer transformer with {} tensors.\n",
        tensors.len()
    );

    // --- Section 2: Sort by size (largest first) ---
    tensors.sort_by_key(|t| std::cmp::Reverse(t.size_bytes()));

    // --- Section 3: Compact table (no stats) ---
    println!("--- Tensor Table ---");
    print_tensor_table(&tensors, false);

    // --- Section 4: Table with stats ---
    println!("\n--- Tensor Table (--stats) ---");
    print_tensor_table(&tensors, true);

    // --- Section 5: Summary and dtype breakdown ---
    print_summary(&tensors);

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small model with the same 15-tensor structure but tiny dims
    /// so tests finish in milliseconds.
    fn small_model() -> Vec<TensorInfo> {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut ts = Vec::with_capacity(15);

        ts.push(make_tensor(
            &mut rng,
            "embed_tokens.weight",
            vec![32, 8],
            DType::FP16,
        ));
        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let name = format!("layers.0.self_attn.{proj}.weight");
            ts.push(make_tensor(&mut rng, &name, vec![8, 8], DType::FP32));
        }
        ts.push(make_tensor(
            &mut rng,
            "layers.0.mlp.gate_proj.weight",
            vec![16, 8],
            DType::FP32,
        ));
        ts.push(make_tensor(
            &mut rng,
            "layers.0.mlp.up_proj.weight",
            vec![16, 8],
            DType::FP32,
        ));
        ts.push(make_tensor(
            &mut rng,
            "layers.0.mlp.down_proj.weight",
            vec![8, 16],
            DType::FP32,
        ));
        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            let name = format!("layers.1.self_attn.{proj}.weight");
            ts.push(make_tensor(&mut rng, &name, vec![8, 8], DType::FP16));
        }
        ts.push(make_tensor(
            &mut rng,
            "layers.1.mlp.gate_proj.weight",
            vec![16, 8],
            DType::INT8,
        ));
        ts.push(make_tensor(&mut rng, "norm.weight", vec![8], DType::FP32));
        ts.push(make_tensor(
            &mut rng,
            "lm_head.weight",
            vec![32, 8],
            DType::INT8,
        ));
        ts
    }

    #[test]
    fn test_model_has_15_tensors() {
        let tensors = small_model();
        assert_eq!(tensors.len(), 15);
    }

    #[test]
    fn test_dtype_element_bytes() {
        assert_eq!(DType::FP32.element_bytes(), 4);
        assert_eq!(DType::FP16.element_bytes(), 2);
        assert_eq!(DType::INT8.element_bytes(), 1);
    }

    #[test]
    fn test_param_count_2d() {
        let t = TensorInfo {
            name: "w".to_string(),
            shape: vec![100, 200],
            dtype: DType::FP32,
            data: vec![],
        };
        assert_eq!(t.param_count(), 20_000);
    }

    #[test]
    fn test_size_bytes_fp16() {
        let t = TensorInfo {
            name: "w".to_string(),
            shape: vec![1000, 512],
            dtype: DType::FP16,
            data: vec![],
        };
        assert_eq!(t.size_bytes(), 1000 * 512 * 2);
    }

    #[test]
    fn test_compute_stats_basic() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_stats(&data);
        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!((stats.min - 1.0).abs() < 1e-6);
        assert!((stats.max - 5.0).abs() < 1e-6);
        assert_eq!(stats.nan_count, 0);
        assert!((stats.sparsity_pct - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_stats_empty() {
        let stats = compute_stats(&[]);
        assert!((stats.mean - 0.0).abs() < 1e-6);
        assert!((stats.std - 0.0).abs() < 1e-6);
        assert_eq!(stats.nan_count, 0);
        assert!((stats.sparsity_pct - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_stats_with_nans() {
        let data = vec![1.0, f32::NAN, 3.0, f32::NAN, 5.0];
        let stats = compute_stats(&data);
        assert_eq!(stats.nan_count, 2);
        assert!((stats.mean - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_stats_sparsity() {
        let data = vec![0.0, 0.0, 1.0, 0.0, 2.0];
        let stats = compute_stats(&data);
        assert!((stats.sparsity_pct - 60.0).abs() < 1e-6);
    }

    #[test]
    fn test_dtype_breakdown_totals() {
        let tensors = small_model();
        let breakdown = dtype_breakdown(&tensors);
        let total_bytes: usize = breakdown.values().map(|(_, b)| b).sum();
        let expected: usize = tensors.iter().map(TensorInfo::size_bytes).sum();
        assert_eq!(total_bytes, expected);
    }

    #[test]
    fn test_sort_by_size_descending() {
        let mut tensors = small_model();
        tensors.sort_by_key(|t| std::cmp::Reverse(t.size_bytes()));
        for pair in tensors.windows(2) {
            assert!(
                pair[0].size_bytes() >= pair[1].size_bytes(),
                "{} ({}) should be >= {} ({})",
                pair[0].name,
                pair[0].size_bytes(),
                pair[1].name,
                pair[1].size_bytes(),
            );
        }
    }
}
