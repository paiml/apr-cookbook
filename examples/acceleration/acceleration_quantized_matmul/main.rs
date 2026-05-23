#![allow(unused_imports)]
//! # Recipe: Accelerated Quantized Matrix Multiplication
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/avx512-matmul-v1.yaml
//! **Category**: Acceleration
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## Learning Objective
//! Demonstrate how INT8 and INT4 quantized matrix multiplication reduces memory
//! bandwidth while preserving inference accuracy. Compares FP32 baseline, FP16
//! simulated, INT8 (scale + zero-point), and INT4 (packed 2-per-byte) approaches.
//!
//! ## Run Command
//! ```bash
//! cargo run --example acceleration_quantized_matmul --release
//! ```
//!
//! ## Toyota Way Principles
//! - **Muda** (Waste elimination): 4-8x memory reduction via quantization
//! - **Jidoka** (Quality built-in): Error metrics validate precision tradeoff
//! - **Genchi Genbutsu** (Go and see): Concrete throughput numbers per method
//!
//!
//! ## Format Variants
//! ```bash
//! apr bench model.apr          # APR native format
//! apr bench model.gguf         # GGUF (llama.cpp compatible)
//! apr bench model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Hennessy, J. & Patterson, D. (2017). *Computer Architecture: A Quantitative Approach*. DOI: 10.1016/C2012-0-01712-X

use apr_cookbook::prelude::*;
use rand::Rng;
use std::fmt;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    println!("=== APR Cookbook: Accelerated Quantized Matrix Multiplication ===\n");

    let mut ctx = RecipeContext::new("acceleration_quantized_matmul")?;

    // Generate deterministic weights (1024x1024) and input (1024)
    println!("1. Generating synthetic data");
    println!("   Weight matrix: {}x{}", DIM, DIM);
    println!("   Input vector:  {}\n", DIM);

    let weights: Vec<f64> = (0..DIM * DIM)
        .map(|_| ctx.rng().gen_range(-1.0..1.0))
        .collect();
    let input: Vec<f64> = (0..DIM).map(|_| ctx.rng().gen_range(-1.0..1.0)).collect();

    // Run all variants
    println!("2. Running quantized matmul variants\n");
    let results = run_comparison(&weights, &input);

    // Print comparison table
    println!("3. Comparison Table\n");
    print_comparison_table(&results);

    // Print tradeoff curve
    println!("\n4. Accuracy vs Compression\n");
    print_tradeoff_curve(&results);

    // Record metrics
    for r in &results {
        let prefix = format!("{}", r.method);
        ctx.record_metric(&format!("{}_memory_bytes", prefix), r.memory_bytes as i64);
        ctx.record_float_metric(&format!("{}_cosine_sim", prefix), r.cosine_sim);
        ctx.record_float_metric(&format!("{}_max_error", prefix), r.max_error);
        ctx.record_float_metric(&format!("{}_mean_error", prefix), r.mean_error);
    }

    println!();
    ctx.report()?;

    println!("\n[SUCCESS] Quantized matmul comparison complete.");
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx() -> RecipeContext {
        RecipeContext::new("test_quantized_matmul").expect("context creation must succeed")
    }

    fn make_small_data(ctx: &mut RecipeContext) -> (Vec<f64>, Vec<f64>) {
        let weights: Vec<f64> = (0..64).map(|_| ctx.rng().gen_range(-1.0..1.0)).collect();
        let input: Vec<f64> = (0..8).map(|_| ctx.rng().gen_range(-1.0..1.0)).collect();
        (weights, input)
    }

    #[test]
    fn test_fp32_roundtrip_exact() {
        let mut ctx = make_ctx();
        let (weights, _) = make_small_data(&mut ctx);
        let qw = quantize_weights(&weights, 8, 8, QuantMethod::FP32);
        for (i, &original) in weights.iter().enumerate() {
            let restored = dequantize_weight(&qw, i);
            // FP32 roundtrip loses f64 precision but f32 must match
            assert!(
                (original as f32 - restored as f32).abs() < 1e-7,
                "FP32 roundtrip mismatch at index {}: {} vs {}",
                i,
                original,
                restored,
            );
        }
    }

    #[test]
    fn test_int8_quantization_bounded_error() {
        let mut ctx = make_ctx();
        let (weights, _) = make_small_data(&mut ctx);
        let qw = quantize_weights(&weights, 8, 8, QuantMethod::INT8);
        for (i, &original) in weights.iter().enumerate() {
            let restored = dequantize_weight(&qw, i);
            let err = (original - restored).abs();
            // INT8 with 256 levels over range [-1,1] gives step ~0.008
            assert!(
                err < 0.02,
                "INT8 error {} at index {} too large (original={}, restored={})",
                err,
                i,
                original,
                restored,
            );
        }
    }

    #[test]
    fn test_int4_quantization_bounded_error() {
        let mut ctx = make_ctx();
        let (weights, _) = make_small_data(&mut ctx);
        let qw = quantize_weights(&weights, 8, 8, QuantMethod::INT4);
        for (i, &original) in weights.iter().enumerate() {
            let restored = dequantize_weight(&qw, i);
            let err = (original - restored).abs();
            // INT4 with 16 levels over range [-1,1] gives step ~0.133
            assert!(
                err < 0.2,
                "INT4 error {} at index {} too large (original={}, restored={})",
                err,
                i,
                original,
                restored,
            );
        }
    }

    #[test]
    fn test_int4_packing_roundtrip() {
        // Verify 2-per-byte packing/unpacking is correct
        let weights = vec![0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.0, 0.8];
        let qw = quantize_weights(&weights, 2, 4, QuantMethod::INT4);
        // Packed length should be ceil(8/2) = 4 bytes
        assert_eq!(qw.data.len(), 4, "INT4 should pack 2 values per byte");
        // Dequantized values should be close to originals
        for (i, &original) in weights.iter().enumerate() {
            let restored = dequantize_weight(&qw, i);
            assert!(
                (original - restored).abs() < 0.2,
                "INT4 pack roundtrip failed at {}: {} vs {}",
                i,
                original,
                restored,
            );
        }
    }

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&a, &a);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "Identical vectors should have cosine similarity 1.0, got {}",
            sim,
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-10,
            "Orthogonal vectors should have cosine similarity 0.0, got {}",
            sim,
        );
    }

    #[test]
    fn test_memory_footprint_ordering() {
        let mut ctx = make_ctx();
        let weights: Vec<f64> = (0..DIM * DIM)
            .map(|_| ctx.rng().gen_range(-1.0..1.0))
            .collect();
        let fp32 = quantize_weights(&weights, DIM, DIM, QuantMethod::FP32);
        let fp16 = quantize_weights(&weights, DIM, DIM, QuantMethod::FP16);
        let int8 = quantize_weights(&weights, DIM, DIM, QuantMethod::INT8);
        let int4 = quantize_weights(&weights, DIM, DIM, QuantMethod::INT4);
        assert!(
            fp32.data.len() > fp16.data.len(),
            "FP32 ({}) should use more memory than FP16 ({})",
            fp32.data.len(),
            fp16.data.len(),
        );
        assert!(
            fp16.data.len() > int8.data.len(),
            "FP16 ({}) should use more memory than INT8 ({})",
            fp16.data.len(),
            int8.data.len(),
        );
        assert!(
            int8.data.len() > int4.data.len(),
            "INT8 ({}) should use more memory than INT4 ({})",
            int8.data.len(),
            int4.data.len(),
        );
    }

    #[test]
    fn test_matmul_output_length() {
        let mut ctx = make_ctx();
        let (weights, input) = make_small_data(&mut ctx);
        let qw = quantize_weights(&weights, 8, 8, QuantMethod::FP32);
        let output = quantized_matmul(&qw, &input);
        assert_eq!(
            output.len(),
            8,
            "Output should have {} rows, got {}",
            8,
            output.len(),
        );
    }

    #[test]
    fn test_error_increases_with_compression() {
        let mut ctx = make_ctx();
        let weights: Vec<f64> = (0..64).map(|_| ctx.rng().gen_range(-1.0..1.0)).collect();
        let input: Vec<f64> = (0..8).map(|_| ctx.rng().gen_range(-1.0..1.0)).collect();

        let fp32_qw = quantize_weights(&weights, 8, 8, QuantMethod::FP32);
        let baseline = quantized_matmul(&fp32_qw, &input);

        let int8_qw = quantize_weights(&weights, 8, 8, QuantMethod::INT8);
        let int8_out = quantized_matmul(&int8_qw, &input);
        let int8_err = max_abs_error(&int8_out, &baseline);

        let int4_qw = quantize_weights(&weights, 8, 8, QuantMethod::INT4);
        let int4_out = quantized_matmul(&int4_qw, &input);
        let int4_err = max_abs_error(&int4_out, &baseline);

        assert!(
            int4_err >= int8_err,
            "INT4 error ({}) should be >= INT8 error ({})",
            int4_err,
            int8_err,
        );
    }

    #[test]
    fn test_quant_method_display() {
        assert_eq!(format!("{}", QuantMethod::FP32), "FP32");
        assert_eq!(format!("{}", QuantMethod::FP16), "FP16");
        assert_eq!(format!("{}", QuantMethod::INT8), "INT8");
        assert_eq!(format!("{}", QuantMethod::INT4), "INT4");
    }
}
