#![allow(unused_imports)]
//! # Recipe: Quantization Quality Tradeoff Analysis
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/int4-quantization-v1.yaml
//! Comprehensive analysis of quantization schemes (F32, F16, BF16, Q8_0, Q4_0, Q4_1)
//! measuring accuracy degradation, compression ratios, and reconstruction error.
//!
//! ## QA: Build, test, clippy, fmt PASS. Property tests included.
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Dettmers, T. et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. NeurIPS. arXiv:2208.07339

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};
use std::env;
use std::f32;
use std::time::Instant;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let csv = args.iter().any(|a| a == "--csv");
    let ctx = RecipeContext::new("quantization_quality_tradeoff")?;
    let weights = generate_test_data(hash_name_to_seed(ctx.name()), 10_000);
    println!(
        "Quantization Quality Tradeoff: {} params ({} KB F32)",
        weights.len(),
        weights.len() * 4 / 1024
    );
    let analysis = analyze_quantization("test_model", &weights)?;
    if csv {
        println!("format,compression_ratio,size_bytes,mse,snr_db,psnr_db,max_abs_error,changed_pct,time_us");
        for r in &analysis.results {
            println!(
                "{},{:.4},{},{:.6e},{:.2},{:.2},{:.6e},{:.2},{}",
                r.target_format.name(),
                r.compression_ratio,
                r.quantized_size_bytes,
                r.mse,
                r.snr_db,
                r.psnr_db,
                r.max_abs_error,
                r.changed_pct,
                r.time_us
            );
        }
    } else {
        println!("{:-<80}", "");
        println!(
            " {:6} | {:7} | {:8} | {:9} | {:7} | {:7} | {:8}",
            "FORMAT", "COMPRESS", "SIZE KB", "MSE", "SNR dB", "PSNR dB", "MAX ERR"
        );
        println!("{:-<80}", "");
        for r in &analysis.results {
            println!(
                " {:6} | {:6.2}x | {:8} | {:9.2e} | {:7.1} | {:7.1} | {:8.2e}",
                r.target_format.name(),
                r.compression_ratio,
                r.quantized_size_bytes / 1024,
                r.mse,
                r.snr_db,
                r.psnr_db,
                r.max_abs_error
            );
        }
        println!("{:-<80}", "");
        println!(
            "RECOMMENDED: {} - {}",
            analysis.recommended_format.name(),
            analysis.recommendation_reason
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_f32_roundtrip() {
        let r = quantize_and_measure(&[1.0, -2.0, 3.5, -4.5, 0.0], QuantFormat::F32).expect("q");
        assert!(r.mse < 1e-10);
        assert!(r.snr_db > 90.0);
        assert!((r.compression_ratio - 1.0).abs() < 0.01);
    }
    #[test]
    fn test_f16_bf16() {
        let r16 = quantize_and_measure(
            &[1.0, -2.0, 3.5, -4.5, 0.0, 0.001, -0.001],
            QuantFormat::F16,
        )
        .expect("q");
        assert!((r16.compression_ratio - 2.0).abs() < 0.1 && r16.snr_db > 30.0);
        let rbf = quantize_and_measure(&[1.0, -2.0, 3.5, -4.5, 0.0], QuantFormat::BF16).expect("q");
        assert!((rbf.compression_ratio - 2.0).abs() < 0.1);
    }
    #[test]
    fn test_q8_q4_quantization() {
        let ctx = RecipeContext::new("test_q").expect("ctx");
        let w = generate_test_data(hash_name_to_seed(ctx.name()), 64);
        let r8 = quantize_and_measure(&w, QuantFormat::Q8_0).expect("q");
        assert!(r8.compression_ratio > 3.0 && r8.snr_db > 20.0);
        let r40 = quantize_and_measure(&w, QuantFormat::Q4_0).expect("q");
        assert!(r40.compression_ratio > 6.0);
        let r41 = quantize_and_measure(&w, QuantFormat::Q4_1).expect("q");
        assert!(r41.compression_ratio > 5.0 && r41.snr_db > 10.0);
    }
    #[test]
    fn test_deterministic_and_empty() {
        let r1 = quantize_and_measure(&[1.0, 2.0, 3.0, 4.0], QuantFormat::F16).expect("q1");
        let r2 = quantize_and_measure(&[1.0, 2.0, 3.0, 4.0], QuantFormat::F16).expect("q2");
        assert!((r1.mse - r2.mse).abs() < 1e-10);
        let re = quantize_and_measure(&[], QuantFormat::F32).expect("q");
        assert_eq!(re.weight_count, 0);
    }
    #[test]
    fn test_sparse_and_mse() {
        let rs =
            quantize_and_measure(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0], QuantFormat::Q8_0).expect("q");
        assert!(rs.snr_db > 10.0);
        let mse = compute_mse(&[1.0, 2.0, 3.0, 4.0], &[1.1, 2.1, 3.1, 4.1]);
        assert!((mse - 0.01).abs() < 1e-6);
        assert!(compute_snr_db(&[10.0; 4], &[10.0; 4]) > 90.0);
    }
    #[test]
    fn test_compression_and_block_sizes() {
        assert!((QuantFormat::F32.compression_ratio() - 1.0).abs() < 0.01);
        assert!((QuantFormat::F16.compression_ratio() - 2.0).abs() < 0.01);
        assert!(QuantFormat::Q8_0.compression_ratio() > 3.5);
        assert!(QuantFormat::Q4_0.compression_ratio() > 6.5);
        assert_eq!(QuantFormat::F32.block_size(), 1);
        assert_eq!(QuantFormat::Q8_0.block_size(), 32);
    }
    #[test]
    fn test_analysis() {
        let ctx = RecipeContext::new("test_a").expect("ctx");
        let w = generate_test_data(hash_name_to_seed(ctx.name()), 128);
        let a = analyze_quantization("t", &w).expect("a");
        assert_eq!(a.results.len(), 6);
        assert!(!a.recommendation_reason.is_empty());
    }
    #[test]
    fn test_large_tensor() {
        let ctx = RecipeContext::new("test_l").expect("ctx");
        let w = generate_test_data(hash_name_to_seed(ctx.name()), 10_000);
        let r = quantize_and_measure(&w, QuantFormat::Q4_0).expect("q");
        assert_eq!(r.weight_count, 10_000);
        assert!(r.compression_ratio > 6.0);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    proptest! {
        #[test]
        fn prop_compression_bounds(weights in proptest::collection::vec(0.1f32..10.0, 1024..2048)) {
            for fmt in QuantFormat::ALL {
                let r = quantize_and_measure(&weights, fmt).expect("q");
                match fmt {
                    QuantFormat::F32 => prop_assert!((r.compression_ratio-1.0).abs() < 0.01),
                    QuantFormat::F16|QuantFormat::BF16 => prop_assert!((r.compression_ratio-2.0).abs() < 0.1),
                    QuantFormat::Q8_0 => prop_assert!(r.compression_ratio > 3.3 && r.compression_ratio < 4.0),
                    QuantFormat::Q4_0 => prop_assert!(r.compression_ratio > 6.0 && r.compression_ratio < 7.5),
                    QuantFormat::Q4_1 => prop_assert!(r.compression_ratio > 5.0 && r.compression_ratio < 6.5),
                }
            }
        }
        #[test]
        fn prop_mse_positive_deterministic(weights in proptest::collection::vec(-10.0f32..10.0, 32..64)) {
            for fmt in QuantFormat::ALL {
                let r1 = quantize_and_measure(&weights, fmt).expect("q1");
                let r2 = quantize_and_measure(&weights, fmt).expect("q2");
                prop_assert!(r1.mse >= 0.0);
                prop_assert!((r1.mse - r2.mse).abs() < 1e-10);
            }
        }
    }
}
