//! 4-Bit Quantization
//!
//! CLI equivalent: `apr quantize --scheme int4`
//!
//! Post-training quantization maps floating-point weights to lower bit-width
//! integers using affine quantization: each block of weights shares a scale
//! factor and zero-point. This dramatically reduces memory and enables
//! inference on resource-constrained devices.
//!
//! ## Algorithm
//!
//! ```text
//! Quantize:
//!   scale      = (max - min) / (2^bits - 1)
//!   zero_point = round(-min / scale)
//!   q[i]       = clamp(round(x[i] / scale) + zero_point, 0, 2^bits - 1)
//!
//! Dequantize:
//!   x_hat[i]   = (q[i] - zero_point) * scale
//! ```
//!
//! ## When to Use
//!
//! - Deploying models on edge devices with limited memory
//! - Reducing model size for mobile / WASM inference
//! - Trading small accuracy loss for 4-8x memory savings
//!
//!
//! ## Format Variants
//! ```bash
//! apr quantize model.apr          # APR native format
//! apr quantize model.gguf         # GGUF (llama.cpp compatible)
//! apr quantize model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Dettmers, T. et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. NeurIPS. arXiv:2208.07339

use apr_cookbook::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Quantization primitives
// ---------------------------------------------------------------------------

/// Result of quantizing a float slice to N-bit unsigned integers.
struct QuantizedTensor {
    /// Quantized values as u8 (only lower `bits` bits are used).
    data: Vec<u8>,
    /// Affine scale factor.
    scale: f32,
    /// Affine zero point (integer, stored as u8).
    zero_point: u8,
    /// Bit width used.
    bits: u8,
}

/// Quantize f32 weights to `bits`-bit unsigned integers (affine, per-tensor).
fn quantize(weights: &[f32], bits: u8) -> QuantizedTensor {
    assert!((1..=8).contains(&bits), "bits must be 1-8");
    let qmax = ((1_u32 << bits) - 1) as f32;

    let min_val = weights.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let range = max_val - min_val;
    let scale = if range < 1e-10 { 1.0 } else { range / qmax };
    let zero_point = (-min_val / scale).round().clamp(0.0, qmax) as u8;

    let data: Vec<u8> = weights
        .iter()
        .map(|&x| {
            let q = (x / scale + f32::from(zero_point)).round();
            q.clamp(0.0, qmax) as u8
        })
        .collect();

    QuantizedTensor {
        data,
        scale,
        zero_point,
        bits,
    }
}

/// Dequantize back to f32.
fn dequantize(qt: &QuantizedTensor) -> Vec<f32> {
    qt.data
        .iter()
        .map(|&q| (f32::from(q) - f32::from(qt.zero_point)) * qt.scale)
        .collect()
}

/// Compute RMSE between original and reconstructed weights.
fn rmse(original: &[f32], reconstructed: &[f32]) -> f32 {
    assert_eq!(original.len(), reconstructed.len());
    let mse: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / original.len() as f32;
    mse.sqrt()
}

/// Memory in bytes for quantized representation (data + metadata).
fn quantized_memory(qt: &QuantizedTensor) -> usize {
    // Each element uses ceil(bits/8) bytes; we store as u8 for simplicity
    qt.data.len() + 4 + 1 // data + scale(f32) + zero_point(u8)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn generate_weights(size: usize, seed: u64) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let mut h = DefaultHasher::new();
            (seed, i).hash(&mut h);
            (h.finish() as f32 / u64::MAX as f32 - 0.5) * 2.0
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("quantize_4bit")?;

    let weights = generate_weights(2048, 42);
    println!(
        "Original weights: {} elements, {} bytes (f32)",
        weights.len(),
        weights.len() * 4
    );

    // --- Section 1: Quantization at various bit widths ---
    println!("\n--- Bit Width Comparison ---");
    println!(
        "{:<6} {:<10} {:<12} {:<14} {:<10}",
        "bits", "RMSE", "max_error", "compressed_B", "ratio"
    );

    for &bits in &[4_u8, 8, 16] {
        // 16-bit quantization to show full range
        let effective_bits = bits.min(8); // our impl caps at 8-bit storage
        let qt = quantize(&weights, effective_bits);
        let recon = dequantize(&qt);
        let error = rmse(&weights, &recon);
        let max_err: f32 = weights
            .iter()
            .zip(recon.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        let orig_bytes = weights.len() * 4;
        let q_bytes = quantized_memory(&qt);
        let ratio = orig_bytes as f32 / q_bytes as f32;
        println!("{bits:<6} {error:<10.6} {max_err:<12.6} {q_bytes:<14} {ratio:<10.2}x");
    }

    // --- Section 2: Detailed 4-bit analysis ---
    println!("\n--- 4-Bit Quantization Details ---");
    let qt4 = quantize(&weights, 4);
    let recon4 = dequantize(&qt4);
    println!("Scale:      {:.8}", qt4.scale);
    println!("Zero point: {}", qt4.zero_point);
    println!("Levels:     {} (0..{})", 1 << qt4.bits, (1 << qt4.bits) - 1);

    // Show first 10 values
    println!("\nFirst 10 values:");
    println!(
        "{:<12} {:<8} {:<12} {:<12}",
        "original", "quant", "dequant", "error"
    );
    for i in 0..10 {
        let err = (weights[i] - recon4[i]).abs();
        println!(
            "{:<12.6} {:<8} {:<12.6} {:<12.8}",
            weights[i], qt4.data[i], recon4[i], err
        );
    }

    // --- Section 3: RMSE comparison table ---
    println!("\n--- RMSE vs Bit Width ---");
    let error_4 = rmse(&weights, &recon4);
    let qt8 = quantize(&weights, 8);
    let recon8 = dequantize(&qt8);
    let error_8 = rmse(&weights, &recon8);
    println!("4-bit RMSE: {error_4:.8}");
    println!("8-bit RMSE: {error_8:.8}");
    println!("4-bit error is {:.2}x larger than 8-bit", error_4 / error_8);

    // --- Section 4: Memory savings table ---
    println!("\n--- Memory Savings ---");
    let f32_bytes = weights.len() * 4;
    println!("FP32:  {} bytes (baseline)", f32_bytes);
    println!(
        "INT8:  {} bytes ({:.1}x compression)",
        quantized_memory(&qt8),
        f32_bytes as f32 / quantized_memory(&qt8) as f32
    );
    println!(
        "INT4:  {} bytes ({:.1}x compression)",
        quantized_memory(&qt4),
        f32_bytes as f32 / quantized_memory(&qt4) as f32
    );

    // --- Section 5: Save quantized model as APR v2 ---
    let bundle = ModelBundleV2::new()
        .with_name("quantized-4bit")
        .with_compression(Compression::Lz4)
        .add_tensor("weights_q4", vec![qt4.data.len()], qt4.data.clone())
        .add_tensor("scale", vec![1], qt4.scale.to_le_bytes().to_vec())
        .build();

    assert_eq!(&bundle[0..4], b"APR2");
    println!(
        "\nSaved 4-bit quantized model as APR v2 ({} bytes)",
        bundle.len()
    );

    ctx.report()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_error_bounded_4bit() {
        let weights = generate_weights(1024, 42);
        let qt = quantize(&weights, 4);
        let recon = dequantize(&qt);
        let error = rmse(&weights, &recon);
        // 4-bit has 16 levels over ~2.0 range, so step ~0.133; RMSE should be small
        assert!(error < 0.1, "4-bit RMSE too large: {error}");
    }

    #[test]
    fn test_roundtrip_error_bounded_8bit() {
        let weights = generate_weights(1024, 42);
        let qt = quantize(&weights, 8);
        let recon = dequantize(&qt);
        let error = rmse(&weights, &recon);
        assert!(error < 0.01, "8-bit RMSE too large: {error}");
    }

    #[test]
    fn test_4bit_error_larger_than_8bit() {
        let weights = generate_weights(1024, 42);
        let qt4 = quantize(&weights, 4);
        let qt8 = quantize(&weights, 8);
        let err4 = rmse(&weights, &dequantize(&qt4));
        let err8 = rmse(&weights, &dequantize(&qt8));
        assert!(
            err4 > err8,
            "4-bit error should exceed 8-bit: err4={err4}, err8={err8}"
        );
    }

    #[test]
    fn test_compression_ratio_4bit() {
        let weights = generate_weights(1024, 42);
        let qt = quantize(&weights, 4);
        let orig = weights.len() * 4;
        let compressed = quantized_memory(&qt);
        let ratio = orig as f32 / compressed as f32;
        // 4-bit stored as u8 gives ~4x, minus metadata overhead
        assert!(ratio > 3.5, "compression ratio should be ~4x: {ratio}");
    }

    #[test]
    fn test_quantize_deterministic() {
        let weights = generate_weights(512, 42);
        let qt1 = quantize(&weights, 4);
        let qt2 = quantize(&weights, 4);
        assert_eq!(qt1.data, qt2.data);
        assert_eq!(qt1.scale, qt2.scale);
        assert_eq!(qt1.zero_point, qt2.zero_point);
    }

    #[test]
    fn test_quantize_values_in_range() {
        let weights = generate_weights(512, 42);
        let qt = quantize(&weights, 4);
        let qmax = (1_u8 << qt.bits) - 1;
        for &q in &qt.data {
            assert!(q <= qmax, "quantized value {q} exceeds max {qmax}");
        }
    }

    #[test]
    fn test_quantize_preserves_length() {
        let weights = generate_weights(256, 42);
        let qt = quantize(&weights, 4);
        assert_eq!(qt.data.len(), weights.len());
        let recon = dequantize(&qt);
        assert_eq!(recon.len(), weights.len());
    }

    #[test]
    fn test_symmetric_range_weights_roundtrip() {
        // Weights symmetric around zero should quantize well
        let weights: Vec<f32> = (0..100).map(|i| (i as f32 / 100.0) - 0.5).collect();
        let qt = quantize(&weights, 8);
        let recon = dequantize(&qt);
        let error = rmse(&weights, &recon);
        assert!(
            error < 0.01,
            "symmetric-range weights should have small 8-bit RMSE: {error}"
        );
    }

    #[test]
    fn test_zero_weights_roundtrip() {
        let weights = vec![0.0_f32; 100];
        let qt = quantize(&weights, 4);
        let recon = dequantize(&qt);
        for &v in &recon {
            assert!(v.abs() < 1e-6, "zero weights should dequantize to ~0: {v}");
        }
    }

    #[test]
    fn test_apr_bundle() {
        let weights = generate_weights(512, 42);
        let qt = quantize(&weights, 4);
        let bundle = ModelBundleV2::new()
            .with_name("test-q4")
            .with_compression(Compression::Lz4)
            .add_tensor("w", vec![qt.data.len()], qt.data.clone())
            .build();
        assert_eq!(&bundle[0..4], b"APR2");
    }
}
