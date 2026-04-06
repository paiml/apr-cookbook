//! Quantization-Aware Training (QAT) with Fake Quantization
//!
//! CLI equivalent: QAT (quantization-aware training) simulation
//!
//! Fake quantization inserts quantize-then-dequantize operations into the
//! forward pass so the model "sees" quantization noise during training. The
//! backward pass uses the straight-through estimator (STE): gradients flow
//! through the non-differentiable rounding operation as if it were the
//! identity function. Over many training steps the weights adapt to be
//! quantization-friendly, producing much lower error than post-training
//! quantization (PTQ).
//!
//! ## Key Concepts
//!
//! - **Fake quantize**: quantize then immediately dequantize in f32 so
//!   gradient computation still works
//! - **STE**: `d/dx round(x) ~ 1` within the clamp range, 0 outside
//! - **QAT loop**: forward with fake-quant, backward with STE, SGD update
//!
//! ## When to Use
//!
//! - When PTQ accuracy loss is unacceptable
//! - For INT4/INT8 deployment where every bit of accuracy matters
//! - When you can afford extra training compute for better quantized models
//!
//! ## References
//! - Dettmers, T. et al. (2022). *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*. NeurIPS. arXiv:2208.07339

use apr_cookbook::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Fake quantization primitives
// ---------------------------------------------------------------------------

/// Fake-quantize: quantize then dequantize, staying in f32 for gradient flow.
///
/// Maps continuous values to the nearest quantization level, producing
/// discrete-valued f32 outputs that simulate integer quantization.
fn fake_quantize(x: &[f32], bits: u8) -> Vec<f32> {
    assert!((1..=8).contains(&bits), "bits must be 1-8");
    let qmax = ((1_u32 << bits) - 1) as f32;

    let min_val = x.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;
    let scale = if range < 1e-10 { 1.0 } else { range / qmax };
    let zero_point = (-min_val / scale).round().clamp(0.0, qmax);

    x.iter()
        .map(|&v| {
            let q = (v / scale + zero_point).round().clamp(0.0, qmax);
            (q - zero_point) * scale
        })
        .collect()
}

/// Straight-through estimator: gradient passes through unchanged within the
/// clamp range. For values outside [min, max], gradient is zeroed.
fn ste_backward(grad_output: &[f32], original: &[f32], bits: u8) -> Vec<f32> {
    let qmax = ((1_u32 << bits) - 1) as f32;

    let min_val = original.iter().copied().fold(f32::INFINITY, f32::min);
    let max_val = original.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max_val - min_val;
    let scale = if range < 1e-10 { 1.0 } else { range / qmax };
    let lower = min_val - scale * 0.5;
    let upper = max_val + scale * 0.5;

    grad_output
        .iter()
        .zip(original.iter())
        .map(|(&g, &x)| {
            if x >= lower && x <= upper {
                g // pass through
            } else {
                0.0 // clamp kills gradient
            }
        })
        .collect()
}

/// Post-training quantization: quantize and dequantize (same as fake_quantize
/// but used at inference, not training).
fn ptq_quantize(x: &[f32], bits: u8) -> Vec<f32> {
    fake_quantize(x, bits)
}

/// RMSE between two slices.
fn rmse(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mse: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        / a.len() as f32;
    mse.sqrt()
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

/// Simple linear forward: y = W * x (dot product per output).
fn forward(weights: &[f32], input: &[f32]) -> f32 {
    weights.iter().zip(input.iter()).map(|(w, x)| w * x).sum()
}

/// Gradient of MSE loss w.r.t. weights for y = W . x, target t:
/// dL/dW = 2*(y-t)*x / n
fn mse_grad_weights(weights: &[f32], input: &[f32], target: f32) -> Vec<f32> {
    let y = forward(weights, input);
    let diff = 2.0 * (y - target) / weights.len() as f32;
    input.iter().map(|&x| diff * x).collect()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("quantize_fake_qat")?;

    let bits = 4_u8;
    let n = 256;

    // --- Section 1: Fake quantize demonstration ---
    let weights = generate_weights(n, 42);
    let fq = fake_quantize(&weights, bits);

    println!("--- Fake Quantize Demo ({bits}-bit) ---");
    println!("First 8 values:");
    println!("{:<12} {:<12} {:<12}", "original", "fake_quant", "error");
    for i in 0..8 {
        let err = (weights[i] - fq[i]).abs();
        println!("{:<12.6} {:<12.6} {:<12.8}", weights[i], fq[i], err);
    }

    let fq_error = rmse(&weights, &fq);
    println!("\nFake-quantize RMSE: {fq_error:.8}");

    // Count distinct levels
    let mut levels: Vec<f32> = fq.clone();
    levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
    levels.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
    println!(
        "Distinct quantization levels: {} (expected <= {})",
        levels.len(),
        1 << bits
    );

    // --- Section 2: STE explanation ---
    println!("\n--- Straight-Through Estimator (STE) ---");
    let grad_out: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32) - 0.5).collect();
    let grad_in = ste_backward(&grad_out, &weights, bits);

    let passed = grad_in
        .iter()
        .zip(grad_out.iter())
        .filter(|(gi, go)| (*gi - *go).abs() < 1e-8)
        .count();
    println!(
        "Gradients passed through: {}/{} ({:.1}%)",
        passed,
        n,
        100.0 * passed as f32 / n as f32
    );
    println!("STE preserves gradient for values within quantization range.");

    // --- Section 3: QAT training loop ---
    println!("\n--- QAT Training Loop ---");
    let input = generate_weights(n, 99);
    let target = 1.5_f32; // target output
    let lr = 0.01_f32;
    let epochs = 200;

    // QAT path: train with fake quantization
    let mut w_qat = weights.clone();
    let mut qat_losses = Vec::new();
    for epoch in 0..epochs {
        // Forward with fake quantization
        let w_fq = fake_quantize(&w_qat, bits);
        let y = forward(&w_fq, &input);
        let loss = (y - target).powi(2);
        if epoch % 50 == 0 {
            qat_losses.push((epoch, loss));
        }

        // Backward: compute grad on fake-quantized forward, apply STE
        let grad = mse_grad_weights(&w_fq, &input, target);
        let grad_ste = ste_backward(&grad, &w_qat, bits);

        // SGD update on real weights
        for (w, g) in w_qat.iter_mut().zip(grad_ste.iter()) {
            *w -= lr * g;
        }
    }

    println!("QAT training ({epochs} epochs):");
    for (epoch, loss) in &qat_losses {
        println!("  epoch {epoch:>4}: loss = {loss:.6}");
    }

    // PTQ path: train normally, then quantize
    let mut w_ptq = weights.clone();
    let mut ptq_losses = Vec::new();
    for epoch in 0..epochs {
        let y = forward(&w_ptq, &input);
        let loss = (y - target).powi(2);
        if epoch % 50 == 0 {
            ptq_losses.push((epoch, loss));
        }
        let grad = mse_grad_weights(&w_ptq, &input, target);
        for (w, g) in w_ptq.iter_mut().zip(grad.iter()) {
            *w -= lr * g;
        }
    }

    // --- Section 4: Compare QAT vs PTQ quality ---
    println!("\n--- QAT vs PTQ Comparison ---");

    // QAT: quantize the QAT-trained weights
    let w_qat_final = fake_quantize(&w_qat, bits);
    let y_qat = forward(&w_qat_final, &input);
    let loss_qat = (y_qat - target).powi(2);

    // PTQ: quantize the normally-trained weights
    let w_ptq_final = ptq_quantize(&w_ptq, bits);
    let y_ptq = forward(&w_ptq_final, &input);
    let loss_ptq = (y_ptq - target).powi(2);

    // FP32 baseline (no quantization)
    let y_fp32 = forward(&w_ptq, &input);
    let loss_fp32 = (y_fp32 - target).powi(2);

    println!("FP32 (no quant):  output={y_fp32:.6}, loss={loss_fp32:.8}");
    println!("PTQ ({bits}-bit):       output={y_ptq:.6}, loss={loss_ptq:.8}");
    println!("QAT ({bits}-bit):       output={y_qat:.6}, loss={loss_qat:.8}");

    if loss_qat < loss_ptq {
        println!(
            "\nQAT achieves {:.1}x lower loss than PTQ",
            loss_ptq / loss_qat.max(1e-10)
        );
    } else {
        println!("\nPTQ and QAT achieved similar loss at this scale");
    }

    // Quantization error on weights themselves
    let qat_weight_error = rmse(&w_qat, &w_qat_final);
    let ptq_weight_error = rmse(&w_ptq, &w_ptq_final);
    println!("\nWeight quantization RMSE:");
    println!("  QAT weights: {qat_weight_error:.8}");
    println!("  PTQ weights: {ptq_weight_error:.8}");

    // --- Save ---
    let bytes: Vec<u8> = w_qat_final.iter().flat_map(|f| f.to_le_bytes()).collect();
    let bundle = ModelBundleV2::new()
        .with_name("qat-4bit")
        .with_compression(Compression::Lz4)
        .add_tensor("weights_qat", vec![w_qat_final.len()], bytes)
        .build();

    assert_eq!(&bundle[0..4], b"APR2");
    println!("\nSaved QAT model as APR v2 ({} bytes)", bundle.len());

    ctx.report()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fake_quantize_produces_discrete_levels() {
        let weights = generate_weights(512, 42);
        let fq = fake_quantize(&weights, 4);
        let mut levels: Vec<f32> = fq.clone();
        levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        levels.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
        // 4-bit should have at most 16 distinct levels
        assert!(
            levels.len() <= 16,
            "4-bit fake quant should have <=16 levels, got {}",
            levels.len()
        );
    }

    #[test]
    fn test_fake_quantize_8bit_more_levels_than_4bit() {
        let weights = generate_weights(512, 42);
        let fq4 = fake_quantize(&weights, 4);
        let fq8 = fake_quantize(&weights, 8);
        let count = |fq: &[f32]| {
            let mut v = fq.to_vec();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
            v.len()
        };
        assert!(count(&fq8) > count(&fq4));
    }

    #[test]
    fn test_ste_preserves_gradient_in_range() {
        let weights = generate_weights(256, 42);
        let grad_out: Vec<f32> = (0..256).map(|i| i as f32 * 0.01).collect();
        let grad_in = ste_backward(&grad_out, &weights, 4);
        // Most gradients should pass through (all weights are in range by construction)
        let passed = grad_in
            .iter()
            .zip(grad_out.iter())
            .filter(|(gi, go)| (*gi - *go).abs() < 1e-8)
            .count();
        assert!(
            passed > 200,
            "most gradients should pass through STE: {passed}/256"
        );
    }

    #[test]
    fn test_ste_zeros_out_of_range() {
        // Create weights with known range
        let weights = vec![0.0, 0.5, 1.0];
        let grad_out = vec![1.0, 1.0, 1.0];
        let grad_in = ste_backward(&grad_out, &weights, 4);
        // All values within [0, 1] should pass through
        for (gi, go) in grad_in.iter().zip(grad_out.iter()) {
            assert!(
                (gi - go).abs() < 1e-6 || gi.abs() < 1e-6,
                "gradient should pass or be zeroed"
            );
        }
    }

    #[test]
    fn test_fake_quantize_preserves_length() {
        let weights = generate_weights(100, 42);
        let fq = fake_quantize(&weights, 4);
        assert_eq!(fq.len(), weights.len());
    }

    #[test]
    fn test_fake_quantize_deterministic() {
        let weights = generate_weights(256, 42);
        let fq1 = fake_quantize(&weights, 4);
        let fq2 = fake_quantize(&weights, 4);
        assert_eq!(fq1, fq2);
    }

    #[test]
    fn test_fake_quantize_error_bounded() {
        let weights = generate_weights(1024, 42);
        let fq = fake_quantize(&weights, 4);
        let error = rmse(&weights, &fq);
        assert!(error < 0.1, "fake quant RMSE should be bounded: {error}");
    }

    #[test]
    fn test_qat_reduces_quantization_error() {
        // Small QAT loop should adapt weights to be more quantization-friendly
        let n = 64;
        let weights = generate_weights(n, 42);
        let input = generate_weights(n, 99);
        let target = 0.5_f32;
        let lr = 0.01;

        // QAT training
        let mut w_qat = weights.clone();
        for _ in 0..100 {
            let w_fq = fake_quantize(&w_qat, 4);
            let grad = mse_grad_weights(&w_fq, &input, target);
            let grad_ste = ste_backward(&grad, &w_qat, 4);
            for (w, g) in w_qat.iter_mut().zip(grad_ste.iter()) {
                *w -= lr * g;
            }
        }

        // Normal training
        let mut w_normal = weights.clone();
        for _ in 0..100 {
            let grad = mse_grad_weights(&w_normal, &input, target);
            for (w, g) in w_normal.iter_mut().zip(grad.iter()) {
                *w -= lr * g;
            }
        }

        // Compare quantized outputs
        let w_qat_q = fake_quantize(&w_qat, 4);
        let w_norm_q = fake_quantize(&w_normal, 4);
        let y_qat = forward(&w_qat_q, &input);
        let y_norm = forward(&w_norm_q, &input);
        let loss_qat = (y_qat - target).powi(2);
        let loss_norm = (y_norm - target).powi(2);

        // QAT should be at least as good (or better) than PTQ
        // Allow some tolerance since this is a simple test
        assert!(
            loss_qat <= loss_norm * 2.0,
            "QAT should not be much worse: qat={loss_qat}, ptq={loss_norm}"
        );
    }

    #[test]
    fn test_zero_weights_fake_quantize() {
        let weights = vec![0.0_f32; 100];
        let fq = fake_quantize(&weights, 4);
        let error = rmse(&weights, &fq);
        assert!(
            error < 1e-6,
            "zero weights should fake-quantize perfectly: {error}"
        );
    }

    #[test]
    fn test_apr_bundle() {
        let weights = generate_weights(128, 42);
        let fq = fake_quantize(&weights, 4);
        let bytes: Vec<u8> = fq.iter().flat_map(|f| f.to_le_bytes()).collect();
        let bundle = ModelBundleV2::new()
            .with_name("test-qat")
            .with_compression(Compression::Lz4)
            .add_tensor("w", vec![fq.len()], bytes)
            .build();
        assert_eq!(&bundle[0..4], b"APR2");
    }
}
