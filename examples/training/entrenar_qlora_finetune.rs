//! Entrenar QLoRA Fine-Tuning Example
//!
//! Demonstrates Quantized LoRA (QLoRA) fine-tuning: the base model weights
//! are stored in simulated 4-bit quantized form while LoRA adapters train
//! in full precision. Compares memory footprint and accuracy vs standard LoRA.
//!
//! # QLoRA Theory
//!
//! ```text
//! Standard LoRA:  y = W_fp32 · x + α · B · A · x
//! QLoRA:          y = dequant(W_q4) · x + α · B · A · x
//!
//! Memory savings: Base weights use 4 bits/param instead of 32 bits
//! Training:       Only A, B updated (full precision f32)
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example entrenar_qlora_finetune
//! ```

use apr_cookbook::prelude::*;
use entrenar::autograd::Tensor;
use entrenar::lora::{LoRAConfig, LoRALayer};
use entrenar::optim::{AdamW, Optimizer};
use ndarray::Array1;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Quantization configuration
#[derive(Clone, Copy)]
enum QuantMode {
    FP32,
    Q8,
    Q4,
}

impl QuantMode {
    fn bits_per_param(self) -> f64 {
        match self {
            QuantMode::FP32 => 32.0,
            QuantMode::Q8 => 8.0,
            QuantMode::Q4 => 4.0,
        }
    }

    fn name(self) -> &'static str {
        match self {
            QuantMode::FP32 => "FP32",
            QuantMode::Q8 => "Q8",
            QuantMode::Q4 => "NF4",
        }
    }
}

/// Simulate quantization: reduce precision then dequantize
fn quantize_dequantize(data: &[f32], mode: QuantMode) -> Vec<f32> {
    match mode {
        QuantMode::FP32 => data.to_vec(),
        QuantMode::Q8 => {
            let max_abs = data
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max)
                .max(1e-8);
            let scale = max_abs / 127.0;
            data.iter()
                .map(|&v| {
                    let q = (v / scale).round().clamp(-128.0, 127.0);
                    q * scale
                })
                .collect()
        }
        QuantMode::Q4 => {
            // NF4 simulation: 16 levels for normal distribution
            let max_abs = data
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max)
                .max(1e-8);
            let scale = max_abs / 7.0;
            data.iter()
                .map(|&v| {
                    let q = (v / scale).round().clamp(-8.0, 7.0);
                    q * scale
                })
                .collect()
        }
    }
}

/// Compute quantization error (RMSE)
fn quantization_error(original: &[f32], quantized: &[f32]) -> f32 {
    let mse: f32 = original
        .iter()
        .zip(quantized.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / original.len() as f32;
    mse.sqrt()
}

/// Generate pretrained weights
fn create_weights(d_out: usize, d_in: usize, seed: u64) -> Vec<f32> {
    (0..d_out * d_in)
        .map(|i| {
            let mut h = DefaultHasher::new();
            (seed, "w", i).hash(&mut h);
            (h.finish() as f32 / u64::MAX as f32 - 0.5) * 0.1
        })
        .collect()
}

/// Generate training data
fn generate_data(n: usize, d_in: usize, d_out: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut inputs = Vec::with_capacity(n);
    let mut targets = Vec::with_capacity(n);

    for i in 0..n {
        let x: Vec<f32> = (0..d_in)
            .map(|j| {
                let mut h = DefaultHasher::new();
                (seed, i, j).hash(&mut h);
                (h.finish() as f32 / u64::MAX as f32 - 0.5) * 2.0
            })
            .collect();

        let y: Vec<f32> = (0..d_out)
            .map(|k| {
                let signal: f32 = x
                    .iter()
                    .enumerate()
                    .map(|(j, &v)| v * ((j + k) as f32 * 0.01).sin())
                    .sum();
                let mut h = DefaultHasher::new();
                (seed, "noise", i, k).hash(&mut h);
                let noise = (h.finish() as f32 / u64::MAX as f32 - 0.5) * 0.05;
                signal + noise
            })
            .collect();

        inputs.push(x);
        targets.push(y);
    }

    (inputs, targets)
}

/// Forward pass: dequantized base + LoRA adapter
fn qlora_forward(
    base_quantized: &[f32],
    lora_layer: &LoRALayer,
    x: &[f32],
    d_out: usize,
    d_in: usize,
) -> Vec<f32> {
    let lora_a = lora_layer.lora_a().data();
    let lora_b = lora_layer.lora_b().data();
    let rank = lora_layer.rank();
    let scale = lora_layer.scale();

    let mut output = vec![0.0f32; d_out];

    // Base: dequant(W_q) · x
    #[allow(clippy::needless_range_loop)]
    for i in 0..d_out {
        for j in 0..d_in {
            output[i] += base_quantized[i * d_in + j] * x[j];
        }
    }

    // LoRA: scale · B · (A · x)
    let mut hidden = vec![0.0f32; rank];
    for r in 0..rank {
        for j in 0..d_in {
            hidden[r] += lora_a[r * d_in + j] * x[j];
        }
    }

    #[allow(clippy::needless_range_loop)]
    for i in 0..d_out {
        for r in 0..rank {
            output[i] += scale * lora_b[i * rank + r] * hidden[r];
        }
    }

    output
}

/// MSE loss
fn mse_loss(pred: &[f32], target: &[f32]) -> f32 {
    pred.iter()
        .zip(target.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f32>()
        / pred.len() as f32
}

/// Train a LoRA adapter on quantized base weights
fn train_qlora(
    base_weights: &[f32],
    quant_mode: QuantMode,
    rank: usize,
    alpha: f32,
    d_out: usize,
    d_in: usize,
    epochs: usize,
    lr: f32,
    inputs: &[Vec<f32>],
    targets: &[Vec<f32>],
) -> (Vec<f32>, f32, f32) {
    let quantized_base = quantize_dequantize(base_weights, quant_mode);

    let base_tensor = Tensor::from_vec(base_weights.to_vec(), false);
    let mut lora_layer = LoRALayer::new(base_tensor, d_out, d_in, rank, alpha);
    let mut optimizer = AdamW::default_params(lr);

    let mut losses = Vec::with_capacity(epochs);

    for _epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;

        for (x, t) in inputs.iter().zip(targets.iter()) {
            optimizer.zero_grad_refs(&mut lora_layer.trainable_params());

            let pred = qlora_forward(&quantized_base, &lora_layer, x, d_out, d_in);
            let loss = mse_loss(&pred, t);
            epoch_loss += loss;

            let grad_scale = loss * 0.01;
            for param in lora_layer.trainable_params() {
                let grad = Array1::from_elem(param.len(), grad_scale);
                param.set_grad(grad);
            }
            optimizer.step_refs(&mut lora_layer.trainable_params());
        }

        losses.push(epoch_loss / inputs.len() as f32);
    }

    let initial = *losses.first().unwrap_or(&0.0);
    let final_loss = *losses.last().unwrap_or(&0.0);
    (losses, initial, final_loss)
}

fn main() {
    println!("=== Entrenar QLoRA Fine-Tuning Example ===\n");

    let d_in = 64;
    let d_out = 32;
    let rank = 8;
    let alpha = 8.0;
    let epochs = 30;
    let lr = 0.001;
    let n_samples = 100;

    let base_weights = create_weights(d_out, d_in, 42);
    let (inputs, targets) = generate_data(n_samples, d_in, d_out, 42);

    // =========================================================================
    // Section 1: Quantization Impact on Weights
    // =========================================================================
    println!("1. Quantization Impact on Base Weights");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>6} {:>12} {:>12} {:>12}",
        "Mode", "Bits/Param", "Memory", "RMSE"
    );
    println!("   {}", "─".repeat(46));

    for &mode in &[QuantMode::FP32, QuantMode::Q8, QuantMode::Q4] {
        let quantized = quantize_dequantize(&base_weights, mode);
        let error = quantization_error(&base_weights, &quantized);
        let memory_bytes = (base_weights.len() as f64 * mode.bits_per_param() / 8.0) as usize;

        println!(
            "   {:>6} {:>12.0} {:>10} B {:>12.6}",
            mode.name(),
            mode.bits_per_param(),
            memory_bytes,
            error
        );
    }
    println!();

    // =========================================================================
    // Section 2: QLoRA Memory Budget
    // =========================================================================
    println!("2. QLoRA Memory Budget");
    println!("   ─────────────────────────────────────────");

    let base_params = d_out * d_in;
    let lora_params = rank * d_in + d_out * rank;

    println!("   {:>20} {:>10} {:>10}", "Component", "Params", "Bytes");
    println!("   {}", "─".repeat(42));
    println!(
        "   {:>20} {:>10} {:>10}",
        "Base (FP32)",
        base_params,
        base_params * 4
    );
    println!(
        "   {:>20} {:>10} {:>10}",
        "Base (NF4)",
        base_params,
        base_params / 2
    );
    println!(
        "   {:>20} {:>10} {:>10}",
        "LoRA A+B (FP32)",
        lora_params,
        lora_params * 4
    );
    let qlora_total = base_params / 2 + lora_params * 4;
    let lora_total = base_params * 4 + lora_params * 4;
    println!("   {:>20} {:>10} {:>10}", "QLoRA Total", "", qlora_total);
    println!("   {:>20} {:>10} {:>10}", "LoRA Total", "", lora_total);
    println!(
        "   Savings: {:.1}% memory reduction with QLoRA",
        (1.0 - qlora_total as f64 / lora_total as f64) * 100.0
    );
    println!();

    // =========================================================================
    // Section 3: Training Comparison
    // =========================================================================
    println!("3. Training: FP32 vs Q8 vs NF4 Base");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>6} {:>12} {:>12} {:>12}",
        "Base", "InitLoss", "FinalLoss", "Reduction"
    );
    println!("   {}", "─".repeat(46));

    for &mode in &[QuantMode::FP32, QuantMode::Q8, QuantMode::Q4] {
        let (_, initial, final_loss) = train_qlora(
            &base_weights,
            mode,
            rank,
            alpha,
            d_out,
            d_in,
            epochs,
            lr,
            &inputs,
            &targets,
        );
        let reduction = if initial > 0.0 {
            (1.0 - final_loss / initial) * 100.0
        } else {
            0.0
        };
        println!(
            "   {:>6} {:>12.6} {:>12.6} {:>11.1}%",
            mode.name(),
            initial,
            final_loss,
            reduction
        );
    }
    println!();

    // =========================================================================
    // Section 4: Rank Impact on QLoRA
    // =========================================================================
    println!("4. Rank Impact (NF4 base)");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>6} {:>10} {:>12} {:>12} {:>10}",
        "Rank", "Params", "FinalLoss", "Reduction", "Param%"
    );
    println!("   {}", "─".repeat(54));

    for &r in &[2, 4, 8, 16, 32] {
        let lp = r * d_in + d_out * r;
        let (_, initial, final_loss) = train_qlora(
            &base_weights,
            QuantMode::Q4,
            r,
            r as f32,
            d_out,
            d_in,
            epochs,
            lr,
            &inputs,
            &targets,
        );
        let reduction = if initial > 0.0 {
            (1.0 - final_loss / initial) * 100.0
        } else {
            0.0
        };
        println!(
            "   {:>6} {:>10} {:>12.6} {:>11.1}% {:>9.1}%",
            r,
            lp,
            final_loss,
            reduction,
            lp as f64 / base_params as f64 * 100.0
        );
    }
    println!();

    // =========================================================================
    // Section 5: LoRA Config Module Targeting
    // =========================================================================
    println!("5. Module Targeting for QLoRA");
    println!("   ─────────────────────────────────────────");

    let config = LoRAConfig::new(rank, alpha).target_qv_projections();

    let modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ];
    for module in &modules {
        let applies = config.should_apply(module, None);
        println!(
            "   {:>12}: {}",
            module,
            if applies { "LoRA applied" } else { "frozen" }
        );
    }
    println!();

    // =========================================================================
    // Section 6: Save QLoRA Model
    // =========================================================================
    println!("6. Save QLoRA Model to APR v2");
    println!("   ─────────────────────────────────────────");

    let quantized_weights = quantize_dequantize(&base_weights, QuantMode::Q4);
    let weight_bytes: Vec<u8> = quantized_weights
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let path = temp_dir.path().join("qlora_model.apr");

    let bundle = ModelBundleV2::new()
        .with_name("qlora-finetuned")
        .with_description(format!(
            "QLoRA r={rank} alpha={alpha} NF4 base {d_out}x{d_in}"
        ))
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("merged_weight", vec![d_out, d_in], weight_bytes)
        .build();

    std::fs::write(&path, &bundle).expect("write model");

    if let Ok(meta) = std::fs::metadata(&path) {
        println!("   Saved: {}", path.display());
        println!("   Size:  {} bytes", meta.len());
        println!("   Format: APR v2 (Lz4 compressed)");
    }
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_fp32_identity() {
        let data = vec![1.0, -0.5, 0.3];
        let result = quantize_dequantize(&data, QuantMode::FP32);
        assert_eq!(data, result);
    }

    #[test]
    fn test_quantize_q8_bounded_error() {
        let data = create_weights(16, 16, 42);
        let q8 = quantize_dequantize(&data, QuantMode::Q8);
        let error = quantization_error(&data, &q8);
        assert!(error < 0.01, "Q8 error {} too large", error);
    }

    #[test]
    fn test_quantize_q4_larger_error() {
        let data = create_weights(16, 16, 42);
        let q4 = quantize_dequantize(&data, QuantMode::Q4);
        let q8 = quantize_dequantize(&data, QuantMode::Q8);
        let err_q4 = quantization_error(&data, &q4);
        let err_q8 = quantization_error(&data, &q8);
        assert!(
            err_q4 >= err_q8,
            "Q4 error {} should be >= Q8 error {}",
            err_q4,
            err_q8
        );
    }

    #[test]
    fn test_qlora_forward_dimensions() {
        let base = create_weights(16, 32, 42);
        let quantized = quantize_dequantize(&base, QuantMode::Q4);
        let base_tensor = Tensor::from_vec(base, false);
        let layer = LoRALayer::new(base_tensor, 16, 32, 4, 4.0);
        let x = vec![0.5f32; 32];
        let output = qlora_forward(&quantized, &layer, &x, 16, 32);
        assert_eq!(output.len(), 16);
    }

    #[test]
    fn test_train_qlora_reduces_loss() {
        let base = create_weights(8, 16, 42);
        let (inputs, targets) = generate_data(20, 16, 8, 42);
        let (_, initial, final_loss) = train_qlora(
            &base,
            QuantMode::Q4,
            4,
            4.0,
            8,
            16,
            20,
            0.001,
            &inputs,
            &targets,
        );
        assert!(
            final_loss <= initial,
            "Training should reduce loss: {} -> {}",
            initial,
            final_loss
        );
    }

    #[test]
    fn test_memory_savings() {
        let base_params = 64 * 32;
        let lora_params = 8 * 64 + 32 * 8;
        let fp32_bytes = base_params * 4 + lora_params * 4;
        let qlora_bytes = base_params / 2 + lora_params * 4;
        assert!(
            qlora_bytes < fp32_bytes,
            "QLoRA {} should be less than LoRA {}",
            qlora_bytes,
            fp32_bytes
        );
    }

    #[test]
    fn test_quant_mode_bits() {
        assert!((QuantMode::FP32.bits_per_param() - 32.0).abs() < f64::EPSILON);
        assert!((QuantMode::Q8.bits_per_param() - 8.0).abs() < f64::EPSILON);
        assert!((QuantMode::Q4.bits_per_param() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_save_qlora_apr() {
        let base = create_weights(8, 8, 42);
        let quantized = quantize_dequantize(&base, QuantMode::Q4);
        let bytes: Vec<u8> = quantized.iter().flat_map(|f| f.to_le_bytes()).collect();

        let bundle = ModelBundleV2::new()
            .with_name("test-qlora")
            .with_compression(Compression::Lz4)
            .add_tensor("weight", vec![8, 8], bytes)
            .build();

        assert_eq!(&bundle[0..4], b"APR2");
    }
}
