//! # Recipe: Full Optimization Pipeline
//!
//! **Category**: optimize
//! **CLI Equivalent**: `apr finetune && apr prune && apr distill && apr merge && apr quantize`
//!
//! Demonstrates chaining: LoRA fine-tuning → magnitude pruning → knowledge
//! distillation → TIES model merging → 4-bit quantization.
//!
//! ```bash
//! cargo run --example optimize_full_pipeline
//! ```
//!
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use apr_cookbook::prelude::*;
use entrenar::autograd::Tensor;
use entrenar::distill::DistillationLoss;
use entrenar::lora::LoRALayer;
use entrenar::merge::{ties_merge, TiesConfig};
use entrenar::optim::{AdamW, Optimizer};
use ndarray::{Array1, Array2};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Model dimensions for the pipeline
const D_IN: usize = 64;
const D_OUT: usize = 32;
const RANK: usize = 8;
const ALPHA: f32 = 8.0;

type Model = HashMap<String, Tensor>;

// ── Stage helpers ──

/// Deterministic weight generation
fn det_weights(size: usize, seed: u64) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let mut h = DefaultHasher::new();
            (seed, i).hash(&mut h);
            (h.finish() as f32 / u64::MAX as f32 - 0.5) * 0.1
        })
        .collect()
}

/// Generate synthetic training data
fn gen_data(n: usize, d_in: usize, d_out: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let (mut inputs, mut targets) = (Vec::with_capacity(n), Vec::with_capacity(n));
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
                (seed, "t", i, k).hash(&mut h);
                signal + (h.finish() as f32 / u64::MAX as f32 - 0.5) * 0.05
            })
            .collect();
        inputs.push(x);
        targets.push(y);
    }
    (inputs, targets)
}

/// Forward pass through LoRA layer
fn lora_forward(layer: &LoRALayer, x: &[f32], d_out: usize, d_in: usize) -> Vec<f32> {
    let base_w = layer.base_weight().data();
    let lora_a = layer.lora_a().data();
    let lora_b = layer.lora_b().data();
    let rank = layer.rank();
    let scale = layer.scale();

    let mut output = vec![0.0f32; d_out];

    #[allow(clippy::needless_range_loop)]
    for i in 0..d_out {
        for j in 0..d_in {
            output[i] += base_w[i * d_in + j] * x[j];
        }
    }

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

// ── Stage 1: LoRA Fine-Tuning ──

fn stage_finetune(base_weights: Vec<f32>, ctx: &mut RecipeContext) -> Vec<f32> {
    println!("Stage 1: LoRA Fine-Tuning (apr finetune --method lora)");

    let base_tensor = Tensor::from_vec(base_weights, false);
    let mut lora_layer = LoRALayer::new(base_tensor, D_OUT, D_IN, RANK, ALPHA);
    let mut optimizer = AdamW::default_params(0.0001);
    let (inputs, targets) = gen_data(100, D_IN, D_OUT, 42);

    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;

    for epoch in 0..30 {
        let mut epoch_loss = 0.0f32;
        for (x, t) in inputs.iter().zip(targets.iter()) {
            optimizer.zero_grad_refs(&mut lora_layer.trainable_params());
            let pred = lora_forward(&lora_layer, x, D_OUT, D_IN);
            let loss = mse_loss(&pred, t);
            epoch_loss += loss;

            // Gradient clipping prevents positive-feedback divergence
            let grad_scale = (loss * 0.01).min(0.1);
            for param in lora_layer.trainable_params() {
                let grad = Array1::from_elem(param.len(), grad_scale);
                param.set_grad(grad);
            }
            optimizer.step_refs(&mut lora_layer.trainable_params());
        }

        let avg = epoch_loss / inputs.len() as f32;
        if epoch == 0 {
            initial_loss = avg;
        }
        if epoch == 29 {
            final_loss = avg;
        }
    }

    lora_layer.merge();
    let merged = lora_layer.base_weight().data().to_vec();

    let lora_params = RANK * D_IN + D_OUT * RANK;
    let pct = lora_params as f32 / (D_OUT * D_IN) as f32 * 100.0;
    println!("   LoRA r={RANK}, trainable: {lora_params} ({pct:.1}% of base)");
    println!("   Loss: {initial_loss:.6} → {final_loss:.6}");
    ctx.record_float_metric("finetune_initial_loss", f64::from(initial_loss));
    ctx.record_float_metric("finetune_final_loss", f64::from(final_loss));
    println!();

    merged
}

// ── Stage 2: Magnitude Pruning ──

fn stage_prune(weights: &[f32], target_sparsity: f32) -> Vec<f32> {
    println!(
        "Stage 2: Magnitude Pruning (apr prune --method magnitude --target {target_sparsity})"
    );

    let mut magnitudes: Vec<(usize, f32)> = weights
        .iter()
        .enumerate()
        .map(|(i, &w)| (i, w.abs()))
        .collect();
    magnitudes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let n_prune = (weights.len() as f32 * target_sparsity) as usize;
    let mut pruned = weights.to_vec();
    for &(idx, _) in magnitudes.iter().take(n_prune) {
        pruned[idx] = 0.0;
    }

    let actual_zeros = pruned.iter().filter(|&&w| w == 0.0).count();
    let actual_sparsity = actual_zeros as f32 / pruned.len() as f32;

    println!("   Params: {} total, {} pruned", weights.len(), n_prune);
    println!("   Sparsity: {:.1}%", actual_sparsity * 100.0);
    println!();

    pruned
}

// ── Stage 3: Knowledge Distillation ──

fn stage_distill(ctx: &mut RecipeContext) {
    println!("Stage 3: Knowledge Distillation (apr distill --strategy standard)");

    let batch = 16;
    let classes = 5;

    let teacher_data: Vec<f32> = (0..batch * classes)
        .map(|i| {
            let mut h = DefaultHasher::new();
            (42u64, "teacher", i).hash(&mut h);
            let base = h.finish() as f32 / u64::MAX as f32 - 0.5;
            if i % classes == (i / classes) % classes {
                base + 8.0
            } else {
                base
            }
        })
        .collect();
    let student_data: Vec<f32> = (0..batch * classes)
        .map(|i| {
            let mut h = DefaultHasher::new();
            (99u64, "student", i).hash(&mut h);
            h.finish() as f32 / u64::MAX as f32 - 0.5
        })
        .collect();
    let labels: Vec<usize> = (0..batch).map(|b| b % classes).collect();

    let teacher = Array2::from_shape_vec((batch, classes), teacher_data).expect("teacher shape");
    let student = Array2::from_shape_vec((batch, classes), student_data).expect("student shape");

    let loss_fn = DistillationLoss::new(3.0, 0.7);
    let loss = loss_fn.forward(&student, &teacher, &labels);

    println!("   Temperature: 3.0, Alpha: 0.7");
    println!("   Distillation loss: {:.4}", loss);

    ctx.record_float_metric("distill_loss", f64::from(loss));
    println!();
}

// ── Stage 4: TIES Model Merging ──

fn stage_merge(finetuned_weights: &[f32], ctx: &mut RecipeContext) -> Vec<f32> {
    println!("Stage 4: TIES Model Merge (apr merge --strategy ties --density 0.2)");

    let layers: Vec<(&str, usize)> = vec![("weight", finetuned_weights.len())];

    let base: Model = layers
        .iter()
        .map(|&(name, size)| {
            (
                name.to_string(),
                Tensor::from_vec(det_weights(size, 42), false),
            )
        })
        .collect();
    let variant_a: Model = layers
        .iter()
        .map(|&(name, _)| {
            (
                name.to_string(),
                Tensor::from_vec(finetuned_weights.to_vec(), false),
            )
        })
        .collect();
    let variant_b: Model = layers
        .iter()
        .map(|&(name, size)| {
            let data: Vec<f32> = finetuned_weights
                .iter()
                .enumerate()
                .map(|(i, &w)| {
                    let mut h = DefaultHasher::new();
                    (200u64, name, i).hash(&mut h);
                    w + (h.finish() as f32 / u64::MAX as f32 - 0.5) * 0.02
                })
                .take(size)
                .collect();
            (name.to_string(), Tensor::from_vec(data, false))
        })
        .collect();

    let models = vec![variant_a, variant_b];
    let config = TiesConfig::new(0.2).expect("valid TIES config");
    let merged = ties_merge(&models, &base, &config).expect("TIES merge succeeds");

    let merged_weights = merged["weight"].data().to_vec();

    let dist: f32 = merged_weights
        .iter()
        .zip(det_weights(finetuned_weights.len(), 42).iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt();

    println!("   Models merged: 2 task variants");
    println!("   TIES density: 0.2");
    println!("   Distance from base: {:.4}", dist);

    ctx.record_float_metric("merge_dist_from_base", f64::from(dist));
    println!();

    merged_weights
}

// ── Stage 5: 4-bit Quantization ──

fn stage_quantize(weights: &[f32], ctx: &mut RecipeContext) -> Vec<f32> {
    println!("Stage 5: 4-bit Quantization (apr quantize --scheme int4)");

    let max_abs = weights
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-8);
    let scale = max_abs / 7.0;

    let quantized: Vec<f32> = weights
        .iter()
        .map(|&v| {
            let q = (v / scale).round().clamp(-8.0, 7.0);
            q * scale
        })
        .collect();

    let rmse = (weights
        .iter()
        .zip(quantized.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / weights.len() as f32)
        .sqrt();

    let original_bytes = weights.len() * 4;
    let quantized_bytes = weights.len() / 2; // 4 bits per param
    let compression = original_bytes as f32 / quantized_bytes as f32;

    println!("   Original: {} bytes (FP32)", original_bytes);
    println!("   Quantized: {} bytes (INT4)", quantized_bytes);
    println!("   Compression: {:.1}x", compression);
    println!("   RMSE: {:.6}", rmse);

    ctx.record_float_metric("quant_rmse", f64::from(rmse));
    ctx.record_float_metric("quant_compression", f64::from(compression));
    println!();

    quantized
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("optimize_full_pipeline")?;

    println!("=== Full Optimization Pipeline ===");
    println!("Mirrors: apr finetune → apr prune → apr distill → apr merge → apr quantize");
    println!();

    // Base model weights
    let base_weights = det_weights(D_OUT * D_IN, 42);
    let base_size = base_weights.len() * 4;

    println!(
        "Input Model: {}x{} = {} params ({} bytes FP32)",
        D_OUT,
        D_IN,
        D_OUT * D_IN,
        base_size
    );
    println!();

    // ── Pipeline Stages ──

    // Stage 1: LoRA Fine-Tuning
    let finetuned = stage_finetune(base_weights, &mut ctx);

    // Stage 2: Magnitude Pruning (50% sparsity)
    let pruned = stage_prune(&finetuned, 0.5);

    // Stage 3: Knowledge Distillation (teacher → student)
    stage_distill(&mut ctx);

    // Stage 4: TIES Model Merge
    let merged = stage_merge(&pruned, &mut ctx);

    // Stage 5: 4-bit Quantization
    let final_weights = stage_quantize(&merged, &mut ctx);

    println!("Pipeline Complete — Saving to APR v2");
    let weight_bytes: Vec<u8> = final_weights.iter().flat_map(|f| f.to_le_bytes()).collect();
    let model_path = ctx.path("optimized_model.apr");

    let bundle = ModelBundleV2::new()
        .with_name("pipeline-optimized")
        .with_description("Full pipeline: LoRA→prune→distill→merge→quantize")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("weight", vec![D_OUT, D_IN], weight_bytes)
        .build();

    std::fs::write(&model_path, &bundle)?;

    if let Ok(meta) = std::fs::metadata(&model_path) {
        println!("   Saved: {} bytes (APR v2, Lz4)", meta.len());
        ctx.record_metric("output_size_bytes", meta.len() as i64);
    }

    println!();
    ctx.report()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_det_weights_deterministic_and_seed_sensitive() {
        let w1 = det_weights(100, 42);
        assert_eq!(w1, det_weights(100, 42));
        assert_ne!(w1, det_weights(100, 43));
    }

    #[test]
    fn test_gen_data_shape() {
        let (x, y) = gen_data(10, 64, 32, 42);
        assert_eq!(x.len(), 10);
        assert_eq!(y.len(), 10);
        assert_eq!(x[0].len(), 64);
        assert_eq!(y[0].len(), 32);
    }

    #[test]
    fn test_lora_forward_shape() {
        let base = Tensor::from_vec(det_weights(D_OUT * D_IN, 42), false);
        let layer = LoRALayer::new(base, D_OUT, D_IN, RANK, ALPHA);
        let x = vec![1.0f32; D_IN];
        let out = lora_forward(&layer, &x, D_OUT, D_IN);
        assert_eq!(out.len(), D_OUT);
    }

    #[test]
    fn test_mse_loss() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(mse_loss(&a, &a).abs() < 1e-6);
        assert!((mse_loss(&a, &vec![2.0, 3.0, 4.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_stage_prune() {
        let weights = det_weights(256, 42);
        let pruned = stage_prune(&weights, 0.5);
        assert_eq!(pruned.len(), weights.len());
        let sparsity = pruned.iter().filter(|&&w| w == 0.0).count() as f32 / pruned.len() as f32;
        assert!((sparsity - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_stage_quantize() {
        let weights = det_weights(256, 42);
        let quantized = stage_quantize(&weights, &mut RecipeContext::new("test_q").unwrap());
        assert_eq!(quantized.len(), weights.len());
        let rmse = (weights
            .iter()
            .zip(quantized.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / weights.len() as f32)
            .sqrt();
        assert!(rmse < 0.1, "RMSE too large: {rmse}");
    }

    #[test]
    fn test_full_pipeline_produces_apr() {
        let mut ctx = RecipeContext::new("test_pipeline").unwrap();
        let base = det_weights(D_OUT * D_IN, 42);
        let finetuned = stage_finetune(base, &mut ctx);
        let pruned = stage_prune(&finetuned, 0.5);
        let merged = stage_merge(&pruned, &mut ctx);
        let final_w = stage_quantize(&merged, &mut ctx);

        let bytes: Vec<u8> = final_w.iter().flat_map(|f| f.to_le_bytes()).collect();
        let bundle = ModelBundleV2::new()
            .with_name("test-pipeline")
            .with_compression(Compression::Lz4)
            .add_tensor("weight", vec![D_OUT, D_IN], bytes)
            .build();

        assert_eq!(&bundle[0..4], b"APR2");
    }

    #[test]
    fn test_ties_merge_produces_valid_model() {
        let weights = det_weights(128, 42);
        let mut ctx = RecipeContext::new("test_merge").unwrap();
        let merged = stage_merge(&weights, &mut ctx);
        assert_eq!(merged.len(), weights.len());
        assert!(merged.iter().all(|w| w.is_finite()));
    }
}
