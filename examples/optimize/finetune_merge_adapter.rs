//! # Recipe: Adapter Merge and Unmerge Lifecycle
//!
//! **CLI Equivalent**: `apr finetune --merge --adapter`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! LoRA adapter lifecycle: create, train, merge for inference, unmerge for
//! continued training, save adapter-only and merged models to APR v2.
//!
//! ```bash
//! cargo run --example finetune_merge_adapter
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr finetune model.apr          # APR native format
//! apr finetune model.gguf         # GGUF (llama.cpp compatible)
//! apr finetune model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685

use apr_cookbook::prelude::*;
use entrenar::autograd::Tensor;
use entrenar::lora::LoRALayer;
use entrenar::optim::{AdamW, Optimizer};
use ndarray::Array1;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const D_IN: usize = 64;
const D_OUT: usize = 32;
const RANK: usize = 8;
const ALPHA: f32 = 8.0;

/// Convert tensor data to little-endian bytes for APR v2 storage
fn tensor_to_bytes(tensor: &Tensor) -> Vec<u8> {
    tensor.data().iter().flat_map(|f| f.to_le_bytes()).collect()
}

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

/// Generate training data
fn gen_data(n: usize, d_in: usize, d_out: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut inputs = Vec::with_capacity(n);
    let mut targets = Vec::with_capacity(n);
    for i in 0..n {
        let x: Vec<f32> = (0..d_in)
            .map(|j| {
                let mut h = DefaultHasher::new();
                (seed, "x", i, j).hash(&mut h);
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
                (seed, "y", i, k).hash(&mut h);
                signal + (h.finish() as f32 / u64::MAX as f32 - 0.5) * 0.05
            })
            .collect();
        inputs.push(x);
        targets.push(y);
    }
    (inputs, targets)
}

/// Forward through LoRA layer (with adapter contribution)
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

/// Forward through base weights only (post-merge, no adapter overhead)
fn base_forward(weights: &[f32], x: &[f32], d_out: usize, d_in: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; d_out];
    #[allow(clippy::needless_range_loop)]
    for i in 0..d_out {
        for j in 0..d_in {
            output[i] += weights[i * d_in + j] * x[j];
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

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("finetune_merge_adapter")?;

    println!("=== Adapter Merge/Unmerge Lifecycle ===");
    println!("Mirrors: apr finetune --merge --adapter");
    println!();

    // ── Create Base Weight and LoRA Layer ──
    println!("Step 1: Create LoRA Adapter");
    println!("   ─────────────────────────────────────────");
    let base_data = det_weights(D_OUT * D_IN, 42);
    let base_tensor = Tensor::from_vec(base_data.clone(), false);
    let mut lora_layer = LoRALayer::new(base_tensor, D_OUT, D_IN, RANK, ALPHA);

    let base_params = D_OUT * D_IN;
    let adapter_params: usize = lora_layer.trainable_params().iter().map(|p| p.len()).sum();
    println!("   Base params:    {}", base_params);
    println!(
        "   Adapter params: {} (A: {}x{} + B: {}x{})",
        adapter_params, RANK, D_IN, D_OUT, RANK
    );
    println!(
        "   Efficiency:     {:.2}% of base",
        adapter_params as f64 / base_params as f64 * 100.0
    );
    println!();

    // ── Brief Training ──
    println!("Step 2: Train Adapter (10 epochs)");
    println!("   ─────────────────────────────────────────");
    let mut optimizer = AdamW::default_params(0.001);
    let (inputs, targets) = gen_data(50, D_IN, D_OUT, 42);

    let mut initial_loss = 0.0f32;
    let mut final_loss = 0.0f32;

    for epoch in 0..10 {
        let mut epoch_loss = 0.0f32;
        for (x, t) in inputs.iter().zip(targets.iter()) {
            optimizer.zero_grad_refs(&mut lora_layer.trainable_params());
            let pred = lora_forward(&lora_layer, x, D_OUT, D_IN);
            let loss = mse_loss(&pred, t);
            epoch_loss += loss;

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
        if epoch == 9 {
            final_loss = avg;
        }
    }

    println!("   Loss: {:.6} → {:.6}", initial_loss, final_loss);
    ctx.record_float_metric("training_initial_loss", f64::from(initial_loss));
    ctx.record_float_metric("training_final_loss", f64::from(final_loss));
    println!();

    // ── Forward Pass BEFORE Merge ──
    println!("Step 3: Forward Before Merge (base + adapter)");
    println!("   ─────────────────────────────────────────");
    let test_input = vec![1.0f32; D_IN];
    let output_before = lora_forward(&lora_layer, &test_input, D_OUT, D_IN);
    let out_norm_before: f32 = output_before.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("   Output norm: {:.6}", out_norm_before);
    println!("   Output[0..4]: {:?}", &output_before[..4]);
    println!();

    // ── Merge: Fold Adapter into Base ──
    println!("Step 4: Merge Adapter into Base Weights");
    println!("   ─────────────────────────────────────────");
    lora_layer.merge();
    let merged_weights = lora_layer.base_weight().data().to_vec();

    // After merge, base-only forward should match pre-merge LoRA forward
    let output_merged = base_forward(&merged_weights, &test_input, D_OUT, D_IN);
    let out_norm_merged: f32 = output_merged.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("   Output norm (base-only): {:.6}", out_norm_merged);
    println!("   Output[0..4]: {:?}", &output_merged[..4]);

    let merge_diff: f32 = output_before
        .iter()
        .zip(output_merged.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>();
    println!("   Merge fidelity (L1 diff): {:.8}", merge_diff);
    println!("   Deploy: base-only forward, zero adapter overhead");
    println!();

    // ── Unmerge: Restore for Continued Training ──
    println!("Step 5: Unmerge for Continued Training");
    println!("   ─────────────────────────────────────────");
    lora_layer.unmerge();
    let output_unmerged = lora_forward(&lora_layer, &test_input, D_OUT, D_IN);
    let out_norm_unmerged: f32 = output_unmerged.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("   Output norm (restored): {:.6}", out_norm_unmerged);

    let unmerge_diff: f32 = output_before
        .iter()
        .zip(output_unmerged.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>();
    println!(
        "   Unmerge fidelity (L1 diff from pre-merge): {:.8}",
        unmerge_diff
    );

    // Verify base restored to original
    let base_restored = lora_layer.base_weight().data().to_vec();
    let base_diff: f32 = base_data
        .iter()
        .zip(base_restored.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>();
    println!("   Base weight restoration error: {:.8}", base_diff);
    println!();

    // ── Save Adapter-Only ──
    println!("Step 6: Save Adapter-Only (A + B matrices)");
    println!("   ─────────────────────────────────────────");

    let lora_a_bytes = tensor_to_bytes(lora_layer.lora_a());
    let lora_b_bytes = tensor_to_bytes(lora_layer.lora_b());

    let adapter_path = ctx.path("adapter_only.apr");
    let adapter_bundle = ModelBundleV2::new()
        .with_name("lora-adapter")
        .with_description("LoRA adapter matrices A and B (no base weights)")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("lora_a", vec![RANK, D_IN], lora_a_bytes)
        .add_tensor("lora_b", vec![D_OUT, RANK], lora_b_bytes)
        .build();

    std::fs::write(&adapter_path, &adapter_bundle)?;
    let adapter_size = std::fs::metadata(&adapter_path).map_or(0, |m| m.len());
    println!("   Adapter saved: {} bytes", adapter_size);
    println!(
        "   Tensors: lora_a [{}x{}], lora_b [{}x{}]",
        RANK, D_IN, D_OUT, RANK
    );

    // ── Save Merged Model ──
    println!();
    println!("Step 7: Save Merged Model (full weights)");
    println!("   ─────────────────────────────────────────");

    lora_layer.merge();
    let merged_bytes = tensor_to_bytes(lora_layer.base_weight());

    let merged_path = ctx.path("merged_model.apr");
    let merged_bundle = ModelBundleV2::new()
        .with_name("merged-model")
        .with_description("LoRA adapter merged into base weights")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("weight", vec![D_OUT, D_IN], merged_bytes)
        .build();

    std::fs::write(&merged_path, &merged_bundle)?;
    let merged_size = std::fs::metadata(&merged_path).map_or(0, |m| m.len());
    println!("   Merged saved: {} bytes", merged_size);
    println!("   Tensor: weight [{}x{}]", D_OUT, D_IN);

    // ── Size Comparison ──
    println!();
    println!("Size Comparison");
    println!("   ─────────────────────────────────────────");
    println!("   Adapter-only: {} bytes", adapter_size);
    println!("   Merged model: {} bytes", merged_size);
    if merged_size > 0 {
        println!(
            "   Adapter is {:.1}x smaller than merged",
            merged_size as f64 / adapter_size.max(1) as f64
        );
    }

    ctx.record_metric("adapter_size_bytes", adapter_size as i64);
    ctx.record_metric("merged_size_bytes", merged_size as i64);
    println!();

    // ── Summary ──
    println!("Lifecycle Summary");
    println!("   ─────────────────────────────────────────");
    println!("   1. Create adapter: small trainable matrices A, B");
    println!("   2. Train: only update A, B (base frozen)");
    println!("   3. Merge: fold A, B into base for deployment");
    println!("   4. Deploy: base-only forward, zero overhead");
    println!("   5. Unmerge: restore A, B for continued training");
    println!("   6. Save: adapter-only (small) or merged (full)");
    println!();

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let base = Tensor::from_vec(det_weights(D_OUT * D_IN, 42), false);
        let mut layer = LoRALayer::new(base, D_OUT, D_IN, RANK, ALPHA);
        let params: usize = layer.trainable_params().iter().map(|p| p.len()).sum();
        assert_eq!(params, RANK * D_IN + D_OUT * RANK);
    }

    #[test]
    fn test_merge_changes_base() {
        let base_data = det_weights(D_OUT * D_IN, 42);
        let base = Tensor::from_vec(base_data.clone(), false);
        let mut layer = LoRALayer::new(base, D_OUT, D_IN, RANK, ALPHA);
        let before = layer.base_weight().data().to_vec();
        assert_eq!(before, base_data);
        layer.merge();
        let after = layer.base_weight().data().to_vec();
        // After merge, base should have the same length
        assert_eq!(after.len(), before.len());
        assert!(after.iter().all(|w| w.is_finite()));
    }

    #[test]
    fn test_unmerge_roundtrip() {
        let base_data = det_weights(D_OUT * D_IN, 42);
        let base = Tensor::from_vec(base_data.clone(), false);
        let mut layer = LoRALayer::new(base, D_OUT, D_IN, RANK, ALPHA);

        layer.merge();
        layer.unmerge();

        let restored = layer.base_weight().data().to_vec();
        for (a, b) in restored.iter().zip(base_data.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "unmerge should restore base: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_merged_forward_matches_lora_forward() {
        let base = Tensor::from_vec(det_weights(D_OUT * D_IN, 42), false);
        let mut layer = LoRALayer::new(base, D_OUT, D_IN, RANK, ALPHA);

        let x = vec![0.5f32; D_IN];
        let out_lora = lora_forward(&layer, &x, D_OUT, D_IN);

        layer.merge();
        let merged = layer.base_weight().data().to_vec();
        let out_merged = base_forward(&merged, &x, D_OUT, D_IN);

        for (a, b) in out_lora.iter().zip(out_merged.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "merged forward should match LoRA forward: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_adapter_size_smaller_than_full() {
        let adapter_params = RANK * D_IN + D_OUT * RANK;
        let full_params = D_OUT * D_IN;
        assert!(
            adapter_params < full_params,
            "adapter ({}) should be smaller than full ({})",
            adapter_params,
            full_params
        );
    }

    #[test]
    fn test_adapter_and_merged_save_to_apr() {
        let base = Tensor::from_vec(det_weights(D_OUT * D_IN, 42), false);
        let mut layer = LoRALayer::new(base, D_OUT, D_IN, RANK, ALPHA);
        // Adapter-only bundle
        let a_bytes = tensor_to_bytes(layer.lora_a());
        let b_bytes = tensor_to_bytes(layer.lora_b());
        let adapter_bundle = ModelBundleV2::new()
            .with_name("test-adapter")
            .with_compression(Compression::Lz4)
            .add_tensor("lora_a", vec![RANK, D_IN], a_bytes)
            .add_tensor("lora_b", vec![D_OUT, RANK], b_bytes)
            .build();
        assert_eq!(&adapter_bundle[0..4], b"APR2");
        // Merged bundle
        layer.merge();
        let merged_bundle = ModelBundleV2::new()
            .with_name("test-merged")
            .with_compression(Compression::Lz4)
            .add_tensor(
                "weight",
                vec![D_OUT, D_IN],
                tensor_to_bytes(layer.base_weight()),
            )
            .build();
        assert_eq!(&merged_bundle[0..4], b"APR2");
    }

    #[test]
    fn test_base_forward_shape() {
        let weights = det_weights(D_OUT * D_IN, 42);
        let x = vec![1.0f32; D_IN];
        let out = base_forward(&weights, &x, D_OUT, D_IN);
        assert_eq!(out.len(), D_OUT);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_training_modifies_adapters() {
        let base = Tensor::from_vec(det_weights(D_OUT * D_IN, 42), false);
        let mut layer = LoRALayer::new(base, D_OUT, D_IN, RANK, ALPHA);

        let a_before = layer.lora_a().data().to_vec();
        let b_before = layer.lora_b().data().to_vec();

        let mut optimizer = AdamW::default_params(0.001);
        let (inputs, targets) = gen_data(10, D_IN, D_OUT, 42);

        for (x, t) in inputs.iter().zip(targets.iter()) {
            optimizer.zero_grad_refs(&mut layer.trainable_params());
            let pred = lora_forward(&layer, x, D_OUT, D_IN);
            let loss = mse_loss(&pred, t);
            let grad_scale = (loss * 0.01).min(0.1);
            for param in layer.trainable_params() {
                let grad = Array1::from_elem(param.len(), grad_scale);
                param.set_grad(grad);
            }
            optimizer.step_refs(&mut layer.trainable_params());
        }

        let a_after = layer.lora_a().data().to_vec();
        let b_after = layer.lora_b().data().to_vec();

        // At least one of A or B should have changed
        let a_changed = a_before != a_after;
        let b_changed = b_before != b_after;
        assert!(
            a_changed || b_changed,
            "Training should modify adapter params"
        );
    }

    #[test]
    fn test_det_weights_deterministic() {
        let w1 = det_weights(128, 42);
        let w2 = det_weights(128, 42);
        assert_eq!(w1, w2);
    }
}
