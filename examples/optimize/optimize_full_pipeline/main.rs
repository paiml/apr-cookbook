#![allow(unused_imports)]
//! # Recipe: Full Optimization Pipeline
//!
//! **Category**: optimize
//! **CLI Equivalent**: `apr finetune && apr prune && apr distill && apr merge && apr quantize`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Demonstrates chaining: LoRA fine-tuning → magnitude pruning → knowledge
//! distillation → TIES model merging → 4-bit quantization.
//!
//! ```bash
//! cargo run --example optimize_full_pipeline
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

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
