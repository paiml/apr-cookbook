#![allow(unused_imports)]
//! # Recipe: VRAM Planning for Fine-Tuning
//!
//! **Category**: optimize
//! **CLI Equivalent**: `apr finetune --plan`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Estimates VRAM requirements for different fine-tuning methods (full,
//! LoRA, QLoRA) and recommends optimal configurations given GPU memory
//! constraints.
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Clippy clean
//! 6. [x] No `unwrap()` in logic
//!
//! ## Learning Objective
//! Understand how to estimate GPU memory requirements for fine-tuning and
//! select the right method (full/LoRA/QLoRA) based on available hardware.
//!
//! ## Run Command
//! ```bash
//! cargo run --example finetune_plan_vram
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

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("finetune_plan_vram")?;

    println!("=== VRAM Planning for Fine-Tuning ===");
    println!("Mirrors: apr finetune --plan");
    println!();

    let models = build_model_specs();
    section_model_specs(&models);
    section_vram_breakdown(&models[1]); // 1B reference model
    section_method_comparison(&models);
    section_gpu_recommendations(&models);
    section_record_metrics(&mut ctx, &models[1]);

    println!();
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_model() -> ModelConfig {
        ModelConfig {
            name: "TestLM".to_string(),
            params: 100_000_000,
            hidden_dim: 512,
            num_layers: 12,
            dtype_bytes: 4,
        }
    }

    #[test]
    fn test_estimate_non_negative() {
        let model = test_model();
        for method in [
            FinetuneMethod::Full,
            FinetuneMethod::LoRA,
            FinetuneMethod::QLoRA,
        ] {
            let est = estimate_vram(&model, method, 8, 4, 512);
            assert!(est.base_model > 0, "{method:?} base_model");
            assert!(est.optimizer_states > 0, "{method:?} optimizer");
            assert!(est.gradients > 0, "{method:?} gradients");
            assert!(est.activations > 0, "{method:?} activations");
            assert!(est.total > 0, "{method:?} total");
        }
    }

    #[test]
    fn test_vram_ordering_full_gt_lora_gt_qlora() {
        let model = test_model();
        let full = estimate_vram(&model, FinetuneMethod::Full, 0, 4, 512);
        let lora = estimate_vram(&model, FinetuneMethod::LoRA, 16, 4, 512);
        let qlora = estimate_vram(&model, FinetuneMethod::QLoRA, 16, 4, 512);
        assert!(full.total > lora.total, "Full > LoRA");
        assert!(lora.total > qlora.total, "LoRA > QLoRA");
    }

    #[test]
    fn test_optimal_config_valid() {
        let model = test_model();
        let opt = find_optimal_config(&model, 24.0);
        assert!(opt.batch_size >= 1);
        assert!(opt.estimated_vram_gb > 0.0);
        assert!(opt.estimated_vram_gb <= 24.0, "must fit in GPU");
    }

    #[test]
    fn test_scaling_with_params() {
        let small = test_model(); // 100M params
        let large = ModelConfig {
            name: "Large".to_string(),
            params: 1_000_000_000,
            hidden_dim: 2048,
            num_layers: 24,
            dtype_bytes: 4,
        };
        let small_est = estimate_vram(&small, FinetuneMethod::Full, 0, 4, 512);
        let large_est = estimate_vram(&large, FinetuneMethod::Full, 0, 4, 512);
        assert!(large_est.total > small_est.total);
    }

    #[test]
    fn test_base_model_sizes_and_overhead() {
        let model = test_model();
        // Full: base = params * dtype_bytes, no LoRA overhead
        let full = estimate_vram(&model, FinetuneMethod::Full, 0, 4, 512);
        assert_eq!(full.base_model, model.params * model.dtype_bytes);
        assert_eq!(full.lora_overhead, 0);
        // QLoRA: base = params/2 (4-bit), has LoRA overhead
        let qlora = estimate_vram(&model, FinetuneMethod::QLoRA, 8, 4, 512);
        assert_eq!(qlora.base_model, model.params / 2);
        assert!(qlora.lora_overhead > 0);
    }

    #[test]
    fn test_batch_size_affects_activations() {
        let model = test_model();
        let bs1 = estimate_vram(&model, FinetuneMethod::LoRA, 8, 1, 512);
        let bs4 = estimate_vram(&model, FinetuneMethod::LoRA, 8, 4, 512);
        assert_eq!(bs4.activations, bs1.activations * 4);
    }

    #[test]
    fn test_total_equals_sum() {
        let model = test_model();
        let est = estimate_vram(&model, FinetuneMethod::LoRA, 8, 4, 512);
        let expected = est.base_model
            + est.optimizer_states
            + est.gradients
            + est.activations
            + est.lora_overhead;
        assert_eq!(est.total, expected);
    }

    #[test]
    fn test_large_model_needs_qlora() {
        let large = ModelConfig {
            name: "70B".to_string(),
            params: 70_000_000_000,
            hidden_dim: 8192,
            num_layers: 80,
            dtype_bytes: 2,
        };
        let opt = find_optimal_config(&large, 24.0);
        assert_eq!(opt.method, FinetuneMethod::QLoRA);
    }
}
