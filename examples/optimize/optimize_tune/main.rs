#![allow(unused_imports)]
//! # Recipe: Memory Planning for LoRA/QLoRA Fine-Tuning
//!
//! **Category**: optimize
//! **CLI Equivalent**: `apr tune`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Plans LoRA/QLoRA fine-tuning configurations by computing optimal rank given
//! a VRAM budget. Compares Full, LoRA, and QLoRA methods across model sizes
//! (1B, 7B, 13B), showing trainable parameters, memory estimates, and speedup.
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
//! Understand how to plan LoRA/QLoRA fine-tuning by computing trainable parameter
//! counts, memory requirements, and speedup estimates for different model sizes
//! and tuning methods.
//!
//! ## Run Command
//! ```bash
//! cargo run --example optimize_tune
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr inspect model.apr          # APR native format
//! apr inspect model.gguf         # GGUF (llama.cpp compatible)
//! apr inspect model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use apr_cookbook::prelude::*;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("optimize_tune")?;

    println!("=== Memory Planning for LoRA/QLoRA Fine-Tuning ===");
    println!("Mirrors: apr tune");
    println!();

    let models = build_model_specs();

    section_model_specs(&models);
    section_tune_plans(&models);
    section_comparison_table(&models);
    section_vram_budget_planning(&models);
    section_max_model_for_budget();
    section_record_metrics(&mut ctx, &models);

    println!();
    ctx.report()?;
    Ok(())
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    fn test_spec_1b() -> ModelSpec {
        ModelSpec {
            name: "Test-1B",
            param_count: 1_000_000_000,
            hidden_dim: 2048,
        }
    }

    fn test_spec_7b() -> ModelSpec {
        ModelSpec {
            name: "Test-7B",
            param_count: 7_000_000_000,
            hidden_dim: 4096,
        }
    }

    #[test]
    fn test_lora_trainable_params_positive() {
        let spec = test_spec_1b();
        let trainable = compute_lora_trainable_params(spec.hidden_dim, 16, spec.param_count);
        assert!(trainable > 0, "LoRA trainable params must be positive");
    }

    #[test]
    fn test_lora_trainable_less_than_full() {
        let spec = test_spec_7b();
        let trainable = compute_lora_trainable_params(spec.hidden_dim, 16, spec.param_count);
        assert!(
            trainable < spec.param_count,
            "LoRA trainable ({}) must be less than full ({})",
            trainable,
            spec.param_count
        );
    }

    #[test]
    fn test_memory_ordering_full_gt_lora_gt_qlora() {
        let spec = test_spec_7b();
        let rank = 16;
        let full = estimate_memory_gb(&spec, TuneMethod::Full, rank);
        let lora = estimate_memory_gb(&spec, TuneMethod::LoRA, rank);
        let qlora = estimate_memory_gb(&spec, TuneMethod::QLoRA, rank);
        assert!(full > lora, "Full ({:.2}) > LoRA ({:.2})", full, lora);
        assert!(lora > qlora, "LoRA ({:.2}) > QLoRA ({:.2})", lora, qlora);
    }

    #[test]
    fn test_speedup_full_is_baseline() {
        let spec = test_spec_1b();
        let speedup = estimate_speedup(&spec, TuneMethod::Full, 16);
        assert!(
            (speedup - 1.0).abs() < f64::EPSILON,
            "Full speedup must be 1.0x"
        );
    }

    #[test]
    fn test_speedup_lora_gt_one() {
        let spec = test_spec_7b();
        let speedup = estimate_speedup(&spec, TuneMethod::LoRA, 16);
        assert!(
            speedup > 1.0,
            "LoRA speedup ({:.1}x) must exceed 1.0x",
            speedup
        );
    }

    #[test]
    fn test_speedup_qlora_less_than_lora() {
        let spec = test_spec_7b();
        let rank = 16;
        let lora_sp = estimate_speedup(&spec, TuneMethod::LoRA, rank);
        let qlora_sp = estimate_speedup(&spec, TuneMethod::QLoRA, rank);
        assert!(
            qlora_sp < lora_sp,
            "QLoRA speedup ({:.1}x) must be less than LoRA ({:.1}x)",
            qlora_sp,
            lora_sp
        );
    }

    #[test]
    fn test_plan_tune_pct_trainable_full() {
        let spec = test_spec_1b();
        let plan = plan_tune(&spec, TuneMethod::Full, 0);
        assert!(
            (plan.pct_trainable - 100.0).abs() < f64::EPSILON,
            "Full pct_trainable must be 100%"
        );
        assert_eq!(plan.trainable_params, spec.param_count);
    }

    #[test]
    fn test_optimal_rank_for_budget_returns_some() {
        let spec = test_spec_1b();
        let rank = optimal_rank_for_budget(&spec, TuneMethod::LoRA, 16.0);
        assert!(rank.is_some(), "Should find a rank for 1B model in 16 GB");
        let r = rank.expect("verified Some above");
        assert!(r >= 4, "Optimal rank should be at least 4");
    }

    #[test]
    fn test_max_model_for_budget_ordering() {
        let results = max_model_for_budget(16.0);
        assert_eq!(results.len(), 3);
        let full_max = results[0].1;
        let lora_max = results[1].1;
        let qlora_max = results[2].1;
        assert!(
            qlora_max >= lora_max,
            "QLoRA ({:.1}B) should fit >= LoRA ({:.1}B)",
            qlora_max,
            lora_max
        );
        assert!(
            lora_max >= full_max,
            "LoRA ({:.1}B) should fit >= Full ({:.1}B)",
            lora_max,
            full_max
        );
    }

    #[test]
    fn test_format_param_count() {
        assert_eq!(format_param_count(500), "500");
        assert_eq!(format_param_count(1_500), "1.5K");
        assert_eq!(format_param_count(1_234_567), "1.23M");
        assert_eq!(format_param_count(7_000_000_000), "7.00B");
    }
}
