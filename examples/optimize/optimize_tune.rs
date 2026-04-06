//! # Recipe: Memory Planning for LoRA/QLoRA Fine-Tuning
//!
//! **Category**: optimize
//! **CLI Equivalent**: `apr tune`
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

// ── Domain Types ──

/// Specification for a model to fine-tune.
#[derive(Debug, Clone)]
struct ModelSpec {
    name: &'static str,
    param_count: u64,
    hidden_dim: u64,
}

/// Fine-tuning method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TuneMethod {
    Full,
    LoRA,
    QLoRA,
}

impl TuneMethod {
    const fn name(self) -> &'static str {
        match self {
            Self::Full => "Full",
            Self::LoRA => "LoRA",
            Self::QLoRA => "QLoRA",
        }
    }

    /// Bytes per parameter for the frozen base model.
    /// Full/LoRA store base in FP16 (2 bytes); QLoRA quantizes to 4-bit (0.5 bytes).
    const fn base_bytes_per_param(self) -> f64 {
        match self {
            Self::Full | Self::LoRA => 2.0,
            Self::QLoRA => 0.5,
        }
    }
}

/// Result of planning a fine-tuning configuration.
#[derive(Debug, Clone)]
struct TunePlan {
    method: TuneMethod,
    #[allow(dead_code)]
    rank: u64,
    #[allow(dead_code)]
    alpha: f64,
    trainable_params: u64,
    pct_trainable: f64,
    memory_gb: f64,
    speedup: f64,
}

// ── Constants ──

const BYTES_PER_GB: f64 = 1_073_741_824.0; // 2^30
const ADAM_STATES_FACTOR: u64 = 2; // momentum + variance
const BYTES_PER_FP32: u64 = 4;
const NUM_LORA_PROJECTIONS: u64 = 4; // Q, K, V, O

/// Standard model specifications: 1B, 7B, 13B.
fn build_model_specs() -> Vec<ModelSpec> {
    vec![
        ModelSpec {
            name: "SmallLM-1B",
            param_count: 1_000_000_000,
            hidden_dim: 2048,
        },
        ModelSpec {
            name: "MediumLM-7B",
            param_count: 7_000_000_000,
            hidden_dim: 4096,
        },
        ModelSpec {
            name: "LargeLM-13B",
            param_count: 13_000_000_000,
            hidden_dim: 5120,
        },
    ]
}

// ── Core Planning Logic ──

/// Compute LoRA trainable parameters for a given rank and hidden dimension.
///
/// Each LoRA-targeted projection (Q, K, V, O) adds two matrices:
///   A: (rank x hidden_dim) and B: (hidden_dim x rank)
/// Total per projection = 2 * rank * hidden_dim.
/// We assume one transformer block's worth of projections.
/// For a full model, multiply by number of layers (estimated from param_count / hidden_dim^2).
fn compute_lora_trainable_params(hidden_dim: u64, rank: u64, param_count: u64) -> u64 {
    let estimated_layers = estimate_layer_count(param_count, hidden_dim);
    let per_projection = 2 * rank * hidden_dim;
    NUM_LORA_PROJECTIONS * per_projection * estimated_layers
}

/// Estimate the number of transformer layers from total parameters and hidden dimension.
///
/// A transformer layer is roughly 12 * hidden_dim^2 parameters
/// (self-attention QKV + O + two FFN layers).
fn estimate_layer_count(param_count: u64, hidden_dim: u64) -> u64 {
    let params_per_layer = 12 * hidden_dim * hidden_dim;
    if params_per_layer == 0 {
        return 1;
    }
    (param_count / params_per_layer).max(1)
}

/// Estimate memory in GB for a given method, model, and rank.
///
/// Components:
/// - Base model weights (dtype depends on method)
/// - Trainable parameter storage (FP32)
/// - Optimizer states (Adam: 2 FP32 states per trainable param)
/// - Gradients (FP32 per trainable param)
fn estimate_memory_gb(spec: &ModelSpec, method: TuneMethod, rank: u64) -> f64 {
    let base_bytes = spec.param_count as f64 * method.base_bytes_per_param();

    let trainable = match method {
        TuneMethod::Full => spec.param_count,
        TuneMethod::LoRA | TuneMethod::QLoRA => {
            compute_lora_trainable_params(spec.hidden_dim, rank, spec.param_count)
        }
    };

    let trainable_bytes = trainable * BYTES_PER_FP32;
    let optimizer_bytes = trainable * BYTES_PER_FP32 * ADAM_STATES_FACTOR;
    let gradient_bytes = trainable * BYTES_PER_FP32;

    let total_bytes = base_bytes + (trainable_bytes + optimizer_bytes + gradient_bytes) as f64;
    total_bytes / BYTES_PER_GB
}

/// Estimate speedup vs full fine-tuning.
///
/// Full fine-tuning: 1.0x baseline.
/// LoRA: speedup proportional to parameter reduction (fewer backward-pass updates).
/// QLoRA: LoRA speedup minus dequantization overhead (~10%).
fn estimate_speedup(spec: &ModelSpec, method: TuneMethod, rank: u64) -> f64 {
    match method {
        TuneMethod::Full => 1.0,
        TuneMethod::LoRA => {
            let trainable = compute_lora_trainable_params(spec.hidden_dim, rank, spec.param_count);
            let ratio = trainable as f64 / spec.param_count as f64;
            // Speedup from fewer gradient updates; bounded by overhead
            (1.0 / ratio).clamp(1.0, 50.0)
        }
        TuneMethod::QLoRA => {
            let lora_speedup = estimate_speedup(spec, TuneMethod::LoRA, rank);
            // QLoRA has ~10% overhead from dequantization during forward pass
            lora_speedup * 0.9
        }
    }
}

/// Plan a fine-tuning configuration for a given method and rank.
fn plan_tune(spec: &ModelSpec, method: TuneMethod, rank: u64) -> TunePlan {
    let trainable = match method {
        TuneMethod::Full => spec.param_count,
        TuneMethod::LoRA | TuneMethod::QLoRA => {
            compute_lora_trainable_params(spec.hidden_dim, rank, spec.param_count)
        }
    };

    let pct = trainable as f64 / spec.param_count as f64 * 100.0;
    let memory = estimate_memory_gb(spec, method, rank);
    let speedup = estimate_speedup(spec, method, rank);

    TunePlan {
        method,
        rank,
        alpha: rank as f64, // standard: alpha = rank
        trainable_params: trainable,
        pct_trainable: pct,
        memory_gb: memory,
        speedup,
    }
}

/// Find the optimal LoRA rank that fits within a VRAM budget.
///
/// Tries ranks from highest to lowest. Returns the highest rank whose
/// estimated memory fits within the budget (with 10% safety margin).
fn optimal_rank_for_budget(
    spec: &ModelSpec,
    method: TuneMethod,
    vram_budget_gb: f64,
) -> Option<u64> {
    let safe_budget = vram_budget_gb * 0.9;
    let candidate_ranks: &[u64] = &[128, 64, 32, 16, 8, 4];

    for &rank in candidate_ranks {
        let mem = estimate_memory_gb(spec, method, rank);
        if mem <= safe_budget {
            return Some(rank);
        }
    }
    None
}

/// Determine the maximum model size (in billions) that fits a VRAM budget for each method.
fn max_model_for_budget(vram_budget_gb: f64) -> Vec<(TuneMethod, f64)> {
    let methods = [TuneMethod::Full, TuneMethod::LoRA, TuneMethod::QLoRA];
    let mut results = Vec::with_capacity(methods.len());

    for method in methods {
        // Binary search over model sizes (0.1B to 200B)
        let mut lo: f64 = 0.1;
        let mut hi: f64 = 200.0;
        let safe_budget = vram_budget_gb * 0.9;

        for _ in 0..50 {
            let mid = (lo + hi) / 2.0;
            let spec = ModelSpec {
                name: "probe",
                param_count: (mid * 1e9) as u64,
                hidden_dim: estimate_hidden_dim_for_params((mid * 1e9) as u64),
            };
            let rank = match method {
                TuneMethod::Full => 0,
                _ => 16,
            };
            let mem = estimate_memory_gb(&spec, method, rank);
            if mem <= safe_budget {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        results.push((method, lo));
    }

    results
}

/// Estimate a reasonable hidden_dim given a parameter count.
/// Roughly: hidden_dim ~ (param_count / 12 / num_layers)^0.5
/// We use a simplified heuristic.
fn estimate_hidden_dim_for_params(param_count: u64) -> u64 {
    // Empirical: hidden_dim scales roughly as param_count^(1/3) * constant
    let estimate = (param_count as f64).powf(1.0 / 3.0) * 2.0;
    (estimate as u64).max(256)
}

// ── Output Sections ──

fn section_model_specs(models: &[ModelSpec]) {
    println!("Model Specifications");
    println!("   ─────────────────────────────────────────");
    for m in models {
        println!(
            "   {:<15}: {:>5.1}B params, hidden_dim={}",
            m.name,
            m.param_count as f64 / 1e9,
            m.hidden_dim,
        );
    }
    println!();
}

fn section_tune_plans(models: &[ModelSpec]) {
    let rank = 16_u64;
    println!("Tune Plans (rank={}, alpha={})", rank, rank);
    println!("   ─────────────────────────────────────────");
    for m in models {
        println!("   {}:", m.name);
        for method in [TuneMethod::Full, TuneMethod::LoRA, TuneMethod::QLoRA] {
            let plan = plan_tune(m, method, rank);
            println!(
                "      {:<5}: trainable={:>12}, pct={:>6.3}%, mem={:>6.2} GB, speedup={:>5.1}x",
                plan.method.name(),
                format_param_count(plan.trainable_params),
                plan.pct_trainable,
                plan.memory_gb,
                plan.speedup,
            );
        }
        println!();
    }
}

fn section_comparison_table(models: &[ModelSpec]) {
    let rank = 16_u64;
    println!("Comparison Table (rank={})", rank);
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>15} {:>8} {:>14} {:>10} {:>8}",
        "Model", "Method", "Trainable", "Memory GB", "Speedup"
    );
    println!("   {}", "-".repeat(62));

    for m in models {
        for method in [TuneMethod::Full, TuneMethod::LoRA, TuneMethod::QLoRA] {
            let plan = plan_tune(m, method, rank);
            println!(
                "   {:>15} {:>8} {:>14} {:>10.2} {:>7.1}x",
                m.name,
                plan.method.name(),
                format_param_count(plan.trainable_params),
                plan.memory_gb,
                plan.speedup,
            );
        }
    }
    println!();
}

fn section_vram_budget_planning(models: &[ModelSpec]) {
    let vram_budget = 16.0_f64;
    println!("VRAM Budget Planning ({:.0} GB)", vram_budget);
    println!("   ─────────────────────────────────────────");

    for m in models {
        println!("   {}:", m.name);
        for method in [TuneMethod::Full, TuneMethod::LoRA, TuneMethod::QLoRA] {
            let rank_label = match method {
                TuneMethod::Full => {
                    let mem = estimate_memory_gb(m, method, 0);
                    if mem <= vram_budget * 0.9 {
                        format!("fits ({:.2} GB)", mem)
                    } else {
                        format!("exceeds budget ({:.2} GB)", mem)
                    }
                }
                _ => match optimal_rank_for_budget(m, method, vram_budget) {
                    Some(r) => {
                        let mem = estimate_memory_gb(m, method, r);
                        format!("rank={}, {:.2} GB", r, mem)
                    }
                    None => "does not fit".to_string(),
                },
            };
            println!("      {:<5}: {}", method.name(), rank_label);
        }
        println!();
    }
}

fn section_max_model_for_budget() {
    let vram_budget = 16.0_f64;
    println!(
        "Max Model Size for {:.0} GB VRAM (each method)",
        vram_budget
    );
    println!("   ─────────────────────────────────────────");

    let results = max_model_for_budget(vram_budget);
    for (method, max_b) in &results {
        println!("   {:<5}: up to {:.1}B parameters", method.name(), max_b);
    }
    println!();
}

fn section_record_metrics(ctx: &mut RecipeContext, models: &[ModelSpec]) {
    let rank = 16_u64;
    for m in models {
        for method in [TuneMethod::Full, TuneMethod::LoRA, TuneMethod::QLoRA] {
            let plan = plan_tune(m, method, rank);
            let key = format!("{}_{}_memory_gb", m.name, method.name().to_lowercase());
            ctx.record_float_metric(&key, plan.memory_gb);
        }
    }

    // Record LoRA vs Full savings for reference model (7B)
    let ref_model = &models[1];
    let full_mem = estimate_memory_gb(ref_model, TuneMethod::Full, 0);
    let lora_mem = estimate_memory_gb(ref_model, TuneMethod::LoRA, rank);
    let qlora_mem = estimate_memory_gb(ref_model, TuneMethod::QLoRA, rank);

    if full_mem > 0.0 {
        ctx.record_float_metric("lora_savings_pct", (1.0 - lora_mem / full_mem) * 100.0);
        ctx.record_float_metric("qlora_savings_pct", (1.0 - qlora_mem / full_mem) * 100.0);
    }
}

/// Format a parameter count for display (e.g., 1_234_567 -> "1.23M").
fn format_param_count(count: u64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.2}B", count as f64 / 1e9)
    } else if count >= 1_000_000 {
        format!("{:.2}M", count as f64 / 1e6)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1e3)
    } else {
        format!("{}", count)
    }
}

// ── Main ──

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
