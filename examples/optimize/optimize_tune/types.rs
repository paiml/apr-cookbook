#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;

// ── Domain Types ──

/// Specification for a model to fine-tune.
#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub name: &'static str,
    pub param_count: u64,
    pub hidden_dim: u64,
}

/// Fine-tuning method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuneMethod {
    Full,
    LoRA,
    QLoRA,
}

impl TuneMethod {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Full => "Full",
            Self::LoRA => "LoRA",
            Self::QLoRA => "QLoRA",
        }
    }

    // Bytes per parameter for the frozen base model.
    /// Full/LoRA store base in FP16 (2 bytes); QLoRA quantizes to 4-bit (0.5 bytes).
    pub const fn base_bytes_per_param(self) -> f64 {
        match self {
            Self::Full | Self::LoRA => 2.0,
            Self::QLoRA => 0.5,
        }
    }
}

/// Result of planning a fine-tuning configuration.
#[derive(Debug, Clone)]
pub struct TunePlan {
    pub method: TuneMethod,
    #[allow(dead_code)]
    pub rank: u64,
    #[allow(dead_code)]
    pub alpha: f64,
    pub trainable_params: u64,
    pub pct_trainable: f64,
    pub memory_gb: f64,
    pub speedup: f64,
}

// ── Constants ──

pub const BYTES_PER_GB: f64 = 1_073_741_824.0; // 2^30
pub const ADAM_STATES_FACTOR: u64 = 2; // momentum + variance
pub const BYTES_PER_FP32: u64 = 4;
pub const NUM_LORA_PROJECTIONS: u64 = 4; // Q, K, V, O

/// Standard model specifications: 1B, 7B, 13B.
pub fn build_model_specs() -> Vec<ModelSpec> {
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

// Compute LoRA trainable parameters for a given rank and hidden dimension.
//
// Each LoRA-targeted projection (Q, K, V, O) adds two matrices:
//   A: (rank x hidden_dim) and B: (hidden_dim x rank)
// Total per projection = 2 * rank * hidden_dim.
// We assume one transformer block's worth of projections.
/// For a full model, multiply by number of layers (estimated from param_count / hidden_dim^2).
pub fn compute_lora_trainable_params(hidden_dim: u64, rank: u64, param_count: u64) -> u64 {
    let estimated_layers = estimate_layer_count(param_count, hidden_dim);
    let per_projection = 2 * rank * hidden_dim;
    NUM_LORA_PROJECTIONS * per_projection * estimated_layers
}

// Estimate the number of transformer layers from total parameters and hidden dimension.
//
// A transformer layer is roughly 12 * hidden_dim^2 parameters
/// (self-attention QKV + O + two FFN layers).
pub fn estimate_layer_count(param_count: u64, hidden_dim: u64) -> u64 {
    let params_per_layer = 12 * hidden_dim * hidden_dim;
    if params_per_layer == 0 {
        return 1;
    }
    (param_count / params_per_layer).max(1)
}

// Estimate memory in GB for a given method, model, and rank.
//
// Components:
// - Base model weights (dtype depends on method)
// - Trainable parameter storage (FP32)
// - Optimizer states (Adam: 2 FP32 states per trainable param)
/// - Gradients (FP32 per trainable param)
pub fn estimate_memory_gb(spec: &ModelSpec, method: TuneMethod, rank: u64) -> f64 {
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

// Estimate speedup vs full fine-tuning.
//
// Full fine-tuning: 1.0x baseline.
// LoRA: speedup proportional to parameter reduction (fewer backward-pass updates).
/// QLoRA: LoRA speedup minus dequantization overhead (~10%).
pub fn estimate_speedup(spec: &ModelSpec, method: TuneMethod, rank: u64) -> f64 {
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
pub fn plan_tune(spec: &ModelSpec, method: TuneMethod, rank: u64) -> TunePlan {
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

// Find the optimal LoRA rank that fits within a VRAM budget.
//
// Tries ranks from highest to lowest. Returns the highest rank whose
/// estimated memory fits within the budget (with 10% safety margin).
pub fn optimal_rank_for_budget(
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
pub fn max_model_for_budget(vram_budget_gb: f64) -> Vec<(TuneMethod, f64)> {
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

// Estimate a reasonable hidden_dim given a parameter count.
// Roughly: hidden_dim ~ (param_count / 12 / num_layers)^0.5
/// We use a simplified heuristic.
pub fn estimate_hidden_dim_for_params(param_count: u64) -> u64 {
    // Empirical: hidden_dim scales roughly as param_count^(1/3) * constant
    let estimate = (param_count as f64).powf(1.0 / 3.0) * 2.0;
    (estimate as u64).max(256)
}

// ── Output Sections ──

pub fn section_model_specs(models: &[ModelSpec]) {
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

pub fn section_tune_plans(models: &[ModelSpec]) {
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

pub fn section_comparison_table(models: &[ModelSpec]) {
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

pub fn section_vram_budget_planning(models: &[ModelSpec]) {
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

pub fn section_max_model_for_budget() {
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

pub fn section_record_metrics(ctx: &mut RecipeContext, models: &[ModelSpec]) {
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
pub fn format_param_count(count: u64) -> String {
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
