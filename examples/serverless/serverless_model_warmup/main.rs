#![allow(unused_imports)]
//! # Recipe: Model Warmup Strategies
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! **Category**: Serverless/Lambda
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Minimize cold start latency with model warmup strategies for serverless deployments.
//!
//! ## Run Command
//! ```bash
//! cargo run --example serverless_model_warmup
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Schleier-Smith, J. et al. (2021). *What Serverless Computing Is and Should Become*. CACM. DOI: 10.1145/3406011

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("serverless_model_warmup")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Model warmup strategies for cold start minimization");
    println!();

    // ----------------------------------------------------------------
    // Section 1: Measure baseline cold start latency
    // ----------------------------------------------------------------
    println!("--- Section 1: Baseline Cold Start Measurement ---");
    println!();

    let models = sample_models();

    let baseline = measure_cold_start_baseline(&models);
    println!("  First request (cold):  {} us", baseline.first_request_us);
    println!("  Warm request (steady): {} us", baseline.warm_request_us);
    println!("  Overhead ratio:        {:.2}x", baseline.overhead_ratio);
    println!();

    ctx.record_metric("cold_start_us", baseline.first_request_us as i64);
    ctx.record_metric("warm_request_us", baseline.warm_request_us as i64);
    ctx.record_float_metric("overhead_ratio", baseline.overhead_ratio);

    // ----------------------------------------------------------------
    // Section 2: Eager vs lazy loading comparison
    // ----------------------------------------------------------------
    println!("--- Section 2: Eager vs Lazy Loading ---");
    println!();

    let eager_plan = build_warmup_plan(&models, &WarmupStrategy::Eager, 512);
    let lazy_plan = build_warmup_plan(&models, &WarmupStrategy::Lazy, 512);

    println!("  Eager loading:");
    println!(
        "    Warmup time:    {} us",
        eager_plan.estimated_warmup_time_us
    );
    println!("    Memory needed:  {} MB", eager_plan.memory_required_mb);
    println!("    Models loaded:  {}", eager_plan.load_order.len());
    println!();
    println!("  Lazy loading:");
    println!(
        "    Warmup time:    {} us",
        lazy_plan.estimated_warmup_time_us
    );
    println!("    Memory needed:  {} MB", lazy_plan.memory_required_mb);
    println!("    Models loaded:  {}", lazy_plan.load_order.len());
    println!();

    // ----------------------------------------------------------------
    // Section 3: Priority-based model preloading
    // ----------------------------------------------------------------
    println!("--- Section 3: Priority-Based Model Preloading ---");
    println!();

    let priority_plan = build_warmup_plan(&models, &WarmupStrategy::PriorityBased, 512);
    println!("  Load order (highest priority first):");
    for (i, name) in priority_plan.load_order.iter().enumerate() {
        let model = models.iter().find(|m| m.name == *name);
        if let Some(m) = model {
            println!(
                "    {}. {} (priority={}, size={}MB, load={}us)",
                i + 1,
                m.name,
                m.priority,
                m.size_mb,
                m.load_time_us
            );
        }
    }
    println!(
        "  Total warmup time: {} us",
        priority_plan.estimated_warmup_time_us
    );
    println!();

    // ----------------------------------------------------------------
    // Section 4: Memory-constrained warmup planning
    // ----------------------------------------------------------------
    println!("--- Section 4: Memory-Constrained Warmup ---");
    println!();

    let memory_limits = [256_u32, 512, 1024];
    for &limit in &memory_limits {
        let state = plan_memory_aware_warmup(&models, limit);
        println!(
            "  Container {}MB: loaded={}/{}, used={}MB, headroom={}MB",
            limit,
            state.models_loaded,
            models.len(),
            state.memory_used_mb,
            limit.saturating_sub(state.memory_used_mb)
        );
    }
    println!();

    // ----------------------------------------------------------------
    // Section 5: Keep-alive strategy simulation
    // ----------------------------------------------------------------
    println!("--- Section 5: Keep-Alive Strategy Simulation ---");
    println!();

    let keep_alive_results = simulate_keep_alive(&models, 300, 60);
    println!(
        "  Idle timeout:         {} s",
        keep_alive_results.idle_timeout_s
    );
    println!(
        "  Ping interval:        {} s",
        keep_alive_results.ping_interval_s
    );
    println!("  Total pings sent:     {}", keep_alive_results.total_pings);
    println!(
        "  Cold starts avoided:  {}",
        keep_alive_results.cold_starts_avoided
    );
    println!(
        "  Estimated cost saved: {:.2}%",
        keep_alive_results.cost_saving_pct
    );
    println!();

    ctx.record_metric("pings_sent", i64::from(keep_alive_results.total_pings));
    ctx.record_metric(
        "cold_starts_avoided",
        i64::from(keep_alive_results.cold_starts_avoided),
    );

    // ----------------------------------------------------------------
    // Section 6: Warmup strategy comparison summary
    // ----------------------------------------------------------------
    println!("--- Section 6: Strategy Comparison Summary ---");
    println!();

    let strategies = [
        WarmupStrategy::Eager,
        WarmupStrategy::Lazy,
        WarmupStrategy::Tiered,
        WarmupStrategy::PriorityBased,
    ];

    println!(
        "  {:<16} {:>12} {:>10} {:>8}",
        "Strategy", "Warmup (us)", "Memory MB", "Models"
    );
    println!("  {:-<50}", "");

    for strategy in &strategies {
        let plan = build_warmup_plan(&models, strategy, 512);
        println!(
            "  {:<16} {:>12} {:>10} {:>8}",
            strategy.label(),
            plan.estimated_warmup_time_us,
            plan.memory_required_mb,
            plan.load_order.len()
        );
    }
    println!();

    let best_plan = build_warmup_plan(&models, &WarmupStrategy::PriorityBased, 512);
    let worst_plan = build_warmup_plan(&models, &WarmupStrategy::Eager, 512);
    let memory_savings =
        f64::from(worst_plan.memory_required_mb) - f64::from(best_plan.memory_required_mb);
    let memory_savings_pct = if worst_plan.memory_required_mb > 0 {
        (memory_savings / f64::from(worst_plan.memory_required_mb)) * 100.0
    } else {
        0.0
    };

    ctx.record_float_metric("memory_savings_pct", memory_savings_pct);

    println!(
        "  Best strategy (PriorityBased) saves {:.1}% memory vs Eager",
        memory_savings_pct
    );

    // Save report
    let report = WarmupReport {
        baseline_metrics: baseline,
        strategies: strategies
            .iter()
            .map(|s| {
                let plan = build_warmup_plan(&models, s, 512);
                StrategyResult {
                    strategy: s.label().to_string(),
                    warmup_time_us: plan.estimated_warmup_time_us,
                    memory_required_mb: plan.memory_required_mb,
                    models_loaded: plan.load_order.len() as u32,
                }
            })
            .collect(),
        keep_alive: keep_alive_results,
    };

    let report_path = ctx.path("warmup_report.json");
    save_report(&report_path, &report)?;
    println!();
    println!("Report saved to: {:?}", report_path);

    Ok(())
}

mod helpers;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod proptests;
