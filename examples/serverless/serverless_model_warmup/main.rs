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
mod tests {
    use super::*;

    #[test]
    fn test_sample_models_non_empty() {
        let models = sample_models();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_sample_models_have_valid_fields() {
        let models = sample_models();
        for m in &models {
            assert!(!m.name.is_empty());
            assert!(m.size_mb > 0);
            assert!(m.load_time_us > 0);
            assert!(m.inference_time_us > 0);
            assert!(m.priority > 0);
        }
    }

    #[test]
    fn test_cold_start_baseline_overhead() {
        let models = sample_models();
        let metrics = measure_cold_start_baseline(&models);
        assert!(metrics.first_request_us > metrics.warm_request_us);
        assert!(metrics.overhead_ratio > 1.0);
    }

    #[test]
    fn test_cold_start_baseline_deterministic() {
        let models = sample_models();
        let m1 = measure_cold_start_baseline(&models);
        let m2 = measure_cold_start_baseline(&models);
        assert_eq!(m1.first_request_us, m2.first_request_us);
        assert_eq!(m1.warm_request_us, m2.warm_request_us);
    }

    #[test]
    fn test_eager_plan_loads_all_that_fit() {
        let models = sample_models();
        let plan = build_warmup_plan(&models, &WarmupStrategy::Eager, 1024);
        assert!(!plan.load_order.is_empty());
        assert!(plan.memory_required_mb <= 1024);
        assert!(plan.estimated_warmup_time_us > 0);
    }

    #[test]
    fn test_eager_plan_respects_memory_limit() {
        let models = sample_models();
        let plan = build_warmup_plan(&models, &WarmupStrategy::Eager, 100);
        assert!(plan.memory_required_mb <= 100);
    }

    #[test]
    fn test_lazy_plan_loads_nothing() {
        let models = sample_models();
        let plan = build_warmup_plan(&models, &WarmupStrategy::Lazy, 512);
        assert!(plan.load_order.is_empty());
        assert_eq!(plan.estimated_warmup_time_us, 0);
        assert_eq!(plan.memory_required_mb, 0);
        let _ = models; // suppress unused warning in older compilers
    }

    #[test]
    fn test_tiered_plan_loads_smallest_first() {
        let models = sample_models();
        let plan = build_warmup_plan(&models, &WarmupStrategy::Tiered, 512);
        // The first model in load_order should be one of the smallest
        if let Some(first_name) = plan.load_order.first() {
            let first_model = models.iter().find(|m| m.name == *first_name);
            assert!(first_model.is_some());
            let first_size = first_model.map_or(0, |m| m.size_mb);
            let min_size = models.iter().map(|m| m.size_mb).min().unwrap_or(0);
            assert_eq!(first_size, min_size);
        }
    }

    #[test]
    fn test_priority_plan_loads_highest_priority_first() {
        let models = sample_models();
        let plan = build_warmup_plan(&models, &WarmupStrategy::PriorityBased, 512);
        if let Some(first_name) = plan.load_order.first() {
            let first_model = models.iter().find(|m| m.name == *first_name);
            assert!(first_model.is_some());
            let first_priority = first_model.map_or(u32::MAX, |m| m.priority);
            let min_priority = models.iter().map(|m| m.priority).min().unwrap_or(0);
            assert_eq!(first_priority, min_priority);
        }
    }

    #[test]
    fn test_memory_aware_warmup_respects_limit() {
        let models = sample_models();
        let state = plan_memory_aware_warmup(&models, 256);
        assert!(state.memory_used_mb <= 256);
        assert_eq!(state.memory_limit_mb, 256);
    }

    #[test]
    fn test_memory_aware_warmup_loads_more_with_higher_limit() {
        let models = sample_models();
        let small = plan_memory_aware_warmup(&models, 128);
        let large = plan_memory_aware_warmup(&models, 1024);
        assert!(large.models_loaded >= small.models_loaded);
    }

    #[test]
    fn test_keep_alive_simulation_deterministic() {
        let models = sample_models();
        let r1 = simulate_keep_alive(&models, 300, 60);
        let r2 = simulate_keep_alive(&models, 300, 60);
        assert_eq!(r1.total_pings, r2.total_pings);
        assert_eq!(r1.cold_starts_avoided, r2.cold_starts_avoided);
    }

    #[test]
    fn test_keep_alive_pings_count() {
        let models = sample_models();
        let result = simulate_keep_alive(&models, 300, 60);
        assert_eq!(result.total_pings, 3600 / 60);
    }

    #[test]
    fn test_keep_alive_cost_saving_bounded() {
        let models = sample_models();
        let result = simulate_keep_alive(&models, 300, 60);
        assert!(result.cost_saving_pct >= 0.0);
        assert!(result.cost_saving_pct <= 100.0);
    }

    #[test]
    fn test_strategy_label() {
        assert_eq!(WarmupStrategy::Eager.label(), "Eager");
        assert_eq!(WarmupStrategy::Lazy.label(), "Lazy");
        assert_eq!(WarmupStrategy::Tiered.label(), "Tiered");
        assert_eq!(WarmupStrategy::PriorityBased.label(), "PriorityBased");
    }

    #[test]
    fn test_deterministic_hash_consistency() {
        let h1 = deterministic_hash("test-seed");
        let h2 = deterministic_hash("test-seed");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_deterministic_hash_variation() {
        let h1 = deterministic_hash("seed-a");
        let h2 = deterministic_hash("seed-b");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_save_report() {
        let recipe_ctx = RecipeContext::new("test_warmup_report").unwrap();
        let path = recipe_ctx.path("report.json");

        let report = WarmupReport {
            baseline_metrics: ColdStartMetrics {
                first_request_us: 100,
                warm_request_us: 10,
                overhead_ratio: 10.0,
            },
            strategies: vec![],
            keep_alive: KeepAliveResult {
                idle_timeout_s: 300,
                ping_interval_s: 60,
                total_pings: 60,
                cold_starts_avoided: 5,
                cost_saving_pct: 50.0,
            },
        };

        save_report(&path, &report).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_container_state_fields() {
        let models = sample_models();
        let state = plan_memory_aware_warmup(&models, 512);
        assert!(state.models_loaded > 0);
        assert!(state.memory_used_mb > 0);
        assert_eq!(state.memory_limit_mb, 512);
        assert_eq!(state.last_request_time, 0);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_eager_never_exceeds_memory(limit in 50u32..2000) {
            let models = sample_models();
            let plan = build_warmup_plan(&models, &WarmupStrategy::Eager, limit);
            prop_assert!(plan.memory_required_mb <= limit);
        }

        #[test]
        fn prop_tiered_never_exceeds_memory(limit in 50u32..2000) {
            let models = sample_models();
            let plan = build_warmup_plan(&models, &WarmupStrategy::Tiered, limit);
            prop_assert!(plan.memory_required_mb <= limit);
        }

        #[test]
        fn prop_priority_never_exceeds_memory(limit in 50u32..2000) {
            let models = sample_models();
            let plan = build_warmup_plan(&models, &WarmupStrategy::PriorityBased, limit);
            prop_assert!(plan.memory_required_mb <= limit);
        }

        #[test]
        fn prop_lazy_always_zero(limit in 50u32..2000) {
            let models = sample_models();
            let plan = build_warmup_plan(&models, &WarmupStrategy::Lazy, limit);
            let _ = models;
            prop_assert_eq!(plan.estimated_warmup_time_us, 0);
            prop_assert_eq!(plan.memory_required_mb, 0);
            prop_assert!(plan.load_order.is_empty());
        }

        #[test]
        fn prop_overhead_ratio_at_least_one(n in 1usize..20) {
            let models: Vec<ServerlessModel> = (0..n).map(|i| ServerlessModel {
                name: format!("model-{}", i),
                size_mb: 10 + (i as u32 * 5),
                load_time_us: 1000 + (i as u64 * 500),
                inference_time_us: 100 + (i as u64 * 50),
                priority: (i as u32) + 1,
            }).collect();

            let metrics = measure_cold_start_baseline(&models);
            prop_assert!(metrics.overhead_ratio >= 1.0);
        }

        #[test]
        fn prop_memory_aware_respects_limit(limit in 100u32..2000) {
            let models = sample_models();
            let state = plan_memory_aware_warmup(&models, limit);
            prop_assert!(state.memory_used_mb <= limit);
        }

        #[test]
        fn prop_keep_alive_pings_deterministic(
            timeout in 60u32..600,
            interval in 10u32..120
        ) {
            let models = sample_models();
            let r1 = simulate_keep_alive(&models, timeout, interval);
            let r2 = simulate_keep_alive(&models, timeout, interval);
            prop_assert_eq!(r1.total_pings, r2.total_pings);
            prop_assert_eq!(r1.cold_starts_avoided, r2.cold_starts_avoided);
        }
    }
}
