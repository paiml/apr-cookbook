#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
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
