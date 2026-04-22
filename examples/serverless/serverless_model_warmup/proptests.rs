//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
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
