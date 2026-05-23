#![allow(unused_imports)]
//! Recipe: Hierarchical Cache Performance Benchmark
//! Category: Advanced | Isolation: Full | Idempotent: Yes
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::env;
use std::time::Instant;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    println!(" Demo B: Hierarchical Cache Performance Benchmark");
    let args: Vec<String> = env::args().collect();
    let scenario_filter = args
        .iter()
        .position(|a| a == "--scenario")
        .and_then(|i| args.get(i + 1))
        .map(String::as_str);
    let ctx = RecipeContext::new("hierarchical_cache_benchmark")?;
    let seed = hash_name_to_seed(ctx.name());
    let (num_keys, num_accesses, l1_cap, l2_cap, l3_cap) = (1000, 10_000, 50, 150, 300);
    println!("\n Configuration: Keys={num_keys} Accesses={num_accesses} L1={l1_cap} L2={l2_cap} L3={l3_cap}");
    let patterns: Vec<AccessPattern> = if let Some(filter) = scenario_filter {
        AccessPattern::ALL
            .into_iter()
            .filter(|p| p.name().to_lowercase().contains(&filter.to_lowercase()))
            .collect()
    } else {
        AccessPattern::ALL.to_vec()
    };
    let mut results = Vec::new();
    for pattern in &patterns {
        for policy in &[EvictionPolicy::LRU, EvictionPolicy::LFU] {
            println!(" Running: {} with {}...", pattern.name(), policy.name());
            results.push(run_benchmark(
                *pattern,
                *policy,
                num_keys,
                num_accesses,
                l1_cap,
                l2_cap,
                l3_cap,
                seed,
            ));
        }
    }
    print_results(&results);
    if let Some(z) = results
        .iter()
        .find(|r| r.pattern == "Zipfian" && r.policy == "LRU")
    {
        println!(
            "\n Zipfian (LRU): {:.1}% hit rate",
            z.statistics.overall_hit_rate * 100.0
        );
    }
    if let Some(s) = results.iter().find(|r| r.pattern == "Sequential Scan") {
        println!(
            " Sequential Scan: {:.1}% hit rate",
            s.statistics.overall_hit_rate * 100.0
        );
    }
    println!("\n Benchmark complete!");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_tier_and_policy_basics() {
        assert_eq!(CacheTier::L1Hot.simulated_latency_ns(), 100);
        assert_eq!(CacheTier::L2Warm.simulated_latency_ns(), 1_000);
        assert_eq!(CacheTier::L3Cold.simulated_latency_ns(), 10_000_000);
        assert_eq!(EvictionPolicy::LRU.name(), "LRU");
        assert_eq!(EvictionPolicy::LFU.name(), "LFU");
        assert_eq!(EvictionPolicy::ARC.name(), "ARC");
        assert_eq!(format_latency(500), "500 ns");
        assert_eq!(format_latency(5000), "5.0 us");
        assert_eq!(format_latency(5_000_000), "5.0 ms");
    }

    #[test]
    fn test_cache_hit_l1() {
        let mut cache = HierarchicalCache::new(10, 20, 30, EvictionPolicy::LRU);
        assert_eq!(cache.access(42), CacheTier::Miss);
        assert_eq!(cache.access(42), CacheTier::L1Hot);
    }

    #[test]
    fn test_cache_promotion() {
        let mut cache = HierarchicalCache::new(2, 5, 10, EvictionPolicy::LRU);
        cache.access(0);
        cache.access(1);
        cache.access(2);
        cache.access(3);
        let tier = cache.access(0);
        assert!(tier == CacheTier::L2Warm || tier == CacheTier::L3Cold || tier == CacheTier::Miss);
    }

    #[test]
    fn test_lru_eviction() {
        let mut tier = CacheTierImpl::new(CacheTier::L1Hot, EvictionPolicy::LRU, 3);
        tier.insert(0, vec![0]);
        tier.insert(1, vec![1]);
        tier.insert(2, vec![2]);
        tier.contains(0);
        tier.insert(3, vec![3]);
        assert!(tier.contains(0), "Key 0 should still exist");
        assert!(!tier.contains(1), "Key 1 should be evicted");
        assert!(tier.contains(2), "Key 2 should still exist");
        assert!(tier.contains(3), "Key 3 should exist");
    }

    #[test]
    fn test_lfu_eviction() {
        let mut tier = CacheTierImpl::new(CacheTier::L1Hot, EvictionPolicy::LFU, 3);
        tier.insert(0, vec![0]);
        tier.insert(1, vec![1]);
        tier.insert(2, vec![2]);
        tier.contains(0);
        tier.contains(0);
        tier.contains(0);
        tier.contains(2);
        tier.insert(3, vec![3]);
        assert!(tier.contains(0), "Key 0 should still exist (most frequent)");
        assert!(
            !tier.contains(1),
            "Key 1 should be evicted (least frequent)"
        );
    }

    #[test]
    fn test_hit_rate_calculation() {
        let mut tier = CacheTierImpl::new(CacheTier::L1Hot, EvictionPolicy::LRU, 10);
        tier.contains(0);
        tier.contains(1);
        tier.contains(2);
        tier.insert(0, vec![0]);
        tier.contains(0);
        assert!(
            (tier.hit_rate() - 0.25).abs() < 0.01,
            "Hit rate should be ~25%"
        );
    }

    #[test]
    fn test_access_patterns() {
        // Zipfian: top key should be popular
        let zipf = generate_access_sequence(AccessPattern::Zipfian, 100, 1000, 42);
        let mut counts = HashMap::new();
        for k in &zipf {
            *counts.entry(*k).or_insert(0) += 1;
        }
        assert!(counts.get(&0).copied().unwrap_or(0) > 50);
        // Uniform: roughly even distribution
        let uni = generate_access_sequence(AccessPattern::UniformRandom, 100, 10000, 42);
        let mut uc = HashMap::new();
        for k in &uni {
            *uc.entry(*k).or_insert(0) += 1;
        }
        let avg = 10000.0 / 100.0;
        assert!((uc.get(&0).copied().unwrap_or(0) as f64 - avg).abs() < avg * 0.5);
        // Sequential: deterministic pattern
        let seq = generate_access_sequence(AccessPattern::SequentialScan, 10, 25, 42);
        assert_eq!(seq[0], 0);
        assert_eq!(seq[9], 9);
        assert_eq!(seq[10], 0);
    }

    #[test]
    fn test_percentile_and_stats() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(percentile(&data, 50), 6);
        assert_eq!(percentile(&data, 90), 10);
        assert_eq!(percentile(&[], 50), 0);
        // Empty cache stats
        let cache = HierarchicalCache::new(10, 20, 30, EvictionPolicy::LRU);
        let stats = cache.statistics();
        assert_eq!(stats.total_accesses, 0);
        assert_eq!(stats.l1_hits, 0);
    }

    #[test]
    fn test_benchmark_run() {
        let result = run_benchmark(
            AccessPattern::UniformRandom,
            EvictionPolicy::LRU,
            100,
            1000,
            10,
            20,
            30,
            42,
        );
        assert_eq!(result.statistics.total_accesses, 1000);
        assert!(
            result.statistics.overall_hit_rate >= 0.0 && result.statistics.overall_hit_rate <= 1.0
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_hit_rate_and_latency_bounds(num_accesses in 100..500usize, seed in 0u64..1000) {
            let result = run_benchmark(AccessPattern::UniformRandom, EvictionPolicy::LRU, 50, num_accesses, 10, 15, 20, seed);
            prop_assert!(result.statistics.overall_hit_rate >= 0.0 && result.statistics.overall_hit_rate <= 1.0);
            prop_assert!(result.statistics.avg_latency_ns > 0);
            prop_assert!(result.statistics.p50_latency_ns <= result.statistics.p95_latency_ns);
            prop_assert!(result.statistics.p95_latency_ns <= result.statistics.p99_latency_ns);
        }

        #[test]
        fn prop_cache_hierarchy_invariant(num_accesses in 100..500usize, seed in 0u64..1000) {
            let result = run_benchmark(AccessPattern::UniformRandom, EvictionPolicy::LRU, 100, num_accesses, 10, 20, 30, seed);
            let sum = result.statistics.l1_hits + result.statistics.l2_hits
                + result.statistics.l3_hits + result.statistics.total_misses;
            prop_assert_eq!(sum, result.statistics.total_accesses);
            prop_assert_eq!(result.statistics.total_accesses as usize, num_accesses);
        }
    }
}
