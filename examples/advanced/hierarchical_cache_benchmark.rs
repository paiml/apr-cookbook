//! Recipe: Hierarchical Cache Performance Benchmark
//! Category: Advanced | Isolation: Full | Idempotent: Yes
//!
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::env;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CacheTier {
    L1Hot,
    L2Warm,
    L3Cold,
    Miss,
}

impl CacheTier {
    #[must_use]
    pub const fn simulated_latency_ns(&self) -> u64 {
        match self {
            Self::L1Hot => 100,
            Self::L2Warm => 1_000,
            Self::L3Cold => 10_000_000,
            Self::Miss => 100_000_000,
        }
    }
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::L1Hot => "L1 (Hot)",
            Self::L2Warm => "L2 (Warm)",
            Self::L3Cold => "L3 (Cold)",
            Self::Miss => "Miss",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvictionPolicy {
    LRU,
    LFU,
    ARC,
    Clock,
    Fixed,
}

impl EvictionPolicy {
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::LRU => "LRU",
            Self::LFU => "LFU",
            Self::ARC => "ARC",
            Self::Clock => "Clock",
            Self::Fixed => "Fixed",
        }
    }
    pub const ALL: [Self; 5] = [Self::LRU, Self::LFU, Self::ARC, Self::Clock, Self::Fixed];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccessPattern {
    Zipfian,
    UniformRandom,
    TemporalBurst,
    SequentialScan,
    WorkingSetShift,
}

impl AccessPattern {
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Zipfian => "Zipfian",
            Self::UniformRandom => "Uniform Random",
            Self::TemporalBurst => "Temporal Burst",
            Self::SequentialScan => "Sequential Scan",
            Self::WorkingSetShift => "Working Set Shift",
        }
    }
    pub const ALL: [Self; 5] = [
        Self::Zipfian,
        Self::UniformRandom,
        Self::TemporalBurst,
        Self::SequentialScan,
        Self::WorkingSetShift,
    ];
}

#[derive(Debug, Clone)]
struct CacheEntry {
    key: usize,
    data: Vec<u8>,
    access_count: u64,
    last_access: Instant,
    clock_bit: bool,
}

#[derive(Debug)]
struct CacheTierImpl {
    #[allow(dead_code)]
    tier: CacheTier,
    policy: EvictionPolicy,
    capacity: usize,
    entries: HashMap<usize, CacheEntry>,
    lru_order: VecDeque<usize>,
    clock_hand: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl CacheTierImpl {
    fn new(tier: CacheTier, policy: EvictionPolicy, capacity: usize) -> Self {
        Self {
            tier,
            policy,
            capacity,
            entries: HashMap::with_capacity(capacity),
            lru_order: VecDeque::with_capacity(capacity),
            clock_hand: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    fn contains(&mut self, key: usize) -> bool {
        if self.entries.contains_key(&key) {
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.access_count += 1;
                entry.last_access = Instant::now();
                entry.clock_bit = true;
            }
            self.update_lru_order(key);
            self.hits += 1;
            true
        } else {
            self.misses += 1;
            false
        }
    }

    fn insert(&mut self, key: usize, data: Vec<u8>) {
        if self.entries.contains_key(&key) {
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.data = data;
                entry.access_count += 1;
                entry.last_access = Instant::now();
                entry.clock_bit = true;
            }
            self.update_lru_order(key);
            return;
        }
        if self.entries.len() >= self.capacity && self.policy != EvictionPolicy::Fixed {
            self.evict_one();
        }
        if self.entries.len() < self.capacity {
            self.entries.insert(
                key,
                CacheEntry {
                    key,
                    data,
                    access_count: 1,
                    last_access: Instant::now(),
                    clock_bit: true,
                },
            );
            self.lru_order.push_back(key);
        }
    }

    fn evict_one(&mut self) {
        let victim_key = match self.policy {
            EvictionPolicy::LRU => self.lru_order.front().copied(),
            EvictionPolicy::LFU => self
                .entries
                .values()
                .min_by_key(|e| e.access_count)
                .map(|e| e.key),
            EvictionPolicy::ARC => self
                .entries
                .values()
                .min_by(|a, b| {
                    let sa =
                        a.access_count as f64 / (a.last_access.elapsed().as_nanos() as f64 + 1.0);
                    let sb =
                        b.access_count as f64 / (b.last_access.elapsed().as_nanos() as f64 + 1.0);
                    sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|e| e.key),
            EvictionPolicy::Clock => self.find_clock_victim(),
            EvictionPolicy::Fixed => return,
        };
        if let Some(key) = victim_key {
            self.entries.remove(&key);
            self.lru_order.retain(|&k| k != key);
            self.evictions += 1;
        }
    }

    fn find_clock_victim(&mut self) -> Option<usize> {
        if self.entries.is_empty() {
            return None;
        }
        let keys: Vec<usize> = self.entries.keys().copied().collect();
        let n = keys.len();
        for _ in 0..2 * n {
            let key = keys[self.clock_hand % n];
            self.clock_hand = (self.clock_hand + 1) % n;
            if let Some(entry) = self.entries.get_mut(&key) {
                if !entry.clock_bit {
                    return Some(key);
                }
                entry.clock_bit = false;
            }
        }
        keys.first().copied()
    }

    fn update_lru_order(&mut self, key: usize) {
        self.lru_order.retain(|&k| k != key);
        self.lru_order.push_back(key);
    }

    fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

#[derive(Debug)]
pub struct HierarchicalCache {
    l1: CacheTierImpl,
    l2: CacheTierImpl,
    l3: CacheTierImpl,
    total_accesses: u64,
    latency_samples: Vec<u64>,
}

impl HierarchicalCache {
    pub fn new(l1_cap: usize, l2_cap: usize, l3_cap: usize, policy: EvictionPolicy) -> Self {
        Self {
            l1: CacheTierImpl::new(CacheTier::L1Hot, policy, l1_cap),
            l2: CacheTierImpl::new(CacheTier::L2Warm, policy, l2_cap),
            l3: CacheTierImpl::new(CacheTier::L3Cold, policy, l3_cap),
            total_accesses: 0,
            latency_samples: Vec::new(),
        }
    }

    pub fn access(&mut self, key: usize) -> CacheTier {
        self.total_accesses += 1;
        if self.l1.contains(key) {
            self.latency_samples
                .push(CacheTier::L1Hot.simulated_latency_ns());
            return CacheTier::L1Hot;
        }
        if self.l2.contains(key) {
            let data = self.generate_data(key);
            self.l1.insert(key, data);
            self.latency_samples
                .push(CacheTier::L2Warm.simulated_latency_ns());
            return CacheTier::L2Warm;
        }
        if self.l3.contains(key) {
            let data = self.generate_data(key);
            self.l1.insert(key, data.clone());
            self.l2.insert(key, data);
            self.latency_samples
                .push(CacheTier::L3Cold.simulated_latency_ns());
            return CacheTier::L3Cold;
        }
        let data = self.generate_data(key);
        self.l1.insert(key, data.clone());
        self.l2.insert(key, data.clone());
        self.l3.insert(key, data);
        self.latency_samples
            .push(CacheTier::Miss.simulated_latency_ns());
        CacheTier::Miss
    }

    fn generate_data(&self, key: usize) -> Vec<u8> {
        let mut data = vec![0u8; 1024];
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = ((key + i) % 256) as u8;
        }
        data
    }

    pub fn statistics(&self) -> CacheStatistics {
        let (l1h, l2h, l3h) = (self.l1.hits, self.l2.hits, self.l3.hits);
        let misses = self.total_accesses - l1h - l2h - l3h;
        let mut sorted = self.latency_samples.clone();
        sorted.sort_unstable();
        let avg = if sorted.is_empty() {
            0
        } else {
            sorted.iter().sum::<u64>() / sorted.len() as u64
        };
        CacheStatistics {
            total_accesses: self.total_accesses,
            l1_hits: l1h,
            l2_hits: l2h,
            l3_hits: l3h,
            total_misses: misses,
            l1_hit_rate: self.l1.hit_rate(),
            l2_hit_rate: self.l2.hit_rate(),
            l3_hit_rate: self.l3.hit_rate(),
            overall_hit_rate: 1.0 - (misses as f64 / self.total_accesses.max(1) as f64),
            l1_evictions: self.l1.evictions,
            l2_evictions: self.l2.evictions,
            l3_evictions: self.l3.evictions,
            avg_latency_ns: avg,
            p50_latency_ns: percentile(&sorted, 50),
            p95_latency_ns: percentile(&sorted, 95),
            p99_latency_ns: percentile(&sorted, 99),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    pub total_accesses: u64,
    pub l1_hits: u64,
    pub l2_hits: u64,
    pub l3_hits: u64,
    pub total_misses: u64,
    pub l1_hit_rate: f64,
    pub l2_hit_rate: f64,
    pub l3_hit_rate: f64,
    pub overall_hit_rate: f64,
    pub l1_evictions: u64,
    pub l2_evictions: u64,
    pub l3_evictions: u64,
    pub avg_latency_ns: u64,
    pub p50_latency_ns: u64,
    pub p95_latency_ns: u64,
    pub p99_latency_ns: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub pattern: String,
    pub policy: String,
    pub statistics: CacheStatistics,
    pub duration_ms: u64,
}

fn generate_access_sequence(
    pattern: AccessPattern,
    num_keys: usize,
    num_accesses: usize,
    seed: u64,
) -> Vec<usize> {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(seed);
    match pattern {
        AccessPattern::Zipfian => {
            let mut weights: Vec<f64> = (1..=num_keys).map(|k| 1.0 / (k as f64)).collect();
            let total: f64 = weights.iter().sum();
            for w in &mut weights {
                *w /= total;
            }
            let mut cdf = Vec::with_capacity(num_keys);
            let mut cumsum = 0.0;
            for w in weights {
                cumsum += w;
                cdf.push(cumsum);
            }
            (0..num_accesses)
                .map(|_| {
                    let r: f64 = rng.gen();
                    cdf.iter().position(|&c| c >= r).unwrap_or(num_keys - 1)
                })
                .collect()
        }
        AccessPattern::UniformRandom => (0..num_accesses)
            .map(|_| rng.gen_range(0..num_keys))
            .collect(),
        AccessPattern::TemporalBurst => {
            let hot = num_keys / 10;
            let mut seq = Vec::with_capacity(num_accesses);
            for _ in 0..num_accesses / 2 {
                seq.push(rng.gen_range(0..hot));
            }
            for _ in 0..num_accesses / 2 {
                seq.push(rng.gen_range(hot..num_keys));
            }
            seq
        }
        AccessPattern::SequentialScan => (0..num_accesses).map(|i| i % num_keys).collect(),
        AccessPattern::WorkingSetShift => {
            let ws = num_keys / 5;
            (0..num_accesses)
                .map(|i| {
                    let center =
                        ((num_keys - ws) as f64 * (i as f64 / num_accesses as f64)) as usize;
                    (center + rng.gen_range(0..ws)) % num_keys
                })
                .collect()
        }
    }
}

fn percentile(sorted: &[u64], p: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    sorted[(sorted.len() * p / 100).min(sorted.len() - 1)]
}

pub fn run_benchmark(
    pattern: AccessPattern,
    policy: EvictionPolicy,
    num_keys: usize,
    num_accesses: usize,
    l1_cap: usize,
    l2_cap: usize,
    l3_cap: usize,
    seed: u64,
) -> BenchmarkResult {
    let mut cache = HierarchicalCache::new(l1_cap, l2_cap, l3_cap, policy);
    let sequence = generate_access_sequence(pattern, num_keys, num_accesses, seed);
    let start = Instant::now();
    for key in sequence {
        cache.access(key);
    }
    let duration = start.elapsed();
    BenchmarkResult {
        pattern: pattern.name().to_string(),
        policy: policy.name().to_string(),
        statistics: cache.statistics(),
        duration_ms: duration.as_millis() as u64,
    }
}

fn format_latency(ns: u64) -> String {
    if ns >= 1_000_000 {
        format!("{:.1} ms", ns as f64 / 1_000_000.0)
    } else if ns >= 1_000 {
        format!("{:.1} us", ns as f64 / 1_000.0)
    } else {
        format!("{ns} ns")
    }
}

fn print_results(results: &[BenchmarkResult]) {
    println!("\n{}", "=".repeat(100));
    println!("             HIERARCHICAL CACHE BENCHMARK RESULTS");
    println!("{}\n{:-<100}", "=".repeat(100), "");
    println!(
        " {:15} | {:6} | {:8} | {:8} | {:8} | {:8} | {:12} | {:10}",
        "PATTERN", "POLICY", "HIT RATE", "L1 HITS", "L2 HITS", "L3 HITS", "AVG LAT (ns)", "P99 LAT"
    );
    println!("{:-<100}", "");
    for r in results {
        println!(
            " {:15} | {:6} | {:7.1}% | {:8} | {:8} | {:8} | {:12} | {:10}",
            r.pattern,
            r.policy,
            r.statistics.overall_hit_rate * 100.0,
            r.statistics.l1_hits,
            r.statistics.l2_hits,
            r.statistics.l3_hits,
            r.statistics.avg_latency_ns,
            format_latency(r.statistics.p99_latency_ns)
        );
    }
    println!("{:-<100}", "");
    println!("\n LATENCY DISTRIBUTION:");
    println!(
        " {:15} | {:6} | {:>12} | {:>12} | {:>12} | {:>12}",
        "Pattern", "Policy", "P50", "P95", "P99", "Avg"
    );
    println!("{:-<80}", "");
    for r in results {
        println!(
            " {:15} | {:6} | {:>12} | {:>12} | {:>12} | {:>12}",
            r.pattern,
            r.policy,
            format_latency(r.statistics.p50_latency_ns),
            format_latency(r.statistics.p95_latency_ns),
            format_latency(r.statistics.p99_latency_ns),
            format_latency(r.statistics.avg_latency_ns)
        );
    }
}

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
