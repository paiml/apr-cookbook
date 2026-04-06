#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports,
    clippy::upper_case_acronyms
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use proptest::prelude::*;
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
pub struct CacheEntry {
    pub key: usize,
    pub data: Vec<u8>,
    pub access_count: u64,
    pub last_access: Instant,
    pub clock_bit: bool,
}

#[derive(Debug)]
pub struct CacheTierImpl {
    #[allow(dead_code)]
    pub tier: CacheTier,
    pub policy: EvictionPolicy,
    pub capacity: usize,
    pub entries: HashMap<usize, CacheEntry>,
    pub lru_order: VecDeque<usize>,
    pub clock_hand: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl CacheTierImpl {
    pub fn new(tier: CacheTier, policy: EvictionPolicy, capacity: usize) -> Self {
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

    pub fn contains(&mut self, key: usize) -> bool {
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

    pub fn insert(&mut self, key: usize, data: Vec<u8>) {
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

    pub fn evict_one(&mut self) {
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

    pub fn find_clock_victim(&mut self) -> Option<usize> {
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

    pub fn update_lru_order(&mut self, key: usize) {
        self.lru_order.retain(|&k| k != key);
        self.lru_order.push_back(key);
    }

    pub fn hit_rate(&self) -> f64 {
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
    pub l1: CacheTierImpl,
    pub l2: CacheTierImpl,
    pub l3: CacheTierImpl,
    pub total_accesses: u64,
    pub latency_samples: Vec<u64>,
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

    pub fn generate_data(&self, key: usize) -> Vec<u8> {
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

pub fn generate_access_sequence(
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

pub fn percentile(sorted: &[u64], p: usize) -> u64 {
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

pub fn format_latency(ns: u64) -> String {
    if ns >= 1_000_000 {
        format!("{:.1} ms", ns as f64 / 1_000_000.0)
    } else if ns >= 1_000 {
        format!("{:.1} us", ns as f64 / 1_000.0)
    } else {
        format!("{ns} ns")
    }
}

pub fn print_results(results: &[BenchmarkResult]) {
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
