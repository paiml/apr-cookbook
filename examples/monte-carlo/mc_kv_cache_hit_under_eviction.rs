//! # Monte-Carlo KV-Cache Hit Under Eviction Pressure
//!
//! Sim KV-cache hit rate as eviction policy varies (LRU vs random).
//! Under heavy memory pressure, returns observed hit rate per policy.
//!
//! Demonstrates the **MC.48** recipe for PMAT-173 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: cache-replacement-policy benchmarks (Belady).
//!
//! Run with: cargo run --example mc_kv_cache_hit_under_eviction
//!
//! Added by PMAT-173 (catalog 1180→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    Lru,
    Random,
}

#[derive(Debug, PartialEq)]
pub enum CacheVerdict {
    Ok { hit_rate: f64, evictions: u32 },
    InvalidConfig,
}

pub fn simulate(
    capacity: u32,
    keyspace: u32,
    requests: u32,
    policy: EvictionPolicy,
    seed: u64,
) -> CacheVerdict {
    if capacity == 0 || keyspace == 0 || requests == 0 {
        return CacheVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut cache: Vec<u32> = Vec::with_capacity(capacity as usize);
    let mut hits = 0u32;
    let mut evictions = 0u32;
    for _ in 0..requests {
        let key = (lcg(&mut rng_state) % u64::from(keyspace)) as u32;
        if let Some(pos) = cache.iter().position(|k| *k == key) {
            hits += 1;
            if matches!(policy, EvictionPolicy::Lru) {
                cache.remove(pos);
                cache.push(key);
            }
        } else {
            if cache.len() >= capacity as usize {
                let evict_idx = match policy {
                    EvictionPolicy::Lru => 0,
                    EvictionPolicy::Random => (lcg(&mut rng_state) % cache.len() as u64) as usize,
                };
                cache.remove(evict_idx);
                evictions += 1;
            }
            cache.push(key);
        }
    }
    let hit_rate = f64::from(hits) / f64::from(requests);
    CacheVerdict::Ok {
        hit_rate,
        evictions,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_kv_cache_hit_under_eviction")?;

    println!(
        "lru: {:?}",
        simulate(50, 200, 10_000, EvictionPolicy::Lru, 42)
    );
    println!(
        "random: {:?}",
        simulate(50, 200, 10_000, EvictionPolicy::Random, 42)
    );
    println!(
        "invalid: {:?}",
        simulate(0, 200, 10_000, EvictionPolicy::Lru, 42)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn capacity_larger_than_keyspace_high_hit() {
        let v = simulate(200, 50, 10_000, EvictionPolicy::Lru, 42);
        if let CacheVerdict::Ok { hit_rate, .. } = v {
            assert!(hit_rate > 0.95);
        }
    }

    #[test]
    fn tight_cache_low_hit() {
        let v = simulate(2, 1000, 1000, EvictionPolicy::Lru, 42);
        if let CacheVerdict::Ok { hit_rate, .. } = v {
            assert!(hit_rate < 0.05);
        }
    }

    #[test]
    fn invalid_zero_capacity() {
        assert_eq!(
            simulate(0, 200, 10_000, EvictionPolicy::Lru, 42),
            CacheVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_keyspace() {
        assert_eq!(
            simulate(50, 0, 10_000, EvictionPolicy::Lru, 42),
            CacheVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_requests() {
        assert_eq!(
            simulate(50, 200, 0, EvictionPolicy::Lru, 42),
            CacheVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(50, 200, 1000, EvictionPolicy::Lru, 42);
        let b = simulate(50, 200, 1000, EvictionPolicy::Lru, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn evictions_only_when_full() {
        let v = simulate(1000, 100, 100, EvictionPolicy::Lru, 42);
        if let CacheVerdict::Ok { evictions, .. } = v {
            assert_eq!(evictions, 0);
        }
    }

    #[test]
    fn hit_rate_in_unit_range() {
        let v = simulate(50, 200, 1000, EvictionPolicy::Random, 42);
        if let CacheVerdict::Ok { hit_rate, .. } = v {
            assert!((0.0..=1.0).contains(&hit_rate));
        }
    }

    #[test]
    fn random_and_lru_both_work() {
        let lru = simulate(50, 200, 1000, EvictionPolicy::Lru, 42);
        let rand = simulate(50, 200, 1000, EvictionPolicy::Random, 42);
        assert!(matches!(lru, CacheVerdict::Ok { .. }));
        assert!(matches!(rand, CacheVerdict::Ok { .. }));
    }
}
