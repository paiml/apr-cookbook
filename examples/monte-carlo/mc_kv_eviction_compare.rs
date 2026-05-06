//! # Monte-Carlo KV Eviction Policy Comparison
//!
//! Compare LRU vs FIFO eviction strategies under the same access trace.
//! Returns observed hit rate per policy.
//!
//! Demonstrates the **MC.18** recipe for PMAT-163 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Caching algorithms (Belady 1966 + Sleator-Tarjan).
//!
//! Run with: cargo run --example mc_kv_eviction_compare
//!
//! Added by PMAT-163 (catalog 1090→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::VecDeque;

#[derive(Debug, PartialEq)]
pub enum EvictVerdict {
    Ok {
        lru_hit_rate: f64,
        fifo_hit_rate: f64,
        better: &'static str,
    },
    InvalidConfig,
}

pub fn compare(cache_size: usize, keyspace: u32, requests: u32, seed: u64) -> EvictVerdict {
    if cache_size == 0 || keyspace == 0 || requests == 0 {
        return EvictVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let trace: Vec<u32> = (0..requests)
        .map(|_| (lcg(&mut rng_state) % u64::from(keyspace)) as u32)
        .collect();
    let lru_hits = simulate_lru(&trace, cache_size);
    let fifo_hits = simulate_fifo(&trace, cache_size);
    let lru_hit_rate = lru_hits as f64 / f64::from(requests);
    let fifo_hit_rate = fifo_hits as f64 / f64::from(requests);
    let better = if lru_hit_rate > fifo_hit_rate {
        "LRU"
    } else if fifo_hit_rate > lru_hit_rate {
        "FIFO"
    } else {
        "Tie"
    };
    EvictVerdict::Ok {
        lru_hit_rate,
        fifo_hit_rate,
        better,
    }
}

fn simulate_lru(trace: &[u32], cap: usize) -> u64 {
    let mut cache: Vec<u32> = Vec::with_capacity(cap);
    let mut hits = 0u64;
    for &k in trace {
        if let Some(pos) = cache.iter().position(|x| *x == k) {
            cache.remove(pos);
            cache.push(k);
            hits += 1;
        } else {
            if cache.len() >= cap {
                cache.remove(0);
            }
            cache.push(k);
        }
    }
    hits
}

fn simulate_fifo(trace: &[u32], cap: usize) -> u64 {
    let mut cache: VecDeque<u32> = VecDeque::with_capacity(cap);
    let mut hits = 0u64;
    for &k in trace {
        if cache.contains(&k) {
            hits += 1;
        } else {
            if cache.len() >= cap {
                cache.pop_front();
            }
            cache.push_back(k);
        }
    }
    hits
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_kv_eviction_compare")?;

    println!("typical: {:?}", compare(50, 100, 1000, 42));
    println!("tight cache: {:?}", compare(10, 100, 1000, 42));
    println!("invalid: {:?}", compare(0, 100, 1000, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_returns_a_winner() {
        let v = compare(50, 100, 1000, 42);
        if let EvictVerdict::Ok { better, .. } = v {
            assert!(["LRU", "FIFO", "Tie"].contains(&better));
        }
    }

    #[test]
    fn invalid_zero_cache() {
        assert_eq!(compare(0, 100, 1000, 42), EvictVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_keyspace() {
        assert_eq!(compare(50, 0, 1000, 42), EvictVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_requests() {
        assert_eq!(compare(50, 100, 0, 42), EvictVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = compare(50, 100, 1000, 42);
        let b = compare(50, 100, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn hit_rates_in_range() {
        let v = compare(50, 100, 1000, 42);
        if let EvictVerdict::Ok {
            lru_hit_rate,
            fifo_hit_rate,
            ..
        } = v
        {
            assert!((0.0..=1.0).contains(&lru_hit_rate));
            assert!((0.0..=1.0).contains(&fifo_hit_rate));
        }
    }

    #[test]
    fn cache_bigger_than_keyspace_perfect() {
        let v = compare(200, 100, 1000, 42);
        if let EvictVerdict::Ok {
            lru_hit_rate,
            fifo_hit_rate,
            ..
        } = v
        {
            // Eventually all keys cached → high hit rate.
            assert!(lru_hit_rate > 0.8);
            assert!(fifo_hit_rate > 0.8);
        }
    }

    #[test]
    fn tiny_cache_low_hit() {
        let v = compare(2, 1000, 100, 42);
        if let EvictVerdict::Ok { lru_hit_rate, .. } = v {
            assert!(lru_hit_rate < 0.1);
        }
    }

    #[test]
    fn single_request_always_miss() {
        let v = compare(10, 100, 1, 42);
        if let EvictVerdict::Ok {
            lru_hit_rate,
            fifo_hit_rate,
            ..
        } = v
        {
            assert!((lru_hit_rate - 0.0).abs() < 1e-9);
            assert!((fifo_hit_rate - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn equal_hit_rates_tie() {
        // Cache covers keyspace → both 100% hit eventually.
        let v = compare(200, 50, 10_000, 42);
        if let EvictVerdict::Ok {
            lru_hit_rate,
            fifo_hit_rate,
            better,
            ..
        } = v
        {
            if (lru_hit_rate - fifo_hit_rate).abs() < 1e-9 {
                assert_eq!(better, "Tie");
            }
        }
    }
}
