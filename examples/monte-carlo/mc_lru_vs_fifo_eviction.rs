//! # Monte-Carlo LRU vs FIFO Eviction
//!
//! Run the same access trace through both LRU and FIFO caches.
//! Returns each policy's hit-rate to compare.
//!
//! Demonstrates the **MC.72** recipe for PMAT-183 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Belady, IBM Sys J 5 (1966) §3 (optimal replacement);
//!  Sleator & Tarjan, JACM 32 (1985) §2.
//!
//! Run with: cargo run --example mc_lru_vs_fifo_eviction
//!
//! Added by PMAT-183 (catalog 1270→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::VecDeque;

#[derive(Debug, PartialEq)]
pub enum CompareVerdict {
    Ok {
        lru_hit_rate: f64,
        fifo_hit_rate: f64,
    },
    InvalidConfig,
}

pub fn simulate(accesses: u32, keyspace: u32, cache_size: u32, seed: u64) -> CompareVerdict {
    if accesses == 0 || keyspace == 0 || cache_size == 0 {
        return CompareVerdict::InvalidConfig;
    }
    let mut lru: VecDeque<u32> = VecDeque::with_capacity(cache_size as usize);
    let mut fifo: VecDeque<u32> = VecDeque::with_capacity(cache_size as usize);
    let mut lru_hits = 0u32;
    let mut fifo_hits = 0u32;
    let mut rng_state = seed | 1;
    for _ in 0..accesses {
        let key = ((lcg(&mut rng_state) >> 32) as u32) % keyspace;
        // LRU: hit moves to back; miss appends, evicting front if full.
        if let Some(pos) = lru.iter().position(|k| *k == key) {
            lru.remove(pos);
            lru.push_back(key);
            lru_hits += 1;
        } else {
            if lru.len() >= cache_size as usize {
                lru.pop_front();
            }
            lru.push_back(key);
        }
        // FIFO: hit just counts; miss evicts oldest insertion.
        if fifo.contains(&key) {
            fifo_hits += 1;
        } else {
            if fifo.len() >= cache_size as usize {
                fifo.pop_front();
            }
            fifo.push_back(key);
        }
    }
    CompareVerdict::Ok {
        lru_hit_rate: f64::from(lru_hits) / f64::from(accesses),
        fifo_hit_rate: f64::from(fifo_hits) / f64::from(accesses),
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_lru_vs_fifo_eviction")?;

    println!("hot working set: {:?}", simulate(10_000, 50, 10, 42));
    println!("uniform random: {:?}", simulate(10_000, 1000, 10, 42));
    println!("invalid: {:?}", simulate(0, 50, 10, 42));
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
    fn hot_working_set_lru_advantage() {
        // With small keyspace + cache, LRU should be at least as good.
        let v = simulate(10_000, 20, 10, 42);
        if let CompareVerdict::Ok {
            lru_hit_rate,
            fifo_hit_rate,
        } = v
        {
            assert!(lru_hit_rate >= fifo_hit_rate - 0.01);
        }
    }

    #[test]
    fn uniform_random_close_rates() {
        let v = simulate(5000, 1000, 10, 42);
        if let CompareVerdict::Ok {
            lru_hit_rate,
            fifo_hit_rate,
        } = v
        {
            // Uniform random → both near 1/100 = 0.01.
            assert!((lru_hit_rate - fifo_hit_rate).abs() < 0.05);
        }
    }

    #[test]
    fn rates_in_unit_range() {
        let v = simulate(1000, 100, 10, 42);
        if let CompareVerdict::Ok {
            lru_hit_rate,
            fifo_hit_rate,
        } = v
        {
            assert!((0.0..=1.0).contains(&lru_hit_rate));
            assert!((0.0..=1.0).contains(&fifo_hit_rate));
        }
    }

    #[test]
    fn invalid_zero_accesses() {
        assert_eq!(simulate(0, 50, 10, 42), CompareVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_keyspace() {
        assert_eq!(simulate(100, 0, 10, 42), CompareVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_cache() {
        assert_eq!(simulate(100, 50, 0, 42), CompareVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 50, 10, 42);
        let b = simulate(500, 50, 10, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn cache_at_keyspace_size_full_hits_after_warmup() {
        let v = simulate(10_000, 5, 5, 42);
        if let CompareVerdict::Ok { lru_hit_rate, .. } = v {
            // After loading all 5 keys, every access hits.
            assert!(lru_hit_rate > 0.99);
        }
    }

    #[test]
    fn small_cache_low_hit_rate() {
        let big = simulate(10_000, 100, 50, 42);
        let tiny = simulate(10_000, 100, 1, 42);
        if let (
            CompareVerdict::Ok {
                lru_hit_rate: b, ..
            },
            CompareVerdict::Ok {
                lru_hit_rate: t, ..
            },
        ) = (big, tiny)
        {
            assert!(b > t);
        }
    }

    #[test]
    fn first_access_always_miss() {
        let v = simulate(1, 100, 10, 42);
        if let CompareVerdict::Ok {
            lru_hit_rate,
            fifo_hit_rate,
        } = v
        {
            assert_eq!(lru_hit_rate, 0.0);
            assert_eq!(fifo_hit_rate, 0.0);
        }
    }

    #[test]
    fn higher_keyspace_lower_hit_rate() {
        let small = simulate(5000, 20, 10, 42);
        let large = simulate(5000, 1000, 10, 42);
        if let (
            CompareVerdict::Ok {
                lru_hit_rate: s, ..
            },
            CompareVerdict::Ok {
                lru_hit_rate: l, ..
            },
        ) = (small, large)
        {
            assert!(s > l);
        }
    }
}
