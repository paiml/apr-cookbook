//! # Monte-Carlo Cache Thrashing
//!
//! Sim cache thrashing: when working-set size > cache size, every
//! access misses. Returns observed thrashing-fraction (proportion of
//! steps where eviction occurred).
//!
//! Demonstrates the **MC.57** recipe for PMAT-176 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Belady's anomaly + working-set theory (Denning 1968).
//!
//! Run with: cargo run --example mc_eviction_thrashing
//!
//! Added by PMAT-176 (catalog 1207→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::VecDeque;

#[derive(Debug, PartialEq)]
pub enum ThrashVerdict {
    Ok {
        eviction_rate: f64,
        is_thrashing: bool,
    },
    InvalidConfig,
}

pub fn simulate(cache_size: u32, working_set: u32, steps: u32, seed: u64) -> ThrashVerdict {
    if cache_size == 0 || working_set == 0 || steps == 0 {
        return ThrashVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut cache: VecDeque<u32> = VecDeque::with_capacity(cache_size as usize);
    let mut evictions = 0u32;
    for _ in 0..steps {
        let key = ((lcg(&mut rng_state) >> 32) as u32) % working_set;
        if !cache.contains(&key) {
            if cache.len() >= cache_size as usize {
                cache.pop_front();
                evictions += 1;
            }
            cache.push_back(key);
        }
    }
    let eviction_rate = f64::from(evictions) / f64::from(steps);
    let is_thrashing = working_set > cache_size && eviction_rate > 0.5;
    ThrashVerdict::Ok {
        eviction_rate,
        is_thrashing,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_eviction_thrashing")?;

    println!("fits: {:?}", simulate(100, 50, 10_000, 42));
    println!("thrashing: {:?}", simulate(10, 200, 10_000, 42));
    println!("invalid: {:?}", simulate(0, 50, 100, 42));
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
    fn working_set_fits_no_thrashing() {
        let v = simulate(100, 50, 10_000, 42);
        if let ThrashVerdict::Ok { is_thrashing, .. } = v {
            assert!(!is_thrashing);
        }
    }

    #[test]
    fn tight_cache_thrashes() {
        let v = simulate(10, 200, 10_000, 42);
        if let ThrashVerdict::Ok { is_thrashing, .. } = v {
            assert!(is_thrashing);
        }
    }

    #[test]
    fn working_set_equals_cache_low_eviction() {
        let v = simulate(100, 100, 10_000, 42);
        if let ThrashVerdict::Ok { is_thrashing, .. } = v {
            assert!(!is_thrashing);
        }
    }

    #[test]
    fn invalid_zero_cache() {
        assert_eq!(simulate(0, 50, 100, 42), ThrashVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_working_set() {
        assert_eq!(simulate(100, 0, 100, 42), ThrashVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_steps() {
        assert_eq!(simulate(100, 50, 0, 42), ThrashVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(50, 100, 1000, 42);
        let b = simulate(50, 100, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn rate_in_unit_range() {
        let v = simulate(50, 100, 1000, 42);
        if let ThrashVerdict::Ok { eviction_rate, .. } = v {
            assert!((0.0..=1.0).contains(&eviction_rate));
        }
    }

    #[test]
    fn higher_working_set_higher_eviction() {
        let small = simulate(20, 30, 5000, 42);
        let large = simulate(20, 200, 5000, 42);
        if let (
            ThrashVerdict::Ok {
                eviction_rate: s, ..
            },
            ThrashVerdict::Ok {
                eviction_rate: l, ..
            },
        ) = (small, large)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn cache_size_one_extreme() {
        let v = simulate(1, 100, 1000, 42);
        if let ThrashVerdict::Ok { eviction_rate, .. } = v {
            assert!(eviction_rate > 0.5);
        }
    }
}
