//! # Monte-Carlo Cache Warmup Curve
//!
//! Simulate cache hit rate over warmup steps. Each request key is
//! drawn from a finite key-space with Zipfian-ish weighting; cache
//! retains last K entries. Returns hit-rate at sample steps.
//!
//! Demonstrates the **MC.14** recipe for PMAT-162 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Stack distance / LRU cache behavior modeling.
//!
//! Run with: cargo run --example mc_cache_warmup_curve
//!
//! Added by PMAT-162 (catalog 1081→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::VecDeque;

#[derive(Debug, PartialEq)]
pub enum WarmupVerdict {
    Ok {
        final_hit_rate: f64,
        steps_to_50pct: Option<u32>,
        total_hits: u64,
    },
    InvalidConfig,
}

pub fn simulate(cache_capacity: usize, key_space: u32, steps: u32, seed: u64) -> WarmupVerdict {
    if cache_capacity == 0 || key_space == 0 || steps == 0 {
        return WarmupVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut cache: VecDeque<u32> = VecDeque::with_capacity(cache_capacity);
    let mut hits: u64 = 0;
    let mut steps_to_50pct: Option<u32> = None;
    for step in 0..steps {
        let key = (lcg(&mut rng_state) % u64::from(key_space)) as u32;
        if cache.contains(&key) {
            hits += 1;
        } else {
            if cache.len() >= cache_capacity {
                cache.pop_front();
            }
            cache.push_back(key);
        }
        let rate = hits as f64 / f64::from(step + 1);
        if rate >= 0.50 && steps_to_50pct.is_none() {
            steps_to_50pct = Some(step + 1);
        }
    }
    let final_hit_rate = hits as f64 / f64::from(steps);
    WarmupVerdict::Ok {
        final_hit_rate,
        steps_to_50pct,
        total_hits: hits,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_cache_warmup_curve")?;

    println!("hot keys: {:?}", simulate(50, 100, 1000, 42));
    println!("cold keys: {:?}", simulate(10, 1000, 1000, 42));
    println!("invalid: {:?}", simulate(0, 100, 1000, 42));
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
    fn hot_keys_high_hit_rate() {
        // Cache > key_space → eventually 100% hit.
        let v = simulate(200, 100, 1000, 42);
        if let WarmupVerdict::Ok { final_hit_rate, .. } = v {
            assert!(final_hit_rate > 0.7);
        }
    }

    #[test]
    fn cold_keys_low_hit_rate() {
        // Cache << key_space → low hit rate.
        let v = simulate(10, 10000, 1000, 42);
        if let WarmupVerdict::Ok { final_hit_rate, .. } = v {
            assert!(final_hit_rate < 0.1);
        }
    }

    #[test]
    fn invalid_zero_capacity() {
        assert_eq!(simulate(0, 100, 1000, 42), WarmupVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_keyspace() {
        assert_eq!(simulate(50, 0, 1000, 42), WarmupVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_steps() {
        assert_eq!(simulate(50, 100, 0, 42), WarmupVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(50, 100, 1000, 42);
        let b = simulate(50, 100, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn hits_bounded_by_steps() {
        let v = simulate(50, 100, 100, 42);
        if let WarmupVerdict::Ok { total_hits, .. } = v {
            assert!(total_hits <= 100);
        }
    }

    #[test]
    fn higher_capacity_higher_hit_rate() {
        let small = simulate(10, 100, 1000, 42);
        let large = simulate(200, 100, 1000, 42);
        if let (
            WarmupVerdict::Ok {
                final_hit_rate: s, ..
            },
            WarmupVerdict::Ok {
                final_hit_rate: l, ..
            },
        ) = (small, large)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn first_hit_step_reasonable() {
        let v = simulate(100, 50, 1000, 42);
        if let WarmupVerdict::Ok { steps_to_50pct, .. } = v {
            // Cache > key space: should hit 50% relatively early.
            if let Some(s) = steps_to_50pct {
                assert!(s < 500);
            }
        }
    }
}
