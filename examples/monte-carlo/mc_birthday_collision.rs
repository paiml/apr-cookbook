//! # Monte-Carlo Birthday-Collision Probability
//!
//! Estimate hash-collision probability across N inputs given a hash
//! space of K buckets. Compares observed vs analytical
//! `1 - exp(-N(N-1) / (2K))`.
//!
//! Demonstrates the **MC.24** recipe for PMAT-165 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Birthday paradox (Feller, Probability vol. 1).
//!
//! Run with: cargo run --example mc_birthday_collision
//!
//! Added by PMAT-165 (catalog 1108→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum CollisionVerdict {
    Ok {
        observed_collision_rate: f64,
        analytical_estimate: f64,
    },
    InvalidConfig,
}

pub fn estimate(n_inputs: u32, hash_space: u64, trials: u32, seed: u64) -> CollisionVerdict {
    if n_inputs < 2 || hash_space == 0 || trials == 0 {
        return CollisionVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut collisions: u32 = 0;
    for _ in 0..trials {
        let mut seen: BTreeSet<u64> = BTreeSet::new();
        let mut had_collision = false;
        for _ in 0..n_inputs {
            let h = lcg(&mut rng_state) % hash_space;
            if !seen.insert(h) {
                had_collision = true;
                break;
            }
        }
        if had_collision {
            collisions += 1;
        }
    }
    let observed_collision_rate = f64::from(collisions) / f64::from(trials);
    let n = f64::from(n_inputs);
    let k = hash_space as f64;
    let analytical_estimate = 1.0 - (-n * (n - 1.0) / (2.0 * k)).exp();
    CollisionVerdict::Ok {
        observed_collision_rate,
        analytical_estimate,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_birthday_collision")?;

    println!("23 inputs, 365 buckets: {:?}", estimate(23, 365, 1000, 42));
    println!(
        "128-bit hash, 10k inputs: {:?}",
        estimate(10_000, 1u64 << 60, 100, 42)
    );
    println!("invalid: {:?}", estimate(1, 100, 100, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn classic_birthday_paradox() {
        let v = estimate(23, 365, 1000, 42);
        if let CollisionVerdict::Ok {
            observed_collision_rate,
            analytical_estimate,
        } = v
        {
            // 23/365 → ~50% collision probability.
            assert!((analytical_estimate - 0.5).abs() < 0.05);
            assert!((observed_collision_rate - analytical_estimate).abs() < 0.10);
        }
    }

    #[test]
    fn small_inputs_low_collision() {
        let v = estimate(2, 1_000_000, 1000, 42);
        if let CollisionVerdict::Ok {
            analytical_estimate,
            ..
        } = v
        {
            assert!(analytical_estimate < 0.001);
        }
    }

    #[test]
    fn large_hash_space_low_collision() {
        let v = estimate(100, 1_000_000_000, 100, 42);
        if let CollisionVerdict::Ok {
            observed_collision_rate,
            ..
        } = v
        {
            assert!(observed_collision_rate < 0.1);
        }
    }

    #[test]
    fn invalid_one_input() {
        assert_eq!(estimate(1, 365, 100, 42), CollisionVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_buckets() {
        assert_eq!(estimate(10, 0, 100, 42), CollisionVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_trials() {
        assert_eq!(estimate(10, 100, 0, 42), CollisionVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = estimate(23, 365, 100, 42);
        let b = estimate(23, 365, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn rates_in_unit_interval() {
        let v = estimate(50, 100, 1000, 42);
        if let CollisionVerdict::Ok {
            observed_collision_rate,
            analytical_estimate,
        } = v
        {
            assert!((0.0..=1.0).contains(&observed_collision_rate));
            assert!((0.0..=1.0).contains(&analytical_estimate));
        }
    }

    #[test]
    fn collision_increases_with_n() {
        let lo = estimate(5, 100, 1000, 42);
        let hi = estimate(50, 100, 1000, 42);
        if let (
            CollisionVerdict::Ok {
                observed_collision_rate: a,
                ..
            },
            CollisionVerdict::Ok {
                observed_collision_rate: b,
                ..
            },
        ) = (lo, hi)
        {
            assert!(b > a);
        }
    }

    #[test]
    fn n_equals_buckets_high_collision() {
        // Pigeonhole: N + 1 inputs in N buckets → guaranteed collision.
        let v = estimate(101, 100, 100, 42);
        if let CollisionVerdict::Ok {
            observed_collision_rate,
            ..
        } = v
        {
            assert!((observed_collision_rate - 1.0).abs() < 1e-9);
        }
    }
}
