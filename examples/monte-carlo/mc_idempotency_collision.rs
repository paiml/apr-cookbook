//! # Monte-Carlo Idempotency Key Collision
//!
//! Sim collision rate of N idempotency keys drawn uniformly from a
//! 2^bits hash space (UUID-like). Returns observed collision rate
//! and analytical estimate.
//!
//! Demonstrates the **MC.54** recipe for PMAT-175 (catalog crosses 1200).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: birthday paradox + Stripe Idempotency-Key collision study.
//!
//! Run with: cargo run --example mc_idempotency_collision
//!
//! Added by PMAT-175 (catalog 1198→).

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

pub fn simulate(keys_per_trial: u32, bits: u32, trials: u32, seed: u64) -> CollisionVerdict {
    if keys_per_trial < 2 || bits == 0 || bits > 60 || trials == 0 {
        return CollisionVerdict::InvalidConfig;
    }
    let space = 1u64 << bits;
    let shift = 64 - bits;
    let mut rng_state = seed | 1;
    let mut collisions = 0u32;
    for _ in 0..trials {
        let mut seen: BTreeSet<u64> = BTreeSet::new();
        let mut had = false;
        for _ in 0..keys_per_trial {
            // Use high bits — LCG low bits have short cycles.
            let key = lcg(&mut rng_state) >> shift;
            if !seen.insert(key) {
                had = true;
                break;
            }
        }
        if had {
            collisions += 1;
        }
    }
    let observed_collision_rate = f64::from(collisions) / f64::from(trials);
    let n = f64::from(keys_per_trial);
    let s = space as f64;
    let analytical_estimate = 1.0 - (-n * (n - 1.0) / (2.0 * s)).exp();
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
    let _ctx = RecipeContext::new("mc_idempotency_collision")?;

    println!("10 keys / 16 bits: {:?}", simulate(10, 16, 1000, 42));
    println!("1000 keys / 60 bits: {:?}", simulate(1000, 60, 100, 42));
    println!("invalid: {:?}", simulate(1, 16, 1000, 42));
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
    fn small_space_high_collision() {
        let v = simulate(20, 8, 1000, 42);
        if let CollisionVerdict::Ok {
            observed_collision_rate,
            ..
        } = v
        {
            assert!(observed_collision_rate > 0.1);
        }
    }

    #[test]
    fn large_space_low_collision() {
        let v = simulate(10, 60, 1000, 42);
        if let CollisionVerdict::Ok {
            observed_collision_rate,
            ..
        } = v
        {
            assert!(observed_collision_rate < 0.01);
        }
    }

    #[test]
    fn observed_near_analytical() {
        let v = simulate(20, 12, 5000, 42);
        if let CollisionVerdict::Ok {
            observed_collision_rate,
            analytical_estimate,
        } = v
        {
            assert!((observed_collision_rate - analytical_estimate).abs() < 0.1);
        }
    }

    #[test]
    fn invalid_few_keys() {
        assert_eq!(simulate(1, 16, 1000, 42), CollisionVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_bits() {
        assert_eq!(simulate(10, 0, 1000, 42), CollisionVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_high_bits() {
        assert_eq!(simulate(10, 64, 1000, 42), CollisionVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_trials() {
        assert_eq!(simulate(10, 16, 0, 42), CollisionVerdict::InvalidConfig);
    }

    #[test]
    fn rate_in_unit_range() {
        let v = simulate(20, 16, 1000, 42);
        if let CollisionVerdict::Ok {
            observed_collision_rate,
            ..
        } = v
        {
            assert!((0.0..=1.0).contains(&observed_collision_rate));
        }
    }

    #[test]
    fn deterministic() {
        let a = simulate(20, 16, 1000, 42);
        let b = simulate(20, 16, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn more_keys_more_collisions() {
        let few = simulate(10, 16, 5000, 42);
        let many = simulate(100, 16, 5000, 42);
        if let (
            CollisionVerdict::Ok {
                observed_collision_rate: f,
                ..
            },
            CollisionVerdict::Ok {
                observed_collision_rate: m,
                ..
            },
        ) = (few, many)
        {
            assert!(m > f);
        }
    }
}
