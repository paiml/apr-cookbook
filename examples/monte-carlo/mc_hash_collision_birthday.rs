//! # Monte-Carlo Hash Collision Birthday Paradox
//!
//! Sim hash-collision detection by sampling random integers in
//! [0, hash_space) until two collide. Returns expected vs observed
//! mean draws to first collision.
//!
//! Demonstrates the **MC.153** recipe for PMAT-209 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Bloom, "Generalized Birthday Problem" (1973); birthday-
//!  bound √(πN/2) for hash-collision attack analysis.
//!
//! Run with: cargo run --example mc_hash_collision_birthday
//!
//! Added by PMAT-209 (catalog 1504→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashSet;

#[derive(Debug, PartialEq)]
pub enum BirthdayVerdict {
    Ok {
        mean_draws_to_collision: u32,
        max_draws_observed: u32,
    },
    InvalidConfig,
}

pub fn simulate(hash_space: u32, trials: u32, seed: u64) -> BirthdayVerdict {
    if hash_space < 10 || trials < 100 {
        return BirthdayVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut total_draws: u64 = 0;
    let mut max_draws = 0u32;
    for _ in 0..trials {
        let mut seen: HashSet<u32> = HashSet::new();
        let mut draws = 0u32;
        loop {
            let v = (lcg(&mut state) % hash_space as u64) as u32;
            draws += 1;
            if !seen.insert(v) {
                break;
            }
            if draws > hash_space {
                break; // pigeonhole guarantee
            }
        }
        total_draws += draws as u64;
        if draws > max_draws {
            max_draws = draws;
        }
    }
    BirthdayVerdict::Ok {
        mean_draws_to_collision: (total_draws / trials as u64) as u32,
        max_draws_observed: max_draws,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_hash_collision_birthday")?;

    // Birthday bound for n=365 days: √(π·365/2) ≈ 23.9.
    println!("space=365: {:?}", simulate(365, 1000, 42));
    println!("space=10000: {:?}", simulate(10_000, 500, 42));
    println!("invalid: {:?}", simulate(5, 100, 42));
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
    fn invalid_too_small_space() {
        assert_eq!(simulate(5, 100, 42), BirthdayVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_trials() {
        assert_eq!(simulate(100, 50, 42), BirthdayVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 200, 42);
        let b = simulate(100, 200, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn mean_at_least_two() {
        // Need at least 2 draws to collide.
        let v = simulate(100, 200, 42);
        if let BirthdayVerdict::Ok {
            mean_draws_to_collision,
            ..
        } = v
        {
            assert!(mean_draws_to_collision >= 2);
        }
    }

    #[test]
    fn larger_space_more_draws() {
        let small = simulate(100, 500, 42);
        let large = simulate(10_000, 500, 42);
        if let (
            BirthdayVerdict::Ok {
                mean_draws_to_collision: s,
                ..
            },
            BirthdayVerdict::Ok {
                mean_draws_to_collision: l,
                ..
            },
        ) = (small, large)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn birthday_bound_for_365() {
        // √(π·365/2) ≈ 23.94. Mean should be in same ballpark.
        let v = simulate(365, 5000, 42);
        if let BirthdayVerdict::Ok {
            mean_draws_to_collision,
            ..
        } = v
        {
            assert!((15..=40).contains(&mean_draws_to_collision));
        }
    }

    #[test]
    fn max_at_least_mean() {
        let v = simulate(100, 200, 42);
        if let BirthdayVerdict::Ok {
            mean_draws_to_collision,
            max_draws_observed,
        } = v
        {
            assert!(max_draws_observed >= mean_draws_to_collision);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(10, 100, 42);
        assert!(matches!(v, BirthdayVerdict::Ok { .. }));
    }

    #[test]
    fn many_trials_handled() {
        let v = simulate(100, 10_000, 42);
        assert!(matches!(v, BirthdayVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_means() {
        let a = simulate(100, 200, 42);
        let b = simulate(100, 200, 999);
        assert!(a != b);
    }

    #[test]
    fn max_le_space_plus_one() {
        let v = simulate(100, 200, 42);
        if let BirthdayVerdict::Ok {
            max_draws_observed, ..
        } = v
        {
            assert!(max_draws_observed <= 101);
        }
    }
}
