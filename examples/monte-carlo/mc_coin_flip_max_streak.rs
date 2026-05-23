//! # Monte-Carlo Coin Flip Max Streak
//!
//! Compute the longest consecutive run of heads in N fair coin flips,
//! averaged over T trials. Returns mean max-streak (×100 fixed) and
//! the maximum observed across trials.
//!
//! Demonstrates the **MC.150** recipe for PMAT-208 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Erdős & Révész, "On the length of the longest head-run"
//!  Topics in Information Theory (1975); Schilling 1990 expected
//!  max-streak ≈ log₂(N).
//!
//! Run with: cargo run --example mc_coin_flip_max_streak
//!
//! Added by PMAT-208 (catalog 1495→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum StreakVerdict {
    Ok {
        mean_max_streak_x100: u32,
        max_observed: u32,
    },
    InvalidConfig,
}

pub fn simulate(n_flips: u32, trials: u32, seed: u64) -> StreakVerdict {
    if n_flips < 10 || trials < 10 {
        return StreakVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut total_max: u64 = 0;
    let mut overall_max = 0u32;
    for _ in 0..trials {
        let mut current = 0u32;
        let mut trial_max = 0u32;
        for _ in 0..n_flips {
            let bit = lcg(&mut state) & 1;
            if bit == 1 {
                current += 1;
                if current > trial_max {
                    trial_max = current;
                }
            } else {
                current = 0;
            }
        }
        total_max += trial_max as u64;
        if trial_max > overall_max {
            overall_max = trial_max;
        }
    }
    let mean = (total_max as f64 / trials as f64) * 100.0;
    StreakVerdict::Ok {
        mean_max_streak_x100: mean as u32,
        max_observed: overall_max,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_coin_flip_max_streak")?;

    // E[max streak] ≈ log₂(100) ≈ 6.6
    println!("100 flips: {:?}", simulate(100, 1000, 42));
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
    fn invalid_too_few_flips() {
        assert_eq!(simulate(5, 100, 42), StreakVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_trials() {
        assert_eq!(simulate(100, 5, 42), StreakVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 100, 42);
        let b = simulate(100, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn mean_near_log2_n() {
        // E[max streak in 100 flips] ≈ log₂(100) ≈ 6.6 → 660.
        let v = simulate(100, 5000, 42);
        if let StreakVerdict::Ok {
            mean_max_streak_x100,
            ..
        } = v
        {
            assert!((400..=900).contains(&mean_max_streak_x100));
        }
    }

    #[test]
    fn max_observed_at_least_one() {
        let v = simulate(100, 100, 42);
        if let StreakVerdict::Ok { max_observed, .. } = v {
            assert!(max_observed >= 1);
        }
    }

    #[test]
    fn max_observed_le_n() {
        let v = simulate(100, 100, 42);
        if let StreakVerdict::Ok { max_observed, .. } = v {
            assert!(max_observed <= 100);
        }
    }

    #[test]
    fn longer_flips_longer_max() {
        let short = simulate(50, 1000, 42);
        let long = simulate(500, 1000, 42);
        if let (
            StreakVerdict::Ok {
                mean_max_streak_x100: s,
                ..
            },
            StreakVerdict::Ok {
                mean_max_streak_x100: l,
                ..
            },
        ) = (short, long)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = simulate(10, 10, 42);
        assert!(matches!(v, StreakVerdict::Ok { .. }));
    }

    #[test]
    fn many_trials_handled() {
        let v = simulate(100, 10_000, 42);
        assert!(matches!(v, StreakVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(100, 100, 42);
        let b = simulate(100, 100, 999);
        assert!(a != b);
    }

    #[test]
    fn mean_finite() {
        let v = simulate(100, 100, 42);
        if let StreakVerdict::Ok {
            mean_max_streak_x100,
            ..
        } = v
        {
            assert!(mean_max_streak_x100 < u32::MAX);
        }
    }
}
