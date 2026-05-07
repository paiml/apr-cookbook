//! # Monte-Carlo Gambler's Ruin
//!
//! Sim symmetric/biased random walk where a gambler with starting
//! capital plays until reaching either bankruptcy (0) or a target
//! fortune. Returns ruin probability across N trials.
//!
//! Demonstrates the **MC.133** recipe for PMAT-203 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Feller, An Introduction to Probability Theory, vol. 1,
//!  ch. XIV (1968) gambler's ruin.
//!
//! Run with: cargo run --example mc_gamblers_ruin
//!
//! Added by PMAT-203 (catalog 1450→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RuinVerdict {
    Ok {
        ruin_probability: f64,
        avg_steps: u32,
    },
    InvalidConfig,
}

pub fn simulate(start: u32, target: u32, win_prob: f64, trials: u32, seed: u64) -> RuinVerdict {
    if start == 0 || target <= start || trials == 0 || !(0.0..=1.0).contains(&win_prob) {
        return RuinVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut ruined = 0u32;
    let mut total_steps = 0u64;
    for _ in 0..trials {
        let mut capital = start;
        let mut steps = 0u32;
        while capital > 0 && capital < target {
            let r = (lcg(&mut state) >> 32) as f64 / (u32::MAX as f64);
            if r < win_prob {
                capital += 1;
            } else {
                capital -= 1;
            }
            steps += 1;
            if steps > 100_000 {
                break;
            }
        }
        if capital == 0 {
            ruined += 1;
        }
        total_steps += steps as u64;
    }
    RuinVerdict::Ok {
        ruin_probability: ruined as f64 / trials as f64,
        avg_steps: (total_steps / trials as u64) as u32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_gamblers_ruin")?;

    println!("symmetric: {:?}", simulate(10, 20, 0.5, 1000, 42));
    println!("invalid: {:?}", simulate(0, 20, 0.5, 1000, 42));
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
    fn symmetric_walk_half_ruin_in_middle() {
        // start=10, target=20, fair coin → ruin prob ≈ (target-start)/target = 0.5
        let v = simulate(10, 20, 0.5, 5000, 42);
        if let RuinVerdict::Ok {
            ruin_probability, ..
        } = v
        {
            assert!((0.4..=0.6).contains(&ruin_probability));
        }
    }

    #[test]
    fn invalid_zero_start() {
        assert_eq!(simulate(0, 10, 0.5, 100, 42), RuinVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_target_le_start() {
        assert_eq!(simulate(10, 5, 0.5, 100, 42), RuinVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_trials() {
        assert_eq!(simulate(10, 20, 0.5, 0, 42), RuinVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_prob_above_one() {
        assert_eq!(simulate(10, 20, 1.5, 100, 42), RuinVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(5, 10, 0.5, 100, 42);
        let b = simulate(5, 10, 0.5, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn ruin_prob_in_zero_one() {
        let v = simulate(5, 10, 0.5, 100, 42);
        if let RuinVerdict::Ok {
            ruin_probability, ..
        } = v
        {
            assert!((0.0..=1.0).contains(&ruin_probability));
        }
    }

    #[test]
    fn favorable_odds_lower_ruin() {
        // win_prob=0.7 strongly favors gambler → low ruin
        let v = simulate(10, 20, 0.7, 1000, 42);
        if let RuinVerdict::Ok {
            ruin_probability, ..
        } = v
        {
            assert!(ruin_probability < 0.3);
        }
    }

    #[test]
    fn unfavorable_odds_high_ruin() {
        let v = simulate(10, 20, 0.3, 1000, 42);
        if let RuinVerdict::Ok {
            ruin_probability, ..
        } = v
        {
            assert!(ruin_probability > 0.6);
        }
    }

    #[test]
    fn small_walk_handled() {
        let v = simulate(1, 2, 0.5, 100, 42);
        assert!(matches!(v, RuinVerdict::Ok { .. }));
    }

    #[test]
    fn avg_steps_positive() {
        let v = simulate(5, 10, 0.5, 100, 42);
        if let RuinVerdict::Ok { avg_steps, .. } = v {
            assert!(avg_steps > 0);
        }
    }
}
