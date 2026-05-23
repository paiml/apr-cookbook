//! # Monte-Carlo Dependency-Graph Dropout
//!
//! Sim end-to-end success when each step in a sequential pipeline
//! has independent failure probability `step_failure_prob`. Reports
//! end-to-end success rate.
//!
//! Demonstrates the **MC.81** recipe for PMAT-186 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: reliability theory (Birnbaum 1969); R. Barlow,
//!  Statistical Theory of Reliability §3.
//!
//! Run with: cargo run --example mc_dropout_dependency
//!
//! Added by PMAT-186 (catalog 1297→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DropoutVerdict {
    Ok {
        end_to_end_success_rate: f64,
        avg_steps_completed: f64,
    },
    InvalidConfig,
}

pub fn simulate(trials: u32, steps: u32, step_failure_prob: f64, seed: u64) -> DropoutVerdict {
    if trials == 0 || steps == 0 || !(0.0..=1.0).contains(&step_failure_prob) {
        return DropoutVerdict::InvalidConfig;
    }
    let mut successes = 0u32;
    let mut total_completed: u64 = 0;
    let mut rng_state = seed | 1;
    for _ in 0..trials {
        let mut completed = 0u32;
        let mut alive = true;
        for _ in 0..steps {
            if !alive {
                break;
            }
            let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
            if r < step_failure_prob {
                alive = false;
            } else {
                completed += 1;
            }
        }
        if alive {
            successes += 1;
        }
        total_completed += u64::from(completed);
    }
    DropoutVerdict::Ok {
        end_to_end_success_rate: f64::from(successes) / f64::from(trials),
        avg_steps_completed: total_completed as f64 / f64::from(trials),
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_dropout_dependency")?;

    println!("99% reliable: {:?}", simulate(2000, 10, 0.01, 42));
    println!("flaky: {:?}", simulate(2000, 10, 0.3, 42));
    println!("invalid: {:?}", simulate(0, 10, 0.5, 42));
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
    fn high_reliability_high_success() {
        // 10 steps × 99% = 0.99^10 ≈ 0.904.
        let v = simulate(2000, 10, 0.01, 42);
        if let DropoutVerdict::Ok {
            end_to_end_success_rate,
            ..
        } = v
        {
            assert!(end_to_end_success_rate > 0.85);
        }
    }

    #[test]
    fn many_steps_low_success() {
        // 20 steps × 90% = 0.9^20 ≈ 0.12.
        let v = simulate(2000, 20, 0.1, 42);
        if let DropoutVerdict::Ok {
            end_to_end_success_rate,
            ..
        } = v
        {
            assert!(end_to_end_success_rate < 0.20);
        }
    }

    #[test]
    fn invalid_zero_trials() {
        assert_eq!(simulate(0, 10, 0.5, 42), DropoutVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_steps() {
        assert_eq!(simulate(100, 0, 0.5, 42), DropoutVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_prob_out_of_range() {
        assert_eq!(simulate(100, 10, 1.5, 42), DropoutVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 10, 0.1, 42);
        let b = simulate(500, 10, 0.1, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn rate_in_unit_range() {
        let v = simulate(500, 10, 0.5, 42);
        if let DropoutVerdict::Ok {
            end_to_end_success_rate,
            ..
        } = v
        {
            assert!((0.0..=1.0).contains(&end_to_end_success_rate));
        }
    }

    #[test]
    fn always_failing_zero_success() {
        let v = simulate(100, 10, 1.0, 42);
        if let DropoutVerdict::Ok {
            end_to_end_success_rate,
            ..
        } = v
        {
            assert_eq!(end_to_end_success_rate, 0.0);
        }
    }

    #[test]
    fn never_failing_full_success() {
        let v = simulate(100, 10, 0.0, 42);
        if let DropoutVerdict::Ok {
            end_to_end_success_rate,
            ..
        } = v
        {
            assert_eq!(end_to_end_success_rate, 1.0);
        }
    }

    #[test]
    fn avg_steps_le_total_steps() {
        let v = simulate(100, 10, 0.5, 42);
        if let DropoutVerdict::Ok {
            avg_steps_completed,
            ..
        } = v
        {
            assert!(avg_steps_completed <= 10.0);
        }
    }

    #[test]
    fn higher_failure_lower_avg() {
        let lo = simulate(2000, 10, 0.05, 42);
        let hi = simulate(2000, 10, 0.5, 42);
        if let (
            DropoutVerdict::Ok {
                avg_steps_completed: l,
                ..
            },
            DropoutVerdict::Ok {
                avg_steps_completed: h,
                ..
            },
        ) = (lo, hi)
        {
            assert!(l > h);
        }
    }
}
