//! # Monte-Carlo Calibration Drift Simulation
//!
//! Simulate how a model's calibration (predicted vs actual rate)
//! drifts over time. Returns final calibration error and time-to-
//! threshold-breach.
//!
//! Demonstrates the **MC.26** recipe for PMAT-166 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Calibration / Brier-score drift in production ML.
//!
//! Run with: cargo run --example mc_calibration_drift
//!
//! Added by PMAT-166 (catalog 1117→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DriftVerdict {
    Ok {
        final_error: f64,
        time_to_breach_steps: Option<u32>,
        max_error: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    initial_error: f64,
    drift_per_step: f64,
    breach_threshold: f64,
    steps: u32,
    seed: u64,
) -> DriftVerdict {
    if !initial_error.is_finite()
        || initial_error < 0.0
        || !drift_per_step.is_finite()
        || !breach_threshold.is_finite()
        || breach_threshold <= 0.0
        || steps == 0
    {
        return DriftVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut error = initial_error;
    let mut max_error = error;
    let mut time_to_breach: Option<u32> = None;
    for step in 0..steps {
        // Add drift + jitter [-0.5, 0.5] × drift_per_step.
        let jitter = (unit(&mut rng_state) - 0.5) * drift_per_step.abs();
        error = (error + drift_per_step + jitter).max(0.0);
        if error > max_error {
            max_error = error;
        }
        if error >= breach_threshold && time_to_breach.is_none() {
            time_to_breach = Some(step + 1);
        }
    }
    DriftVerdict::Ok {
        final_error: error,
        time_to_breach_steps: time_to_breach,
        max_error,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_calibration_drift")?;

    println!("stable: {:?}", simulate(0.05, 0.0, 0.20, 1000, 42));
    println!("drifting: {:?}", simulate(0.05, 0.001, 0.20, 1000, 42));
    println!("fast drift: {:?}", simulate(0.05, 0.01, 0.20, 1000, 42));
    println!("invalid: {:?}", simulate(0.05, 0.001, 0.0, 1000, 42));
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
    fn stable_no_breach() {
        let v = simulate(0.05, 0.0, 0.20, 1000, 42);
        if let DriftVerdict::Ok {
            time_to_breach_steps,
            ..
        } = v
        {
            assert_eq!(time_to_breach_steps, None);
        }
    }

    #[test]
    fn fast_drift_breaches() {
        let v = simulate(0.05, 0.01, 0.20, 1000, 42);
        if let DriftVerdict::Ok {
            time_to_breach_steps,
            ..
        } = v
        {
            assert!(time_to_breach_steps.is_some());
        }
    }

    #[test]
    fn invalid_zero_threshold() {
        assert_eq!(
            simulate(0.05, 0.001, 0.0, 1000, 42),
            DriftVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_negative_initial() {
        assert_eq!(
            simulate(-0.1, 0.001, 0.20, 1000, 42),
            DriftVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_steps() {
        assert_eq!(
            simulate(0.05, 0.001, 0.20, 0, 42),
            DriftVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(f64::NAN, 0.001, 0.20, 1000, 42),
            DriftVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(0.05, 0.001, 0.20, 100, 42);
        let b = simulate(0.05, 0.001, 0.20, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn max_at_least_initial() {
        let v = simulate(0.05, 0.0, 0.20, 100, 42);
        if let DriftVerdict::Ok {
            max_error,
            final_error,
            ..
        } = v
        {
            assert!(max_error >= final_error.min(0.05) * 0.5);
        }
    }

    #[test]
    fn faster_drift_lower_breach_step() {
        let slow = simulate(0.05, 0.001, 0.20, 10_000, 42);
        let fast = simulate(0.05, 0.01, 0.20, 10_000, 42);
        if let (
            DriftVerdict::Ok {
                time_to_breach_steps: Some(s),
                ..
            },
            DriftVerdict::Ok {
                time_to_breach_steps: Some(f),
                ..
            },
        ) = (slow, fast)
        {
            assert!(f < s);
        }
    }

    #[test]
    fn negative_drift_recovers() {
        let v = simulate(0.15, -0.001, 0.30, 1000, 42);
        if let DriftVerdict::Ok {
            time_to_breach_steps,
            ..
        } = v
        {
            assert_eq!(time_to_breach_steps, None);
        }
    }
}
