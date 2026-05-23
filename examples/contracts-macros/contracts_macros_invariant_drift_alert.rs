//! # Contracts-Macros Invariant Drift Alert
//!
//! Watch a numerical invariant (e.g., total tokens emitted, sum of
//! probabilities) over a window. Alert if it drifts outside [target ±
//! tolerance × target]. Returns first violating sample index.
//!
//! Demonstrates the **CMM.37** recipe for PMAT-170 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: invariant-violation alerting (Hoare 1974 + SLO drift).
//!
//! Run with: cargo run --example contracts_macros_invariant_drift_alert
//!
//! Added by PMAT-170 (catalog 1153→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DriftVerdict {
    Stable,
    Drifted {
        sample_index: u32,
        observed: f64,
        target: f64,
    },
    EmptySamples,
    InvalidConfig,
}

pub fn detect(samples: &[f64], target: f64, tolerance_pct: f64) -> DriftVerdict {
    if samples.is_empty() {
        return DriftVerdict::EmptySamples;
    }
    if !target.is_finite() || target == 0.0 || !tolerance_pct.is_finite() || tolerance_pct < 0.0 {
        return DriftVerdict::InvalidConfig;
    }
    let bound = target.abs() * tolerance_pct / 100.0;
    let lo = target - bound;
    let hi = target + bound;
    for (i, &s) in samples.iter().enumerate() {
        if !s.is_finite() {
            return DriftVerdict::InvalidConfig;
        }
        if s < lo || s > hi {
            return DriftVerdict::Drifted {
                sample_index: i as u32,
                observed: s,
                target,
            };
        }
    }
    DriftVerdict::Stable
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_invariant_drift_alert")?;

    let stable = vec![1.0, 1.01, 0.99, 1.02];
    println!("stable: {:?}", detect(&stable, 1.0, 5.0));

    let drift = vec![1.0, 1.01, 1.5, 1.0];
    println!("drift: {:?}", detect(&drift, 1.0, 5.0));

    println!("invalid: {:?}", detect(&stable, 0.0, 5.0));
    println!("empty: {:?}", detect(&[], 1.0, 5.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn within_tolerance_stable() {
        assert_eq!(
            detect(&[1.0, 1.01, 0.99, 1.02], 1.0, 5.0),
            DriftVerdict::Stable
        );
    }

    #[test]
    fn first_drift_returned() {
        let v = detect(&[1.0, 1.5, 1.6, 0.5], 1.0, 5.0);
        if let DriftVerdict::Drifted { sample_index, .. } = v {
            assert_eq!(sample_index, 1);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(detect(&[], 1.0, 5.0), DriftVerdict::EmptySamples);
    }

    #[test]
    fn zero_target_invalid() {
        assert_eq!(detect(&[1.0], 0.0, 5.0), DriftVerdict::InvalidConfig);
    }

    #[test]
    fn negative_tolerance_invalid() {
        assert_eq!(detect(&[1.0], 1.0, -5.0), DriftVerdict::InvalidConfig);
    }

    #[test]
    fn nan_sample_invalid() {
        assert_eq!(detect(&[f64::NAN], 1.0, 5.0), DriftVerdict::InvalidConfig);
    }

    #[test]
    fn negative_target_works() {
        let v = detect(&[-1.0, -1.02, -0.98], -1.0, 5.0);
        assert_eq!(v, DriftVerdict::Stable);
    }

    #[test]
    fn at_boundary_stable() {
        let v = detect(&[1.05], 1.0, 5.0);
        assert_eq!(v, DriftVerdict::Stable);
    }

    #[test]
    fn just_over_boundary_drifts() {
        let v = detect(&[1.06], 1.0, 5.0);
        assert!(matches!(v, DriftVerdict::Drifted { .. }));
    }

    #[test]
    fn deterministic() {
        let s = vec![1.0, 1.01];
        let a = detect(&s, 1.0, 5.0);
        let b = detect(&s, 1.0, 5.0);
        assert_eq!(a, b);
    }
}
