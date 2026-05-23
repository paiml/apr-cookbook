//! # CGP Baseline-Diff Classifier
//!
//! Continuous Gradient Performance (CGP) compares a current run vs a
//! baseline. Tiers: Improvement (≥ 5% faster), NoChange (-1% to +5%),
//! Regression (1-10% slower), SevereRegression (> 10% slower). This
//! recipe builds the classifier with explicit lower-is-better semantics
//! (lower latency = better).
//!
//! Demonstrates the **CGP.4** recipe for PMAT-128 (cgp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CGP-001.
//!
//! Run with: cargo run --example cgp_baseline_diff_classifier
//!
//! Added by PMAT-128 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DiffTier {
    Improvement { pct: f64 },
    NoChange,
    Regression { pct: f64 },
    SevereRegression { pct: f64 },
    InvalidBaseline,
}

const IMPROVEMENT_THRESHOLD: f64 = -0.05;
const NOISE_THRESHOLD: f64 = 0.01;
const SEVERE_THRESHOLD: f64 = 0.10;

pub fn classify(baseline: f64, current: f64) -> DiffTier {
    if !baseline.is_finite() || !current.is_finite() || baseline <= 0.0 {
        return DiffTier::InvalidBaseline;
    }
    let pct = (current - baseline) / baseline;
    if pct <= IMPROVEMENT_THRESHOLD {
        DiffTier::Improvement {
            pct: pct.abs() * 100.0,
        }
    } else if pct < NOISE_THRESHOLD {
        DiffTier::NoChange
    } else if pct < SEVERE_THRESHOLD {
        DiffTier::Regression { pct: pct * 100.0 }
    } else {
        DiffTier::SevereRegression { pct: pct * 100.0 }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cgp_baseline_diff_classifier")?;

    let baseline = 100.0;
    for current in [80.0, 99.0, 105.0, 130.0, 0.0] {
        println!(
            "{baseline} → {current}  =  {:?}",
            classify(baseline, current)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn faster_classified_improvement() {
        // 100 → 80 = 20% faster.
        let v = classify(100.0, 80.0);
        assert!(matches!(v, DiffTier::Improvement { .. }));
    }

    #[test]
    fn within_noise_no_change() {
        // 100 → 100.5 = 0.5% slower (within 1% noise).
        assert_eq!(classify(100.0, 100.5), DiffTier::NoChange);
    }

    #[test]
    fn small_regression_classified() {
        // 100 → 105 = 5% slower (regression, but not severe).
        let v = classify(100.0, 105.0);
        assert!(matches!(v, DiffTier::Regression { .. }));
    }

    #[test]
    fn severe_regression_classified() {
        // 100 → 130 = 30% slower.
        let v = classify(100.0, 130.0);
        assert!(matches!(v, DiffTier::SevereRegression { .. }));
    }

    #[test]
    fn boundary_at_5pct_faster_improvement() {
        // 100 → 95 = exactly 5% faster.
        let v = classify(100.0, 95.0);
        assert!(matches!(v, DiffTier::Improvement { .. }));
    }

    #[test]
    fn boundary_at_10pct_severe() {
        // 100 → 110 = 10% slower → severe (≥ threshold).
        let v = classify(100.0, 110.0);
        assert!(matches!(v, DiffTier::SevereRegression { .. }));
    }

    #[test]
    fn zero_baseline_invalid() {
        assert_eq!(classify(0.0, 100.0), DiffTier::InvalidBaseline);
    }

    #[test]
    fn negative_baseline_invalid() {
        assert_eq!(classify(-1.0, 100.0), DiffTier::InvalidBaseline);
    }

    #[test]
    fn nan_inputs_invalid() {
        assert_eq!(classify(f64::NAN, 100.0), DiffTier::InvalidBaseline);
        assert_eq!(classify(100.0, f64::NAN), DiffTier::InvalidBaseline);
    }

    #[test]
    fn improvement_pct_is_positive_magnitude() {
        if let DiffTier::Improvement { pct } = classify(100.0, 80.0) {
            assert!((pct - 20.0).abs() < 1e-9);
        }
    }
}
