//! # apr diff --magnitude — Per-Tensor Magnitude Difference Classifier
//!
//! `apr diff --magnitude <A> <B>` reports per-tensor max-abs delta.
//! Tiers: < 1e-7 = numerically identical, < 1e-3 = drift, < 0.1 =
//! material change, ≥ 0.1 = significant divergence. This recipe
//! builds the classifier.
//!
//! Demonstrates the **DIFF.3** recipe for PMAT-118 (apr diff coverage —
//! closing F-invariant gap).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DIFF-001 + IEEE 754 epsilon thresholds
//!
//! Run with: cargo run --example cli_diff_magnitude_classifier
//!
//! Added by PMAT-118 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MagnitudeTier {
    Identical,
    NumericalDrift,
    MaterialChange,
    SignificantDivergence,
    InvalidDelta,
}

const IDENTICAL_THRESHOLD: f64 = 1e-7;
const DRIFT_THRESHOLD: f64 = 1e-3;
const MATERIAL_THRESHOLD: f64 = 0.1;

pub fn classify(max_abs_delta: f64) -> MagnitudeTier {
    if !max_abs_delta.is_finite() || max_abs_delta < 0.0 {
        return MagnitudeTier::InvalidDelta;
    }
    if max_abs_delta < IDENTICAL_THRESHOLD {
        MagnitudeTier::Identical
    } else if max_abs_delta < DRIFT_THRESHOLD {
        MagnitudeTier::NumericalDrift
    } else if max_abs_delta < MATERIAL_THRESHOLD {
        MagnitudeTier::MaterialChange
    } else {
        MagnitudeTier::SignificantDivergence
    }
}

pub fn max_abs_delta(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    let mut max = 0.0f64;
    for (a, b) in left.iter().zip(right) {
        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        let d = (a - b).abs();
        if d > max {
            max = d;
        }
    }
    Some(max)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_diff_magnitude_classifier")?;

    let cases = [0.0, 1e-9, 5e-5, 0.05, 0.5, -0.1, f64::NAN];
    for d in cases {
        println!("Δ={d:>10.2e}  →  {:?}", classify(d));
    }

    let left = [1.0, 2.0, 3.0];
    let right = [1.0001, 2.0001, 3.0001];
    println!(
        "max delta: {:?}  →  {:?}",
        max_abs_delta(&left, &right),
        classify(max_abs_delta(&left, &right).unwrap())
    );
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
    fn under_eps_identical() {
        assert_eq!(classify(0.0), MagnitudeTier::Identical);
        assert_eq!(classify(1e-9), MagnitudeTier::Identical);
    }

    #[test]
    fn between_eps_and_milli_drift() {
        assert_eq!(classify(1e-7), MagnitudeTier::NumericalDrift);
        assert_eq!(classify(5e-5), MagnitudeTier::NumericalDrift);
    }

    #[test]
    fn between_milli_and_tenth_material() {
        assert_eq!(classify(1e-3), MagnitudeTier::MaterialChange);
        assert_eq!(classify(0.05), MagnitudeTier::MaterialChange);
    }

    #[test]
    fn over_tenth_significant() {
        assert_eq!(classify(0.1), MagnitudeTier::SignificantDivergence);
        assert_eq!(classify(10.0), MagnitudeTier::SignificantDivergence);
    }

    #[test]
    fn negative_invalid() {
        assert_eq!(classify(-0.1), MagnitudeTier::InvalidDelta);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(classify(f64::NAN), MagnitudeTier::InvalidDelta);
        assert_eq!(classify(f64::INFINITY), MagnitudeTier::InvalidDelta);
    }

    #[test]
    fn max_abs_delta_finds_largest() {
        let l = [1.0, 2.0, 3.0];
        let r = [1.1, 2.0, 3.5];
        let d = max_abs_delta(&l, &r).unwrap();
        assert!((d - 0.5).abs() < 1e-9);
    }

    #[test]
    fn max_abs_delta_mismatched_lengths_yields_none() {
        assert!(max_abs_delta(&[1.0], &[1.0, 2.0]).is_none());
    }

    #[test]
    fn max_abs_delta_with_nan_yields_none() {
        assert!(max_abs_delta(&[f64::NAN], &[1.0]).is_none());
    }

    #[test]
    fn max_abs_delta_empty_yields_none() {
        assert!(max_abs_delta(&[], &[]).is_none());
    }
}
