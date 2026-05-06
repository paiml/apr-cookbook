//! # apr qa --regression — Per-Metric Delta Classifier
//!
//! When comparing a candidate vs baseline, deltas are classified by
//! relative magnitude: < 1% noise, 1-5% drift, 5-20% material, > 20%
//! regression. Sign matters: 5% improvement is good, 5% degradation
//! is bad. This recipe builds the classifier with direction awareness.
//!
//! Demonstrates the **QA.5** recipe for PMAT-121 (apr qa coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender QA-001 + Pedersen 2014 (regression analysis in CI)
//!
//! Run with: cargo run --example cli_qa_regression_delta_classifier
//!
//! Added by PMAT-121 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DeltaTier {
    Noise,
    Drift,
    MaterialRegression,
    SevereRegression,
    Improvement,
    InvalidBaseline,
}

const NOISE_PCT: f64 = 0.01;
const DRIFT_PCT: f64 = 0.05;
const MATERIAL_PCT: f64 = 0.20;

pub fn classify(baseline: f64, candidate: f64, lower_is_better: bool) -> DeltaTier {
    if !baseline.is_finite() || !candidate.is_finite() || baseline == 0.0 {
        return DeltaTier::InvalidBaseline;
    }
    let raw_delta = candidate - baseline;
    let signed = if lower_is_better {
        raw_delta
    } else {
        -raw_delta
    };
    let rel = signed / baseline.abs();
    if rel <= -NOISE_PCT {
        return DeltaTier::Improvement;
    }
    if rel.abs() < NOISE_PCT {
        DeltaTier::Noise
    } else if rel < DRIFT_PCT {
        DeltaTier::Drift
    } else if rel < MATERIAL_PCT {
        DeltaTier::MaterialRegression
    } else {
        DeltaTier::SevereRegression
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_qa_regression_delta_classifier")?;

    let cases = [
        ("loss equal", 1.0, 1.001, true), // noise
        ("loss + 3%", 1.0, 1.03, true),   // drift
        ("loss + 10%", 1.0, 1.10, true),  // material regression
        ("loss + 50%", 1.0, 1.50, true),  // severe
        ("loss − 5%", 1.0, 0.95, true),   // improvement
        ("acc + 5%", 0.80, 0.84, false),  // improvement (higher is better)
        ("acc − 30%", 0.80, 0.56, false), // severe regression
    ];
    for (label, b, c, lower) in cases {
        println!("{label:>14}  →  {:?}", classify(b, c, lower));
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
    fn within_1pct_noise() {
        // Loss 1.0 → 1.005: 0.5% drift (lower-is-better) → noise.
        assert_eq!(classify(1.0, 1.005, true), DeltaTier::Noise);
    }

    #[test]
    fn one_to_5_drift() {
        assert_eq!(classify(1.0, 1.03, true), DeltaTier::Drift);
    }

    #[test]
    fn five_to_20_material() {
        assert_eq!(classify(1.0, 1.10, true), DeltaTier::MaterialRegression);
    }

    #[test]
    fn over_20_severe() {
        assert_eq!(classify(1.0, 1.50, true), DeltaTier::SevereRegression);
    }

    #[test]
    fn improvement_for_lower_is_better() {
        // Loss going down → improvement.
        assert_eq!(classify(1.0, 0.90, true), DeltaTier::Improvement);
    }

    #[test]
    fn improvement_for_higher_is_better() {
        // Accuracy going up → improvement.
        assert_eq!(classify(0.80, 0.85, false), DeltaTier::Improvement);
    }

    #[test]
    fn higher_is_better_degradation_classified_correctly() {
        // Accuracy dropping 30% → severe regression.
        assert_eq!(classify(0.80, 0.56, false), DeltaTier::SevereRegression);
    }

    #[test]
    fn zero_baseline_invalid() {
        assert_eq!(classify(0.0, 1.0, true), DeltaTier::InvalidBaseline);
    }

    #[test]
    fn nan_baseline_invalid() {
        assert_eq!(classify(f64::NAN, 1.0, true), DeltaTier::InvalidBaseline);
        assert_eq!(classify(1.0, f64::NAN, true), DeltaTier::InvalidBaseline);
    }

    #[test]
    fn boundary_at_drift_classified_correctly() {
        // Exactly at 5% boundary: 0.05 → MaterialRegression (>=).
        // 1.0 → 1.05 = +5% which equals DRIFT_PCT — exclusive lower bound for material.
        assert_eq!(classify(1.0, 1.05, true), DeltaTier::MaterialRegression);
    }
}
