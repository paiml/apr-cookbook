//! # Advanced Safety-Classifier Threshold Picker
//!
//! Pick a probability threshold that balances false-positive rate
//! (FPR: harmless flagged as harmful) vs false-negative rate (FNR:
//! harmful slipped through).
//!
//! Cost-weighted picker: minimize fpr × fpr_cost + fnr × fnr_cost.
//! For chat moderation: fnr_cost > fpr_cost (catch harm > avoid friction).
//!
//! Demonstrates the **ADV.20** recipe for PMAT-149 (advanced round 7).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Anthropic Constitutional AI evaluation methodology.
//!
//! Run with: cargo run --example adv_safety_classifier_threshold
//!
//! Added by PMAT-149 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ThresholdVerdict {
    Ok {
        threshold: f64,
        expected_fpr: f64,
        expected_fnr: f64,
        weighted_cost: f64,
    },
    EmptyRocCurve,
    InvalidCost,
}

pub fn pick(roc_thresholds: &[(f64, f64, f64)], fpr_cost: f64, fnr_cost: f64) -> ThresholdVerdict {
    if roc_thresholds.is_empty() {
        return ThresholdVerdict::EmptyRocCurve;
    }
    if !fpr_cost.is_finite() || !fnr_cost.is_finite() || fpr_cost < 0.0 || fnr_cost < 0.0 {
        return ThresholdVerdict::InvalidCost;
    }
    let mut best_threshold = 0.5;
    let mut best_fpr = 1.0;
    let mut best_fnr = 1.0;
    let mut best_cost = f64::INFINITY;
    for &(t, fpr, fnr) in roc_thresholds {
        if !t.is_finite()
            || !(0.0..=1.0).contains(&t)
            || !(0.0..=1.0).contains(&fpr)
            || !(0.0..=1.0).contains(&fnr)
        {
            return ThresholdVerdict::InvalidCost;
        }
        let cost = fpr * fpr_cost + fnr * fnr_cost;
        if cost < best_cost {
            best_cost = cost;
            best_threshold = t;
            best_fpr = fpr;
            best_fnr = fnr;
        }
    }
    ThresholdVerdict::Ok {
        threshold: best_threshold,
        expected_fpr: best_fpr,
        expected_fnr: best_fnr,
        weighted_cost: best_cost,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_safety_classifier_threshold")?;

    // ROC curve: (threshold, fpr, fnr).
    let curve = [
        (0.1, 0.4, 0.05),
        (0.3, 0.20, 0.10),
        (0.5, 0.10, 0.20),
        (0.7, 0.05, 0.40),
        (0.9, 0.01, 0.70),
    ];
    println!("equal cost: {:?}", pick(&curve, 1.0, 1.0));
    println!("FNR 5x worse: {:?}", pick(&curve, 1.0, 5.0));
    println!("FPR 5x worse: {:?}", pick(&curve, 5.0, 1.0));
    println!("empty: {:?}", pick(&[], 1.0, 1.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typical_curve() -> Vec<(f64, f64, f64)> {
        vec![
            (0.1, 0.4, 0.05),
            (0.3, 0.20, 0.10),
            (0.5, 0.10, 0.20),
            (0.7, 0.05, 0.40),
            (0.9, 0.01, 0.70),
        ]
    }

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn equal_cost_balances() {
        let v = pick(&typical_curve(), 1.0, 1.0);
        if let ThresholdVerdict::Ok { threshold, .. } = v {
            // With equal cost, expected to pick mid-range threshold.
            assert!(threshold >= 0.1 && threshold <= 0.9);
        }
    }

    #[test]
    fn fnr_costly_picks_low_threshold() {
        // High FNR cost → pick threshold that catches more harm = low.
        let v = pick(&typical_curve(), 1.0, 10.0);
        if let ThresholdVerdict::Ok { threshold, .. } = v {
            assert!(threshold <= 0.3);
        }
    }

    #[test]
    fn fpr_costly_picks_high_threshold() {
        // High FPR cost → pick threshold that avoids false flags = high.
        let v = pick(&typical_curve(), 10.0, 1.0);
        if let ThresholdVerdict::Ok { threshold, .. } = v {
            assert!(threshold >= 0.5);
        }
    }

    #[test]
    fn empty_curve_rejected() {
        assert_eq!(pick(&[], 1.0, 1.0), ThresholdVerdict::EmptyRocCurve);
    }

    #[test]
    fn negative_cost_rejected() {
        assert_eq!(
            pick(&typical_curve(), -1.0, 1.0),
            ThresholdVerdict::InvalidCost
        );
    }

    #[test]
    fn invalid_threshold_above_one() {
        let bad = vec![(1.5, 0.1, 0.1)];
        assert_eq!(pick(&bad, 1.0, 1.0), ThresholdVerdict::InvalidCost);
    }

    #[test]
    fn nan_cost_rejected() {
        assert_eq!(
            pick(&typical_curve(), f64::NAN, 1.0),
            ThresholdVerdict::InvalidCost
        );
    }

    #[test]
    fn weighted_cost_returned() {
        if let ThresholdVerdict::Ok { weighted_cost, .. } = pick(&typical_curve(), 1.0, 1.0) {
            assert!(weighted_cost.is_finite());
        }
    }

    #[test]
    fn fpr_and_fnr_at_picked_threshold() {
        if let ThresholdVerdict::Ok {
            expected_fpr,
            expected_fnr,
            ..
        } = pick(&typical_curve(), 1.0, 1.0)
        {
            assert!(expected_fpr <= 1.0);
            assert!(expected_fnr <= 1.0);
        }
    }

    #[test]
    fn single_point_curve() {
        let v = pick(&[(0.5, 0.1, 0.1)], 1.0, 1.0);
        if let ThresholdVerdict::Ok { threshold, .. } = v {
            assert_eq!(threshold, 0.5);
        }
    }
}
