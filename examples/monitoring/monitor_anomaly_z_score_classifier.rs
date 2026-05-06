//! # Monitoring Anomaly Z-Score Classifier
//!
//! Stream of metric values with rolling mean μ and stddev σ. Z-score
//! z = (x − μ) / σ tiers: |z| ≤ 2 normal, 2 < |z| ≤ 3 elevated, > 3
//! anomalous (3σ rule). This recipe builds the classifier with explicit
//! σ-validity guard.
//!
//! Demonstrates the **MON.8** recipe for PMAT-124 (monitoring coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Pukelsheim, F. (1994). The three sigma rule. The American Statistician 48(2).
//!
//! Run with: cargo run --example monitor_anomaly_z_score_classifier
//!
//! Added by PMAT-124 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AnomalyTier {
    Normal,
    Elevated { z: f64 },
    Anomalous { z: f64 },
    InvalidStddev,
    InvalidValue,
}

const ELEVATED_THRESHOLD: f64 = 2.0;
const ANOMALOUS_THRESHOLD: f64 = 3.0;

pub fn z_score(value: f64, mean: f64, stddev: f64) -> Option<f64> {
    if !value.is_finite() || !mean.is_finite() || !stddev.is_finite() {
        return None;
    }
    if stddev <= 0.0 {
        return None;
    }
    Some((value - mean) / stddev)
}

pub fn classify(value: f64, mean: f64, stddev: f64) -> AnomalyTier {
    if !value.is_finite() || !mean.is_finite() {
        return AnomalyTier::InvalidValue;
    }
    let Some(z) = z_score(value, mean, stddev) else {
        return AnomalyTier::InvalidStddev;
    };
    let abs_z = z.abs();
    if abs_z <= ELEVATED_THRESHOLD {
        AnomalyTier::Normal
    } else if abs_z <= ANOMALOUS_THRESHOLD {
        AnomalyTier::Elevated { z }
    } else {
        AnomalyTier::Anomalous { z }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_anomaly_z_score_classifier")?;

    let mean = 100.0;
    let stddev = 5.0;
    for v in [98.0, 105.0, 110.0, 115.0, 200.0, f64::NAN] {
        println!("v={v}  →  {:?}", classify(v, mean, stddev));
    }
    println!("invalid σ=0: {:?}", classify(100.0, 100.0, 0.0));
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
    fn within_2sigma_normal() {
        // (105 - 100) / 5 = 1.0 → Normal.
        assert_eq!(classify(105.0, 100.0, 5.0), AnomalyTier::Normal);
    }

    #[test]
    fn between_2_and_3sigma_elevated() {
        // (113 - 100) / 5 = 2.6 → Elevated.
        let v = classify(113.0, 100.0, 5.0);
        assert!(matches!(v, AnomalyTier::Elevated { .. }));
    }

    #[test]
    fn over_3sigma_anomalous() {
        // (200 - 100) / 5 = 20 → Anomalous.
        let v = classify(200.0, 100.0, 5.0);
        assert!(matches!(v, AnomalyTier::Anomalous { .. }));
    }

    #[test]
    fn negative_z_classified_by_abs() {
        // (50 - 100) / 5 = -10 → Anomalous (|z| = 10).
        let v = classify(50.0, 100.0, 5.0);
        assert!(matches!(v, AnomalyTier::Anomalous { .. }));
    }

    #[test]
    fn at_2sigma_boundary_normal() {
        // (110 - 100) / 5 = 2.0 → Normal (inclusive).
        assert_eq!(classify(110.0, 100.0, 5.0), AnomalyTier::Normal);
    }

    #[test]
    fn at_3sigma_boundary_elevated() {
        // (115 - 100) / 5 = 3.0 → Elevated (inclusive).
        let v = classify(115.0, 100.0, 5.0);
        assert!(matches!(v, AnomalyTier::Elevated { .. }));
    }

    #[test]
    fn zero_stddev_invalid() {
        assert_eq!(classify(100.0, 100.0, 0.0), AnomalyTier::InvalidStddev);
    }

    #[test]
    fn negative_stddev_invalid() {
        assert_eq!(classify(100.0, 100.0, -1.0), AnomalyTier::InvalidStddev);
    }

    #[test]
    fn nan_value_invalid() {
        assert_eq!(classify(f64::NAN, 100.0, 5.0), AnomalyTier::InvalidValue);
    }

    #[test]
    fn z_score_basic_math() {
        let z = z_score(110.0, 100.0, 5.0).unwrap();
        assert!((z - 2.0).abs() < 1e-12);
    }
}
