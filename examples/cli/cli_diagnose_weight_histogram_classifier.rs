//! # apr diagnose --weight-histogram — Distribution Health Classifier
//!
//! Healthy weight distributions are roughly Gaussian with σ ∝ 1/√fan_in.
//! Tails too heavy → optimizer instability; mean far from 0 → broken
//! init or numerical drift; near-constant → dead layer (e.g., dying
//! ReLU). This recipe builds the classifier over (mean, std, max_abs).
//!
//! Demonstrates the **DIAG.6** recipe for PMAT-116 (apr diagnose coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DIAG-001 + Glorot & Bengio 2010 (Xavier init)
//!
//! Run with: cargo run --example cli_diagnose_weight_histogram_classifier
//!
//! Added by PMAT-116 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum HealthVerdict {
    Healthy,
    DeadLayer,                          // std near 0
    DriftedMean { mean: f64 },          // |mean| > std
    HeavyTails { kurtosis_proxy: f64 }, // max_abs / std > 6
    InvalidShape,
}

const DEAD_THRESHOLD: f64 = 1e-6;
const TAIL_RATIO_LIMIT: f64 = 6.0;

pub fn classify(mean: f64, std: f64, max_abs: f64) -> HealthVerdict {
    if !mean.is_finite() || !std.is_finite() || !max_abs.is_finite() || std < 0.0 || max_abs < 0.0 {
        return HealthVerdict::InvalidShape;
    }
    if std < DEAD_THRESHOLD {
        return HealthVerdict::DeadLayer;
    }
    if mean.abs() > std {
        return HealthVerdict::DriftedMean { mean };
    }
    let ratio = max_abs / std;
    if ratio > TAIL_RATIO_LIMIT {
        return HealthVerdict::HeavyTails {
            kurtosis_proxy: ratio,
        };
    }
    HealthVerdict::Healthy
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_diagnose_weight_histogram_classifier")?;

    let cases = [
        ("healthy", 0.0, 0.02, 0.08),
        ("dead", 0.0, 1e-9, 1e-9),
        ("drifted", 0.5, 0.1, 0.6),
        ("heavy tails", 0.0, 0.02, 0.5),
        ("invalid std<0", 0.0, -1.0, 0.5),
    ];
    for (label, m, s, ma) in cases {
        println!("{label:>14}  →  {:?}", classify(m, s, ma));
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
    fn typical_xavier_init_healthy() {
        // mean=0, std=0.02, max=0.08 (4σ), no drift.
        assert_eq!(classify(0.0, 0.02, 0.08), HealthVerdict::Healthy);
    }

    #[test]
    fn dead_layer_detected() {
        assert_eq!(classify(0.0, 1e-9, 1e-9), HealthVerdict::DeadLayer);
    }

    #[test]
    fn drifted_mean_detected() {
        // |mean|=0.5 > std=0.1 → drift.
        let v = classify(0.5, 0.1, 0.6);
        assert!(matches!(v, HealthVerdict::DriftedMean { .. }));
    }

    #[test]
    fn heavy_tails_detected() {
        // max_abs/std = 25 > 6 → heavy.
        let v = classify(0.0, 0.02, 0.5);
        assert!(matches!(v, HealthVerdict::HeavyTails { .. }));
    }

    #[test]
    fn at_tail_boundary_healthy() {
        // ratio = 6.0 exactly — strict >, so still Healthy.
        assert_eq!(classify(0.0, 0.1, 0.6), HealthVerdict::Healthy);
    }

    #[test]
    fn invalid_negative_std_rejected() {
        assert_eq!(classify(0.0, -0.1, 0.5), HealthVerdict::InvalidShape);
    }

    #[test]
    fn invalid_negative_max_abs_rejected() {
        assert_eq!(classify(0.0, 0.1, -0.5), HealthVerdict::InvalidShape);
    }

    #[test]
    fn nan_inputs_rejected() {
        assert_eq!(classify(f64::NAN, 0.1, 0.5), HealthVerdict::InvalidShape);
        assert_eq!(classify(0.0, f64::NAN, 0.5), HealthVerdict::InvalidShape);
        assert_eq!(
            classify(0.0, 0.1, f64::INFINITY),
            HealthVerdict::InvalidShape
        );
    }

    #[test]
    fn drift_takes_priority_over_tails() {
        // Both drift and heavy tails — drift fires first by check order.
        let v = classify(1.0, 0.1, 1.0);
        assert!(matches!(v, HealthVerdict::DriftedMean { .. }));
    }

    #[test]
    fn dead_layer_ignores_drift_check() {
        // std under threshold short-circuits before drift check.
        assert_eq!(classify(100.0, 1e-9, 1e-9), HealthVerdict::DeadLayer);
    }
}
