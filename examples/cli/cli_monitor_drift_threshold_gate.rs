//! # apr monitor --drift-threshold — KL Divergence Drift Gate
//!
//! Distribution drift over time: KL(P_baseline ‖ P_current) crossing a
//! threshold triggers an alert. Tiers: < 0.1 = none, 0.1-0.5 = mild,
//! 0.5-1.0 = significant, > 1.0 = critical. This recipe builds the
//! gate (also handles invalid baseline = empty).
//!
//! Demonstrates the **MON.5** recipe for PMAT-114 (apr monitor coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender MON-001 + Kullback & Leibler 1951
//!
//! Run with: cargo run --example cli_monitor_drift_threshold_gate
//!
//! Added by PMAT-114 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DriftTier {
    None,
    Mild,
    Significant,
    Critical,
    InvalidKl,
}

pub fn classify(kl: f64) -> DriftTier {
    if !kl.is_finite() || kl < 0.0 {
        return DriftTier::InvalidKl;
    }
    if kl < 0.1 {
        DriftTier::None
    } else if kl < 0.5 {
        DriftTier::Mild
    } else if kl < 1.0 {
        DriftTier::Significant
    } else {
        DriftTier::Critical
    }
}

#[derive(Debug, PartialEq)]
pub enum GateVerdict {
    Pass,
    Fail { observed: f64, threshold: f64 },
    InvalidThreshold,
    InvalidKl,
}

pub fn gate(observed_kl: f64, threshold: f64) -> GateVerdict {
    if !observed_kl.is_finite() || observed_kl < 0.0 {
        return GateVerdict::InvalidKl;
    }
    if !threshold.is_finite() || threshold < 0.0 {
        return GateVerdict::InvalidThreshold;
    }
    if observed_kl <= threshold {
        GateVerdict::Pass
    } else {
        GateVerdict::Fail {
            observed: observed_kl,
            threshold,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_monitor_drift_threshold_gate")?;

    for kl in [0.0, 0.05, 0.3, 0.7, 1.5, -0.1, f64::NAN] {
        println!(
            "KL={kl:>5.2}  →  tier={:?}  gate@0.5={:?}",
            classify(kl),
            gate(kl, 0.5)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn under_0_1_none() {
        assert_eq!(classify(0.0), DriftTier::None);
        assert_eq!(classify(0.05), DriftTier::None);
    }

    #[test]
    fn point_one_to_half_mild() {
        assert_eq!(classify(0.1), DriftTier::Mild);
        assert_eq!(classify(0.49), DriftTier::Mild);
    }

    #[test]
    fn half_to_one_significant() {
        assert_eq!(classify(0.5), DriftTier::Significant);
        assert_eq!(classify(0.99), DriftTier::Significant);
    }

    #[test]
    fn over_one_critical() {
        assert_eq!(classify(1.0), DriftTier::Critical);
        assert_eq!(classify(10.0), DriftTier::Critical);
    }

    #[test]
    fn negative_or_nan_invalid() {
        // KL is non-negative by construction.
        assert_eq!(classify(-0.1), DriftTier::InvalidKl);
        assert_eq!(classify(f64::NAN), DriftTier::InvalidKl);
        assert_eq!(classify(f64::INFINITY), DriftTier::InvalidKl);
    }

    #[test]
    fn gate_at_threshold_passes() {
        // ≤ threshold is Pass (inclusive).
        assert_eq!(gate(0.5, 0.5), GateVerdict::Pass);
    }

    #[test]
    fn gate_above_threshold_fails() {
        let v = gate(0.6, 0.5);
        assert!(matches!(v, GateVerdict::Fail { .. }));
    }

    #[test]
    fn gate_invalid_kl_returns_invalid_kl() {
        assert_eq!(gate(-0.1, 0.5), GateVerdict::InvalidKl);
    }

    #[test]
    fn gate_invalid_threshold_returns_invalid_threshold() {
        assert_eq!(gate(0.1, -0.5), GateVerdict::InvalidThreshold);
    }
}
