//! # Monitoring P50 Latency Drift
//!
//! Detect drift in p50 latency vs a moving baseline. Drift > 20% from
//! baseline = degradation; drift < -20% = improvement (or model
//! caching). Numeric drift_pct returned for graphing.
//!
//! Demonstrates the **MON.49** recipe for PMAT-159 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Drift detection in SLO monitoring (Google SRE workbook).
//!
//! Run with: cargo run --example monitor_p50_drift
//!
//! Added by PMAT-159 (catalog 1054→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DriftVerdict {
    Stable { drift_pct: f64 },
    Degradation { drift_pct: f64 },
    Improvement { drift_pct: f64 },
    InvalidInput,
}

pub fn check(current_p50_ms: f64, baseline_p50_ms: f64) -> DriftVerdict {
    if !current_p50_ms.is_finite()
        || !baseline_p50_ms.is_finite()
        || current_p50_ms < 0.0
        || baseline_p50_ms <= 0.0
    {
        return DriftVerdict::InvalidInput;
    }
    let drift_pct = ((current_p50_ms - baseline_p50_ms) / baseline_p50_ms) * 100.0;
    if drift_pct > 20.0 {
        DriftVerdict::Degradation { drift_pct }
    } else if drift_pct < -20.0 {
        DriftVerdict::Improvement { drift_pct }
    } else {
        DriftVerdict::Stable { drift_pct }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_p50_drift")?;

    println!("stable: {:?}", check(105.0, 100.0));
    println!("degradation: {:?}", check(150.0, 100.0));
    println!("improvement: {:?}", check(70.0, 100.0));
    println!("invalid: {:?}", check(-1.0, 100.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn stable_within_20_pct() {
        let v = check(105.0, 100.0);
        assert!(matches!(v, DriftVerdict::Stable { .. }));
    }

    #[test]
    fn degradation_above_20() {
        let v = check(150.0, 100.0);
        assert!(matches!(v, DriftVerdict::Degradation { .. }));
    }

    #[test]
    fn improvement_below_neg_20() {
        let v = check(70.0, 100.0);
        assert!(matches!(v, DriftVerdict::Improvement { .. }));
    }

    #[test]
    fn boundary_at_20_pct_stable() {
        let v = check(120.0, 100.0);
        assert!(matches!(v, DriftVerdict::Stable { .. }));
    }

    #[test]
    fn boundary_at_neg_20_pct_stable() {
        let v = check(80.0, 100.0);
        assert!(matches!(v, DriftVerdict::Stable { .. }));
    }

    #[test]
    fn just_above_20_pct_degradation() {
        let v = check(120.1, 100.0);
        assert!(matches!(v, DriftVerdict::Degradation { .. }));
    }

    #[test]
    fn negative_current_invalid() {
        assert_eq!(check(-1.0, 100.0), DriftVerdict::InvalidInput);
    }

    #[test]
    fn zero_baseline_invalid() {
        assert_eq!(check(50.0, 0.0), DriftVerdict::InvalidInput);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(check(f64::NAN, 100.0), DriftVerdict::InvalidInput);
    }

    #[test]
    fn drift_pct_correct() {
        let v = check(150.0, 100.0);
        if let DriftVerdict::Degradation { drift_pct } = v {
            assert!((drift_pct - 50.0).abs() < 1e-9);
        }
    }

    #[test]
    fn deterministic() {
        let a = check(150.0, 100.0);
        let b = check(150.0, 100.0);
        assert_eq!(a, b);
    }
}
