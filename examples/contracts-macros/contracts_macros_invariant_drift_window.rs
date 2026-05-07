//! # Contracts-Macros Invariant Drift Window
//!
//! Track a rolling window of invariant violations; flag when the
//! recent-window violation rate exceeds the long-term baseline by
//! more than a threshold percentage. Returns the latest rate and
//! whether drift is detected.
//!
//! Demonstrates the **CMM.168** recipe for PMAT-213 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SPC control-chart Western Electric rules; SRE error-
//!  budget burn-rate alerts.
//!
//! Run with: cargo run --example contracts_macros_invariant_drift_window
//!
//! Added by PMAT-213 (catalog 1540→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DriftVerdict {
    Ok {
        recent_rate_x1000: u32,
        baseline_rate_x1000: u32,
        drift_detected: bool,
    },
    InvalidConfig,
}

pub fn detect(events: &[bool], window_size: u32, drift_threshold_pct: u32) -> DriftVerdict {
    if events.is_empty() || window_size == 0 || (window_size as usize) > events.len() {
        return DriftVerdict::InvalidConfig;
    }
    let n = events.len();
    let baseline_failures = events.iter().filter(|e| !**e).count() as u32;
    let baseline_rate = (baseline_failures as f64 / n as f64) * 1000.0;
    let recent_start = n - window_size as usize;
    let recent_failures = events[recent_start..].iter().filter(|e| !**e).count() as u32;
    let recent_rate = (recent_failures as f64 / window_size as f64) * 1000.0;
    let baseline_x1000 = baseline_rate as u32;
    let recent_x1000 = recent_rate as u32;
    let drift = if baseline_x1000 == 0 {
        recent_x1000 > drift_threshold_pct * 10
    } else {
        let increase_pct = ((recent_rate - baseline_rate) / baseline_rate * 100.0) as i32;
        increase_pct > drift_threshold_pct as i32
    };
    DriftVerdict::Ok {
        recent_rate_x1000: recent_x1000,
        baseline_rate_x1000: baseline_x1000,
        drift_detected: drift,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_invariant_drift_window")?;

    let stable: Vec<bool> = (0..100).map(|i| i % 10 != 0).collect();
    println!("stable: {:?}", detect(&stable, 20, 50));
    let drifted: Vec<bool> = (0..80)
        .map(|_| true)
        .chain((0..20).map(|_| false))
        .collect();
    println!("drifted: {:?}", detect(&drifted, 20, 50));
    println!("invalid: {:?}", detect(&[], 20, 50));
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
    fn no_failures_no_drift() {
        let events = vec![true; 100];
        let v = detect(&events, 20, 50);
        if let DriftVerdict::Ok { drift_detected, .. } = v {
            assert!(!drift_detected);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(detect(&[], 20, 50), DriftVerdict::InvalidConfig);
    }

    #[test]
    fn zero_window_rejected() {
        let events = vec![true; 10];
        assert_eq!(detect(&events, 0, 50), DriftVerdict::InvalidConfig);
    }

    #[test]
    fn window_over_total_rejected() {
        let events = vec![true; 10];
        assert_eq!(detect(&events, 100, 50), DriftVerdict::InvalidConfig);
    }

    #[test]
    fn recent_drift_detected() {
        let mut events: Vec<bool> = (0..80).map(|_| true).collect();
        events.extend((0..20).map(|_| false));
        let v = detect(&events, 20, 50);
        if let DriftVerdict::Ok { drift_detected, .. } = v {
            assert!(drift_detected);
        }
    }

    #[test]
    fn baseline_rate_correct() {
        let mut events: Vec<bool> = vec![true; 90];
        events.extend(vec![false; 10]);
        let v = detect(&events, 10, 50);
        if let DriftVerdict::Ok {
            baseline_rate_x1000,
            ..
        } = v
        {
            assert_eq!(baseline_rate_x1000, 100);
        }
    }

    #[test]
    fn recent_rate_correct() {
        let mut events: Vec<bool> = vec![true; 90];
        events.extend(vec![false; 10]);
        let v = detect(&events, 10, 50);
        if let DriftVerdict::Ok {
            recent_rate_x1000, ..
        } = v
        {
            assert_eq!(recent_rate_x1000, 1000);
        }
    }

    #[test]
    fn deterministic() {
        let events = vec![true; 100];
        let r1 = detect(&events, 20, 50);
        let r2 = detect(&events, 20, 50);
        assert_eq!(r1, r2);
    }

    #[test]
    fn many_events_handled() {
        let events = vec![true; 10_000];
        let v = detect(&events, 100, 50);
        assert!(matches!(v, DriftVerdict::Ok { .. }));
    }

    #[test]
    fn min_input_accepted() {
        let v = detect(&[true], 1, 50);
        assert!(matches!(v, DriftVerdict::Ok { .. }));
    }

    #[test]
    fn equal_recent_baseline_no_drift() {
        // 50% baseline, 50% recent → no drift.
        let events: Vec<bool> = (0..100).map(|i| i % 2 == 0).collect();
        let v = detect(&events, 20, 50);
        if let DriftVerdict::Ok { drift_detected, .. } = v {
            assert!(!drift_detected);
        }
    }
}
