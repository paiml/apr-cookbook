//! # Monitoring Seasonal-Decomposition Anomaly Detector
//!
//! STL decomposition splits a time series into:
//!   trend (slow drift) + seasonal (periodic) + residual
//!
//! Anomaly = |residual_t| > N × σ(residual). This recipe approximates
//! seasonal as the mean of period_lag offsets and computes residual
//! z-score.
//!
//! Demonstrates the **MON.30** recipe for PMAT-147 (monitoring round 6).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Cleveland et al. (1990). STL: Seasonal-Trend Decomposition.
//!
//! Run with: cargo run --example monitor_seasonal_decompose
//!
//! Added by PMAT-147 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const ANOMALY_Z_THRESHOLD: f64 = 3.0;

#[derive(Debug, PartialEq)]
pub enum DecomposeVerdict {
    Anomaly { z_score: f64 },
    Normal { z_score: f64 },
    InsufficientHistory,
    InvalidPeriod,
    NumericFailure,
}

pub fn check(history: &[f64], current: f64, period: usize) -> DecomposeVerdict {
    if period == 0 {
        return DecomposeVerdict::InvalidPeriod;
    }
    if history.len() < period * 2 {
        return DecomposeVerdict::InsufficientHistory;
    }
    if !current.is_finite() || history.iter().any(|x| !x.is_finite()) {
        return DecomposeVerdict::NumericFailure;
    }
    // Approximate seasonal: average of values at same phase.
    let phase = history.len() % period;
    let same_phase: Vec<f64> = history
        .iter()
        .enumerate()
        .filter(|(i, _)| i % period == phase)
        .map(|(_, v)| *v)
        .collect();
    if same_phase.is_empty() {
        return DecomposeVerdict::InsufficientHistory;
    }
    let seasonal_mean: f64 = same_phase.iter().sum::<f64>() / same_phase.len() as f64;
    let trend: f64 = history.iter().sum::<f64>() / history.len() as f64;
    let residuals: Vec<f64> = history
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let p = i % period;
            let phase_mean = if p == phase {
                seasonal_mean
            } else {
                let same: Vec<f64> = history
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| j % period == p)
                    .map(|(_, x)| *x)
                    .collect();
                same.iter().sum::<f64>() / same.len() as f64
            };
            v - trend - (phase_mean - trend)
        })
        .collect();
    let mean_resid: f64 = residuals.iter().sum::<f64>() / residuals.len() as f64;
    let var: f64 = residuals
        .iter()
        .map(|r| (r - mean_resid).powi(2))
        .sum::<f64>()
        / residuals.len() as f64;
    let sigma = var.sqrt().max(1e-9);
    let current_residual = current - trend - (seasonal_mean - trend);
    let z = (current_residual - mean_resid) / sigma;
    if z.abs() >= ANOMALY_Z_THRESHOLD {
        DecomposeVerdict::Anomaly { z_score: z }
    } else {
        DecomposeVerdict::Normal { z_score: z }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_seasonal_decompose")?;

    // Daily pattern: low at hours 0-6, high at hours 12-18.
    let pattern = [10.0, 10.0, 50.0, 50.0, 80.0, 80.0, 50.0, 50.0];
    let mut history = Vec::new();
    for _ in 0..3 {
        history.extend_from_slice(&pattern);
    }

    println!("normal continuation: {:?}", check(&history, 10.0, 8));
    println!("anomaly: {:?}", check(&history, 200.0, 8));
    println!("insufficient: {:?}", check(&[1.0, 2.0], 5.0, 8));
    println!("invalid period: {:?}", check(&history, 5.0, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn periodic_history() -> Vec<f64> {
        let pattern = [10.0, 50.0, 80.0, 50.0];
        let mut history = Vec::new();
        for _ in 0..5 {
            history.extend_from_slice(&pattern);
        }
        history
    }

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn normal_continuation_no_anomaly() {
        let history = periodic_history();
        // 20 entries, period 4. Phase = 20 % 4 = 0 → expects ~10.
        let v = check(&history, 10.0, 4);
        assert!(matches!(v, DecomposeVerdict::Normal { .. }));
    }

    #[test]
    fn anomaly_detected_for_outlier() {
        let history = periodic_history();
        let v = check(&history, 500.0, 4);
        assert!(matches!(v, DecomposeVerdict::Anomaly { .. }));
    }

    #[test]
    fn insufficient_history_rejected() {
        let v = check(&[1.0, 2.0, 3.0], 5.0, 4);
        assert_eq!(v, DecomposeVerdict::InsufficientHistory);
    }

    #[test]
    fn invalid_period_rejected() {
        let v = check(&periodic_history(), 5.0, 0);
        assert_eq!(v, DecomposeVerdict::InvalidPeriod);
    }

    #[test]
    fn nan_current_rejected() {
        let v = check(&periodic_history(), f64::NAN, 4);
        assert_eq!(v, DecomposeVerdict::NumericFailure);
    }

    #[test]
    fn nan_history_rejected() {
        let mut h = periodic_history();
        h[0] = f64::NAN;
        let v = check(&h, 5.0, 4);
        assert_eq!(v, DecomposeVerdict::NumericFailure);
    }

    #[test]
    fn z_score_returned() {
        let v = check(&periodic_history(), 10.0, 4);
        if let DecomposeVerdict::Normal { z_score } = v {
            assert!(z_score.is_finite());
        }
    }

    #[test]
    fn anomaly_threshold_at_3_sigma() {
        // Manually construct: history with low variance + extreme current.
        let mut h: Vec<f64> = (0..40).map(|_| 100.0).collect();
        h[0] = 100.5; // tiny noise.
        let v = check(&h, 100.0 + 50.0, 4);
        // Current is way off → high z.
        assert!(matches!(v, DecomposeVerdict::Anomaly { .. }));
    }

    #[test]
    fn at_minimum_history_works() {
        // Period 4 needs ≥ 8 entries.
        let h = vec![10.0, 50.0, 80.0, 50.0, 10.0, 50.0, 80.0, 50.0];
        let v = check(&h, 10.0, 4);
        assert!(matches!(
            v,
            DecomposeVerdict::Normal { .. } | DecomposeVerdict::Anomaly { .. }
        ));
    }

    #[test]
    fn just_below_minimum_insufficient() {
        // Period 4 + only 7 entries → insufficient.
        let h = vec![10.0; 7];
        let v = check(&h, 10.0, 4);
        assert_eq!(v, DecomposeVerdict::InsufficientHistory);
    }
}
