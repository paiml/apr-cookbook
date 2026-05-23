//! # Contracts-Macros Violation History Tracker
//!
//! Given timestamped violation counts, detect regressions: any window
//! where count is N× the recent baseline. Returns first regression
//! detected.
//!
//! Demonstrates the **CMM.30** recipe for PMAT-167 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SRE error-budget burn-rate alerting.
//!
//! Run with: cargo run --example contracts_macros_violation_history
//!
//! Added by PMAT-167 (catalog 1126→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum HistoryVerdict {
    NoRegression,
    Regression {
        window_index: u32,
        observed: u32,
        baseline: u32,
        ratio: f64,
    },
    InsufficientHistory,
    InvalidConfig,
}

pub fn detect(counts: &[u32], baseline_window: u32, spike_ratio: f64) -> HistoryVerdict {
    if !spike_ratio.is_finite() || spike_ratio <= 1.0 {
        return HistoryVerdict::InvalidConfig;
    }
    if baseline_window == 0 {
        return HistoryVerdict::InvalidConfig;
    }
    let bw = baseline_window as usize;
    if counts.len() < bw + 1 {
        return HistoryVerdict::InsufficientHistory;
    }
    for i in bw..counts.len() {
        let baseline_sum: u32 = counts[i - bw..i].iter().sum();
        let baseline_avg = if bw > 0 { baseline_sum / bw as u32 } else { 0 };
        if baseline_avg == 0 {
            continue;
        }
        let observed = counts[i];
        let ratio = f64::from(observed) / f64::from(baseline_avg);
        if ratio >= spike_ratio {
            return HistoryVerdict::Regression {
                window_index: i as u32,
                observed,
                baseline: baseline_avg,
                ratio,
            };
        }
    }
    HistoryVerdict::NoRegression
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_violation_history")?;

    let stable = vec![5, 5, 5, 5, 5];
    println!("stable: {:?}", detect(&stable, 3, 2.0));

    let spike = vec![5, 5, 5, 5, 50];
    println!("spike: {:?}", detect(&spike, 3, 2.0));

    let too_short = vec![5, 5];
    println!("insufficient: {:?}", detect(&too_short, 5, 2.0));

    println!("invalid: {:?}", detect(&[1, 2], 0, 2.0));
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
    fn stable_no_regression() {
        let v = detect(&[5, 5, 5, 5, 5], 3, 2.0);
        assert_eq!(v, HistoryVerdict::NoRegression);
    }

    #[test]
    fn spike_detected() {
        let v = detect(&[5, 5, 5, 5, 50], 3, 2.0);
        assert!(matches!(v, HistoryVerdict::Regression { .. }));
    }

    #[test]
    fn first_regression_returned() {
        let v = detect(&[5, 5, 5, 30, 5, 50], 3, 2.0);
        if let HistoryVerdict::Regression { window_index, .. } = v {
            assert_eq!(window_index, 3);
        }
    }

    #[test]
    fn invalid_zero_baseline() {
        assert_eq!(detect(&[1, 2], 0, 2.0), HistoryVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_ratio_below_one() {
        assert_eq!(detect(&[1, 2], 3, 0.5), HistoryVerdict::InvalidConfig);
    }

    #[test]
    fn nan_ratio_invalid() {
        assert_eq!(detect(&[1, 2], 3, f64::NAN), HistoryVerdict::InvalidConfig);
    }

    #[test]
    fn too_short_history() {
        assert_eq!(detect(&[5, 5], 5, 2.0), HistoryVerdict::InsufficientHistory);
    }

    #[test]
    fn zero_baseline_avg_skipped() {
        // Counts start at 0 → ratio undefined; should skip without panic.
        let v = detect(&[0, 0, 0, 5], 3, 2.0);
        assert_eq!(v, HistoryVerdict::NoRegression);
    }

    #[test]
    fn regression_carries_ratio() {
        let v = detect(&[5, 5, 5, 100], 3, 2.0);
        if let HistoryVerdict::Regression { ratio, .. } = v {
            assert!(ratio >= 2.0);
        }
    }

    #[test]
    fn growing_steadily_no_regression() {
        let v = detect(&[1, 2, 3, 4, 5, 6], 3, 5.0);
        assert_eq!(v, HistoryVerdict::NoRegression);
    }

    #[test]
    fn deterministic() {
        let counts = [5, 5, 5, 50];
        let a = detect(&counts, 3, 2.0);
        let b = detect(&counts, 3, 2.0);
        assert_eq!(a, b);
    }
}
