//! # Monitoring Canary Metric Comparison
//!
//! Compare pre/post canary deployment metrics:
//!   relative_diff = (post - pre) / pre
//!   significance: relative_diff > threshold AND sample_size ≥ min
//!
//! Verdict: NoChange / SignificantImprovement / SignificantRegression /
//! InsufficientData.
//!
//! Demonstrates the **MON.35** recipe for PMAT-154 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Spinnaker Kayenta canary analysis.
//!
//! Run with: cargo run --example monitor_canary_metric
//!
//! Added by PMAT-154 (catalog 1009→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MIN_SAMPLES: u64 = 1000;
const SIGNIFICANCE_THRESHOLD: f64 = 0.05;

#[derive(Debug, PartialEq)]
pub enum CanaryVerdict {
    NoChange { relative_diff: f64 },
    SignificantImprovement { relative_diff: f64 },
    SignificantRegression { relative_diff: f64 },
    InsufficientData,
    InvalidMetrics,
}

pub fn compare(
    pre_value: f64,
    post_value: f64,
    sample_count: u64,
    higher_is_better: bool,
) -> CanaryVerdict {
    if !pre_value.is_finite() || !post_value.is_finite() {
        return CanaryVerdict::InvalidMetrics;
    }
    if pre_value <= 0.0 {
        return CanaryVerdict::InvalidMetrics;
    }
    if sample_count < MIN_SAMPLES {
        return CanaryVerdict::InsufficientData;
    }
    let relative_diff = (post_value - pre_value) / pre_value;
    if relative_diff.abs() < SIGNIFICANCE_THRESHOLD {
        return CanaryVerdict::NoChange { relative_diff };
    }
    let is_improvement = if higher_is_better {
        relative_diff > 0.0
    } else {
        relative_diff < 0.0
    };
    if is_improvement {
        CanaryVerdict::SignificantImprovement { relative_diff }
    } else {
        CanaryVerdict::SignificantRegression { relative_diff }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_canary_metric")?;

    println!(
        "p99 went up (regression): {:?}",
        compare(100.0, 110.0, 5000, false)
    );
    println!(
        "throughput up (improvement): {:?}",
        compare(1000.0, 1100.0, 5000, true)
    );
    println!("no change: {:?}", compare(100.0, 102.0, 5000, false));
    println!("insufficient: {:?}", compare(100.0, 110.0, 100, false));
    println!("invalid: {:?}", compare(0.0, 50.0, 5000, false));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn p99_increase_is_regression() {
        // Higher latency = bad.
        let v = compare(100.0, 110.0, 5000, false);
        assert!(matches!(v, CanaryVerdict::SignificantRegression { .. }));
    }

    #[test]
    fn throughput_increase_is_improvement() {
        let v = compare(1000.0, 1100.0, 5000, true);
        assert!(matches!(v, CanaryVerdict::SignificantImprovement { .. }));
    }

    #[test]
    fn small_diff_no_change() {
        let v = compare(100.0, 102.0, 5000, false);
        assert!(matches!(v, CanaryVerdict::NoChange { .. }));
    }

    #[test]
    fn insufficient_samples() {
        let v = compare(100.0, 110.0, 100, false);
        assert_eq!(v, CanaryVerdict::InsufficientData);
    }

    #[test]
    fn invalid_zero_pre() {
        assert_eq!(
            compare(0.0, 50.0, 5000, true),
            CanaryVerdict::InvalidMetrics
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            compare(f64::NAN, 50.0, 5000, true),
            CanaryVerdict::InvalidMetrics
        );
    }

    #[test]
    fn relative_diff_returned() {
        let v = compare(100.0, 110.0, 5000, false);
        if let CanaryVerdict::SignificantRegression { relative_diff } = v {
            assert!((relative_diff - 0.10).abs() < 1e-9);
        }
    }

    #[test]
    fn boundary_at_5_pct_significant() {
        // 5% diff is exactly the threshold; treated as not-significant
        // (rule is `< THRESHOLD`).
        let v = compare(100.0, 105.0, 5000, false);
        assert!(matches!(v, CanaryVerdict::SignificantRegression { .. }));
    }

    #[test]
    fn just_below_threshold_no_change() {
        let v = compare(100.0, 104.0, 5000, false);
        assert!(matches!(v, CanaryVerdict::NoChange { .. }));
    }

    #[test]
    fn improvement_for_lower_is_better() {
        // Pre=100ms p99, post=80ms p99 (lower = better). Improvement.
        let v = compare(100.0, 80.0, 5000, false);
        assert!(matches!(v, CanaryVerdict::SignificantImprovement { .. }));
    }

    #[test]
    fn regression_for_higher_is_better() {
        // Pre=1000 throughput, post=900 (higher = better). Regression.
        let v = compare(1000.0, 900.0, 5000, true);
        assert!(matches!(v, CanaryVerdict::SignificantRegression { .. }));
    }
}
