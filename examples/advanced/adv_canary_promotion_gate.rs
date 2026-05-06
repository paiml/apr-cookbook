//! # Advanced Canary Promotion Gate
//!
//! Promote candidate model from canary (X% traffic) to full production
//! when:
//!
//! - error_rate(candidate) < error_rate(baseline) × max_regression_factor
//! - p99_latency(candidate) < p99_latency(baseline) × max_regression_factor
//! - sample size on candidate >= min_samples
//!
//! Otherwise: ContinueCanary or Rollback.
//!
//! Demonstrates the **ADV.7** recipe for PMAT-139 (advanced coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Spinnaker canary analysis Kayenta methodology.
//!
//! Run with: cargo run --example adv_canary_promotion_gate
//!
//! Added by PMAT-139 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const DEFAULT_MIN_SAMPLES: u64 = 1000;
const DEFAULT_MAX_REGRESSION: f64 = 1.05;

#[derive(Debug, PartialEq)]
pub enum CanaryVerdict {
    Promote,
    ContinueCanary { reason: &'static str },
    Rollback { reason: &'static str },
    InvalidMetrics,
}

pub fn evaluate(
    baseline_error_rate: f64,
    candidate_error_rate: f64,
    baseline_p99_ms: u32,
    candidate_p99_ms: u32,
    candidate_sample_count: u64,
) -> CanaryVerdict {
    if !baseline_error_rate.is_finite()
        || !candidate_error_rate.is_finite()
        || baseline_error_rate < 0.0
        || candidate_error_rate < 0.0
    {
        return CanaryVerdict::InvalidMetrics;
    }
    if candidate_sample_count < DEFAULT_MIN_SAMPLES {
        return CanaryVerdict::ContinueCanary {
            reason: "insufficient samples for stable comparison",
        };
    }
    let error_threshold = baseline_error_rate * DEFAULT_MAX_REGRESSION;
    if candidate_error_rate > error_threshold * 2.0 {
        return CanaryVerdict::Rollback {
            reason: "error rate is 2× regression threshold",
        };
    }
    if candidate_error_rate > error_threshold {
        return CanaryVerdict::ContinueCanary {
            reason: "error rate above acceptable regression",
        };
    }
    let latency_threshold = (f64::from(baseline_p99_ms) * DEFAULT_MAX_REGRESSION) as u32;
    if candidate_p99_ms > latency_threshold * 2 {
        return CanaryVerdict::Rollback {
            reason: "p99 latency is 2× regression threshold",
        };
    }
    if candidate_p99_ms > latency_threshold {
        return CanaryVerdict::ContinueCanary {
            reason: "p99 latency above acceptable regression",
        };
    }
    CanaryVerdict::Promote
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_canary_promotion_gate")?;

    println!("promote: {:?}", evaluate(0.001, 0.0009, 100, 95, 5_000));
    println!(
        "continue (low samples): {:?}",
        evaluate(0.001, 0.0009, 100, 95, 500)
    );
    println!(
        "continue (slight regression): {:?}",
        evaluate(0.001, 0.002, 100, 95, 5_000)
    );
    println!("rollback: {:?}", evaluate(0.001, 0.05, 100, 500, 5_000));
    println!("invalid: {:?}", evaluate(-0.1, 0.001, 100, 95, 5_000));
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
    fn equal_quality_promotes() {
        let v = evaluate(0.001, 0.001, 100, 100, 5_000);
        assert_eq!(v, CanaryVerdict::Promote);
    }

    #[test]
    fn slight_improvement_promotes() {
        let v = evaluate(0.001, 0.0005, 100, 95, 5_000);
        assert_eq!(v, CanaryVerdict::Promote);
    }

    #[test]
    fn within_regression_threshold_promotes() {
        // 0.001 × 1.05 = 0.00105.
        let v = evaluate(0.001, 0.00104, 100, 100, 5_000);
        assert_eq!(v, CanaryVerdict::Promote);
    }

    #[test]
    fn slight_regression_continues() {
        // 0.001 × 1.05 = 0.00105; 0.0012 > threshold but < 2x.
        let v = evaluate(0.001, 0.0012, 100, 95, 5_000);
        assert!(matches!(v, CanaryVerdict::ContinueCanary { .. }));
    }

    #[test]
    fn extreme_regression_rolls_back() {
        let v = evaluate(0.001, 0.05, 100, 95, 5_000);
        assert!(matches!(v, CanaryVerdict::Rollback { .. }));
    }

    #[test]
    fn insufficient_samples_continues() {
        let v = evaluate(0.001, 0.001, 100, 100, 500);
        assert!(matches!(v, CanaryVerdict::ContinueCanary { .. }));
    }

    #[test]
    fn high_p99_continues_then_rollback() {
        // baseline 100, threshold 105; 130 > threshold but < 2× = 210.
        let v_continue = evaluate(0.001, 0.001, 100, 130, 5_000);
        assert!(matches!(v_continue, CanaryVerdict::ContinueCanary { .. }));

        // 250 > 2× 105 = 210.
        let v_rb = evaluate(0.001, 0.001, 100, 250, 5_000);
        assert!(matches!(v_rb, CanaryVerdict::Rollback { .. }));
    }

    #[test]
    fn negative_error_rate_invalid() {
        assert_eq!(
            evaluate(-0.1, 0.001, 100, 100, 5_000),
            CanaryVerdict::InvalidMetrics
        );
    }

    #[test]
    fn nan_metrics_invalid() {
        assert_eq!(
            evaluate(f64::NAN, 0.001, 100, 100, 5_000),
            CanaryVerdict::InvalidMetrics
        );
    }

    #[test]
    fn zero_baseline_zero_candidate_promotes() {
        let v = evaluate(0.0, 0.0, 100, 100, 5_000);
        assert_eq!(v, CanaryVerdict::Promote);
    }
}
