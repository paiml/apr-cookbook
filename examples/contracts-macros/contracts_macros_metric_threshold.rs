//! # Contracts-Macros Metric Threshold Validator
//!
//! Validate observed metrics against contract thresholds:
//!   "p99_ms ≤ 100" → check observed p99 ≤ 100
//!   "throughput ≥ 200 tps" → check observed throughput ≥ 200
//! Returns first violating threshold.
//!
//! Demonstrates the **CMM.24** recipe for PMAT-165 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SLO contracts (Google SRE) + Prometheus alerting rules.
//!
//! Run with: cargo run --example contracts_macros_metric_threshold
//!
//! Added by PMAT-165 (catalog 1108→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComparisonOp {
    LessOrEqual,
    GreaterOrEqual,
    Less,
    Greater,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Threshold {
    pub metric: String,
    pub op: ComparisonOp,
    pub value: f64,
}

#[derive(Debug, PartialEq)]
pub enum ThresholdVerdict {
    Ok,
    Violated {
        metric: String,
        observed: f64,
        threshold: f64,
        op: ComparisonOp,
    },
    InvalidObserved,
}

pub fn check(thresholds: &[Threshold], observations: &[(&str, f64)]) -> ThresholdVerdict {
    for t in thresholds {
        let observed = observations
            .iter()
            .find(|(k, _)| *k == t.metric)
            .map(|(_, v)| *v);
        let Some(o) = observed else {
            continue;
        };
        if !o.is_finite() {
            return ThresholdVerdict::InvalidObserved;
        }
        let passes = match t.op {
            ComparisonOp::LessOrEqual => o <= t.value,
            ComparisonOp::GreaterOrEqual => o >= t.value,
            ComparisonOp::Less => o < t.value,
            ComparisonOp::Greater => o > t.value,
        };
        if !passes {
            return ThresholdVerdict::Violated {
                metric: t.metric.clone(),
                observed: o,
                threshold: t.value,
                op: t.op,
            };
        }
    }
    ThresholdVerdict::Ok
}

fn t(metric: &str, op: ComparisonOp, value: f64) -> Threshold {
    Threshold {
        metric: metric.to_string(),
        op,
        value,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_metric_threshold")?;

    let thresholds = vec![
        t("p99_ms", ComparisonOp::LessOrEqual, 100.0),
        t("throughput", ComparisonOp::GreaterOrEqual, 200.0),
    ];

    let healthy = vec![("p99_ms", 50.0), ("throughput", 250.0)];
    println!("ok: {:?}", check(&thresholds, &healthy));

    let unhealthy = vec![("p99_ms", 150.0), ("throughput", 100.0)];
    println!("violated: {:?}", check(&thresholds, &unhealthy));
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
    fn within_threshold_ok() {
        let ts = [t("p99_ms", ComparisonOp::LessOrEqual, 100.0)];
        let obs = [("p99_ms", 50.0)];
        assert_eq!(check(&ts, &obs), ThresholdVerdict::Ok);
    }

    #[test]
    fn violation_returned() {
        let ts = [t("p99_ms", ComparisonOp::LessOrEqual, 100.0)];
        let obs = [("p99_ms", 150.0)];
        let v = check(&ts, &obs);
        if let ThresholdVerdict::Violated {
            metric, observed, ..
        } = v
        {
            assert_eq!(metric, "p99_ms");
            assert!((observed - 150.0).abs() < 1e-9);
        }
    }

    #[test]
    fn at_boundary_le_ok() {
        let ts = [t("p99_ms", ComparisonOp::LessOrEqual, 100.0)];
        let obs = [("p99_ms", 100.0)];
        assert_eq!(check(&ts, &obs), ThresholdVerdict::Ok);
    }

    #[test]
    fn at_boundary_lt_violated() {
        let ts = [t("p99_ms", ComparisonOp::Less, 100.0)];
        let obs = [("p99_ms", 100.0)];
        assert!(matches!(
            check(&ts, &obs),
            ThresholdVerdict::Violated { .. }
        ));
    }

    #[test]
    fn ge_op() {
        let ts = [t("tps", ComparisonOp::GreaterOrEqual, 200.0)];
        let obs = [("tps", 250.0)];
        assert_eq!(check(&ts, &obs), ThresholdVerdict::Ok);
    }

    #[test]
    fn missing_metric_skipped() {
        let ts = [t("missing", ComparisonOp::LessOrEqual, 100.0)];
        let obs = [("other", 50.0)];
        assert_eq!(check(&ts, &obs), ThresholdVerdict::Ok);
    }

    #[test]
    fn nan_observed_invalid() {
        let ts = [t("p99_ms", ComparisonOp::LessOrEqual, 100.0)];
        let obs = [("p99_ms", f64::NAN)];
        assert_eq!(check(&ts, &obs), ThresholdVerdict::InvalidObserved);
    }

    #[test]
    fn first_violation_returned() {
        let ts = [
            t("p99_ms", ComparisonOp::LessOrEqual, 100.0),
            t("tps", ComparisonOp::GreaterOrEqual, 200.0),
        ];
        let obs = [("p99_ms", 150.0), ("tps", 100.0)];
        let v = check(&ts, &obs);
        if let ThresholdVerdict::Violated { metric, .. } = v {
            assert_eq!(metric, "p99_ms");
        }
    }

    #[test]
    fn empty_thresholds_ok() {
        assert_eq!(check(&[], &[("a", 1.0)]), ThresholdVerdict::Ok);
    }

    #[test]
    fn deterministic() {
        let ts = [t("p99_ms", ComparisonOp::LessOrEqual, 100.0)];
        let obs = [("p99_ms", 50.0)];
        let a = check(&ts, &obs);
        let b = check(&ts, &obs);
        assert_eq!(a, b);
    }
}
