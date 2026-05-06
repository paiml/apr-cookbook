//! # apr experiment compare --gate — Metric Comparison CI Gate
//!
//! `apr experiment compare A B --gate <metric>:<delta>` requires the
//! named metric to improve by at least delta. For loss-style metrics,
//! lower is better; for accuracy-style, higher is better. This recipe
//! builds the gate (auto-direction by metric name).
//!
//! Demonstrates the **EXP.4** recipe for PMAT-118 (apr experiment coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender EXP-001 + MLflow run-comparison conventions
//!
//! Run with: cargo run --example cli_experiment_metric_compare_gate
//!
//! Added by PMAT-118 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GateVerdict {
    Pass { delta: f64 },
    Fail { observed: f64, required: f64 },
    InvalidValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricDirection {
    LowerIsBetter,
    HigherIsBetter,
}

pub fn auto_direction(metric_name: &str) -> MetricDirection {
    let lower_keywords = ["loss", "perplexity", "error", "rmse", "mae", "ppl"];
    let lname = metric_name.to_ascii_lowercase();
    if lower_keywords.iter().any(|k| lname.contains(k)) {
        MetricDirection::LowerIsBetter
    } else {
        MetricDirection::HigherIsBetter
    }
}

pub fn check(
    baseline: f64,
    candidate: f64,
    required_delta: f64,
    direction: MetricDirection,
) -> GateVerdict {
    if !baseline.is_finite() || !candidate.is_finite() || !required_delta.is_finite() {
        return GateVerdict::InvalidValue;
    }
    let signed = match direction {
        MetricDirection::LowerIsBetter => baseline - candidate,
        MetricDirection::HigherIsBetter => candidate - baseline,
    };
    if signed >= required_delta {
        GateVerdict::Pass { delta: signed }
    } else {
        GateVerdict::Fail {
            observed: signed,
            required: required_delta,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_experiment_metric_compare_gate")?;

    let cases = [
        ("loss", 1.5, 1.2, 0.1),
        ("loss", 1.5, 1.6, 0.1),
        ("accuracy", 0.80, 0.85, 0.02),
        ("accuracy", 0.80, 0.81, 0.05),
    ];
    for (name, base, cand, req) in cases {
        let dir = auto_direction(name);
        println!(
            "{name} {base} → {cand}  req≥{req}  ({dir:?})  →  {:?}",
            check(base, cand, req, dir)
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
    fn loss_decrease_meets_threshold_passes() {
        let v = check(1.5, 1.2, 0.1, MetricDirection::LowerIsBetter);
        assert!(matches!(v, GateVerdict::Pass { .. }));
    }

    #[test]
    fn loss_increase_fails() {
        let v = check(1.5, 1.6, 0.1, MetricDirection::LowerIsBetter);
        assert!(matches!(v, GateVerdict::Fail { .. }));
    }

    #[test]
    fn accuracy_increase_meets_threshold_passes() {
        let v = check(0.80, 0.85, 0.02, MetricDirection::HigherIsBetter);
        assert!(matches!(v, GateVerdict::Pass { .. }));
    }

    #[test]
    fn accuracy_below_threshold_fails() {
        let v = check(0.80, 0.81, 0.05, MetricDirection::HigherIsBetter);
        assert!(matches!(v, GateVerdict::Fail { .. }));
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            check(f64::NAN, 1.0, 0.1, MetricDirection::LowerIsBetter),
            GateVerdict::InvalidValue
        );
    }

    #[test]
    fn auto_direction_loss_lower() {
        assert_eq!(auto_direction("loss"), MetricDirection::LowerIsBetter);
        assert_eq!(auto_direction("perplexity"), MetricDirection::LowerIsBetter);
        assert_eq!(auto_direction("val_loss"), MetricDirection::LowerIsBetter);
    }

    #[test]
    fn auto_direction_accuracy_higher() {
        assert_eq!(auto_direction("accuracy"), MetricDirection::HigherIsBetter);
        assert_eq!(auto_direction("f1_score"), MetricDirection::HigherIsBetter);
        assert_eq!(auto_direction("recall"), MetricDirection::HigherIsBetter);
    }

    #[test]
    fn boundary_at_required_passes() {
        // ≥ required → Pass (inclusive). Using FP-exact values: 1.5 - 0.5 = 1.0.
        let v = check(1.5, 0.5, 1.0, MetricDirection::LowerIsBetter);
        assert!(matches!(v, GateVerdict::Pass { .. }));
    }

    #[test]
    fn case_insensitive_metric_names() {
        assert_eq!(auto_direction("LOSS"), MetricDirection::LowerIsBetter);
        assert_eq!(auto_direction("Accuracy"), MetricDirection::HigherIsBetter);
    }
}
