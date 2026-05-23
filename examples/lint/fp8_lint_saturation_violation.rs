//! # Recipe: FP8 Lint — Saturation Violation
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr fp8-lint --observation-file observation.json` (fail path)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the fail path of `apr fp8-lint` when an observation reports
//! E4M3 saturation events. Saturation occurs when the producer's chosen
//! scale factor squashes outliers above ±448 (E4M3 amax), silently losing
//! precision. The lint MUST flag any non-zero saturation_count as an
//! error — a "warn" would let the model ship with degraded quality.
//!
//! ## Run Command
//! ```bash
//! cargo run --example fp8_lint_saturation_violation
//! ```
//!
//! ## References
//! - Micikevicius, P. et al. (2022). *FP8 Formats for Deep Learning*. arXiv:2209.05433
//! - NVIDIA Transformer Engine docs: amax history & delayed scaling.
//!
//! Added by PMAT-089 (expand-cookbooks followup — quantization lint coverage).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintFinding {
    pub rule: String,
    pub severity: &'static str,
    pub message: String,
}

pub fn lint_saturation(obs: &Value) -> Vec<LintFinding> {
    let mut out = Vec::new();
    match obs.get("saturation_count").and_then(Value::as_u64) {
        Some(0) => {}
        Some(n) => out.push(LintFinding {
            rule: "FP8-005".into(),
            severity: "error",
            message: format!("saturation_count={n} — re-tune scale factor (amax history)"),
        }),
        None => out.push(LintFinding {
            rule: "FP8-005".into(),
            severity: "error",
            message: "saturation_count missing".into(),
        }),
    }
    out
}

fn build_violation_observation() -> Value {
    json!({
        "schema_version": 1,
        "format": "E4M3",
        "model": "llama-7b-fp8",
        "capability": { "sm_major": 9, "sm_minor": 0, "device": "H100" },
        "frobenius_rel_err": 0.018,
        "saturation_count": 1372,   // outliers in attention.k_proj squashed to amax
        "scale_factor": 448.0,
        "amax_history_len": 16
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("fp8_lint_saturation_violation")?;
    let obs = build_violation_observation();

    let findings = lint_saturation(&obs);
    println!("=== Recipe: {} ===", ctx.name());
    for f in &findings {
        println!("  [{}] {} — {}", f.severity, f.rule, f.message);
    }
    let verdict = if findings.is_empty() { "PASS" } else { "FAIL" };
    println!("verdict: {verdict}");
    ctx.record_string_metric("verdict", verdict);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn violation_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn nonzero_saturation_is_error_not_warn() {
        // Critical: saturation must be elevated to error severity, not warn —
        // otherwise CI grep-for-error misses silent precision loss.
        let f = lint_saturation(&build_violation_observation());
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].severity, "error");
        assert_eq!(f[0].rule, "FP8-005");
    }

    #[test]
    fn zero_saturation_is_clean() {
        let mut obs = build_violation_observation();
        obs["saturation_count"] = json!(0);
        let f = lint_saturation(&obs);
        assert!(f.is_empty());
    }

    #[test]
    fn missing_field_is_error() {
        // A missing field must NOT be silently treated as zero — that would
        // let observation tooling that forgot to emit the field "pass".
        let mut obs = build_violation_observation();
        obs.as_object_mut().unwrap().remove("saturation_count");
        let f = lint_saturation(&obs);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].severity, "error");
    }
}
