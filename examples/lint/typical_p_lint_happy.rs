//! # Recipe: Typical-P Sampling Lint — Happy Path
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr typical-p-lint --observation-file observation.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the typical-p (locally-typical) sampling lint pipeline
//! (CRUX-C-22). Typical-p sampling truncates the next-token distribution to
//! the smallest set whose **conditional entropy contribution** sums to
//! `typical_p`. The lint enforces six rules: schema_version, typical_p in
//! (0, 1.0], min_keep ≥ 1, sampled_set non-empty, sampled cumulative
//! probability ≥ typical_p, and entropy_target ≤ entropy_max.
//!
//! ## Run Command
//! ```bash
//! cargo run --example typical_p_lint_happy
//! ```
//!
//! ## References
//! - Meister, C. et al. (2023). *Locally Typical Sampling*. arXiv:2202.00666
//!
//! Added by PMAT-091 (expand-cookbooks followup — Ollama/sampling/imatrix lint).

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintFinding {
    pub rule: String,
    pub severity: &'static str,
    pub message: String,
}

pub fn lint_typical_p_observation(obs: &Value) -> Vec<LintFinding> {
    let mut out = Vec::new();

    // Rule 1: schema_version present and >= 1.
    match obs.get("schema_version").and_then(Value::as_u64) {
        Some(v) if v >= 1 => {}
        _ => out.push(LintFinding {
            rule: "TYP-001".into(),
            severity: "error",
            message: "schema_version missing or < 1".into(),
        }),
    }

    // Rule 2: typical_p finite, in (0, 1.0].
    match obs.get("typical_p").and_then(Value::as_f64) {
        Some(p) if p.is_finite() && p > 0.0 && p <= 1.0 => {}
        _ => out.push(LintFinding {
            rule: "TYP-002".into(),
            severity: "error",
            message: "typical_p must be finite, in (0, 1.0]".into(),
        }),
    }

    // Rule 3: min_keep must be >= 1.
    match obs.get("min_keep").and_then(Value::as_u64) {
        Some(k) if k >= 1 => {}
        _ => out.push(LintFinding {
            rule: "TYP-003".into(),
            severity: "error",
            message: "min_keep must be >= 1".into(),
        }),
    }

    // Rule 4: sampled_set must be non-empty.
    let set = obs.get("sampled_set").and_then(Value::as_array);
    match set {
        Some(s) if !s.is_empty() => {}
        _ => out.push(LintFinding {
            rule: "TYP-004".into(),
            severity: "error",
            message: "sampled_set must be a non-empty array".into(),
        }),
    }

    // Rule 5: sum of sampled_set probabilities must be >= typical_p (within 1e-6 slack).
    if let (Some(arr), Some(p)) = (set, obs.get("typical_p").and_then(Value::as_f64)) {
        let sum: f64 = arr
            .iter()
            .filter_map(|v| v.get("prob").and_then(Value::as_f64))
            .sum();
        if sum + 1e-6 < p {
            out.push(LintFinding {
                rule: "TYP-005".into(),
                severity: "error",
                message: format!("sum(sampled_set probs)={sum:.6} < typical_p={p:.6}"),
            });
        }
    }

    // Rule 6: entropy_target finite and <= entropy_max.
    let target = obs.get("entropy_target").and_then(Value::as_f64);
    let max = obs.get("entropy_max").and_then(Value::as_f64);
    if let (Some(t), Some(m)) = (target, max) {
        if !t.is_finite() || !m.is_finite() || t > m + 1e-9 {
            out.push(LintFinding {
                rule: "TYP-006".into(),
                severity: "error",
                message: format!("entropy_target={t} must be finite and <= entropy_max={m}"),
            });
        }
    }

    out
}

pub fn build_happy_observation() -> Value {
    json!({
        "schema_version": 1,
        "model": "llama-7b",
        "typical_p": 0.9,
        "min_keep": 1,
        "entropy_target": 1.94,    // log2(vocab) - typical_p band
        "entropy_max": 2.50,       // log2 of full kept set
        "sampled_set": [
            { "token_id": 17,   "prob": 0.42 },
            { "token_id": 23,   "prob": 0.31 },
            { "token_id": 99,   "prob": 0.18 }
        ]
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("typical_p_lint_happy")?;
    let obs = build_happy_observation();

    let obs_path = ctx.path("typical_p_observation.json");
    std::fs::write(
        &obs_path,
        serde_json::to_vec_pretty(&obs).map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let findings = lint_typical_p_observation(&obs);
    let errors = findings.iter().filter(|f| f.severity == "error").count();

    println!("=== Recipe: {} ===", ctx.name());
    println!("Observation: {}", obs_path.display());
    println!("Findings: {errors} errors");
    for f in &findings {
        println!("  [{}] {} — {}", f.severity, f.rule, f.message);
    }

    ctx.record_metric("errors", errors as i64);
    ctx.record_string_metric("verdict", if errors == 0 { "PASS" } else { "FAIL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_observation_has_no_errors() {
        let f = lint_typical_p_observation(&build_happy_observation());
        assert!(f.is_empty(), "expected clean: {f:?}");
    }

    #[test]
    fn rejects_typical_p_above_one() {
        let mut obs = build_happy_observation();
        obs["typical_p"] = json!(1.5);
        let f = lint_typical_p_observation(&obs);
        assert!(f.iter().any(|x| x.rule == "TYP-002"));
    }

    #[test]
    fn rejects_zero_min_keep() {
        let mut obs = build_happy_observation();
        obs["min_keep"] = json!(0);
        let f = lint_typical_p_observation(&obs);
        assert!(f.iter().any(|x| x.rule == "TYP-003"));
    }
}
