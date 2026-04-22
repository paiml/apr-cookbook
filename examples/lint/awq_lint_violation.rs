//! # Recipe: AWQ Lint — Rule Violation
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr awq-lint observation.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example awq_lint_violation` exits 0
//! 2. [x] `cargo test --example awq_lint_violation` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Shows how AWQ lint rules catch concrete quantization defects: a non-power-of-two
//! `group_size`, an out-of-band `clip_ratio`, and mutually-exclusive `zero_point` +
//! `symmetric` flags. The recipe reports WHICH rule triggered for each defect.
//!
//! ## Run Command
//! ```bash
//! cargo run --example awq_lint_violation
//! ```
//!
//! ## References
//! - Lin, J. et al. (2024). *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration*. arXiv:2306.00978

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintFinding {
    pub rule: String,
    pub severity: &'static str,
    pub message: String,
}

/// Condensed rule-set (same rule IDs as `awq_lint_happy`).
pub fn lint_awq_observation(obs: &Value) -> Vec<LintFinding> {
    let mut out = Vec::new();

    if !matches!(obs.get("group_size").and_then(Value::as_u64),
        Some(g) if (32..=256).contains(&g) && g.is_power_of_two())
    {
        out.push(LintFinding {
            rule: "AWQ-002".into(),
            severity: "error",
            message: "group_size must be power-of-two in [32, 256]".into(),
        });
    }
    if !matches!(obs.get("bits").and_then(Value::as_u64), Some(3 | 4)) {
        out.push(LintFinding {
            rule: "AWQ-003".into(),
            severity: "error",
            message: "bits must be 3 or 4".into(),
        });
    }
    if !matches!(obs.get("clip_ratio").and_then(Value::as_f64),
        Some(c) if c > 0.0 && c <= 1.0)
    {
        out.push(LintFinding {
            rule: "AWQ-005".into(),
            severity: "error",
            message: "clip_ratio must be in (0, 1]".into(),
        });
    }
    let zp = obs
        .pointer("/flags/zero_point")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let sym = obs
        .pointer("/flags/symmetric")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if zp && sym {
        out.push(LintFinding {
            rule: "AWQ-006".into(),
            severity: "error",
            message: "zero_point and symmetric are mutually exclusive".into(),
        });
    }
    out
}

fn build_broken_observation() -> Value {
    json!({
        "schema_version": 1,
        "model": "mistral-7b-broken",
        // group_size = 100 (not power-of-two)
        "group_size": 100,
        "bits": 4,
        "ppl_delta": 0.11,
        // clip_ratio > 1.0
        "clip_ratio": 1.2,
        "flags": {
            // both set — mutually exclusive
            "zero_point": true,
            "symmetric": true
        }
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("awq_lint_violation")?;
    let obs = build_broken_observation();

    let obs_path = ctx.path("awq_broken.json");
    std::fs::write(
        &obs_path,
        serde_json::to_vec_pretty(&obs).map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let findings = lint_awq_observation(&obs);
    let errors = findings.iter().filter(|f| f.severity == "error").count();

    println!("=== Recipe: {} ===", ctx.name());
    println!("Observation: {}", obs_path.display());
    println!("Findings: {} errors", errors);
    for f in &findings {
        println!("  [{}] {} — {}", f.severity, f.rule, f.message);
    }
    println!(
        "\nExpected 3 concrete defects (group_size, clip_ratio, flags). \
         Got {}.",
        errors
    );

    ctx.record_metric("errors", errors as i64);
    ctx.record_string_metric("verdict", if errors == 0 { "PASS" } else { "FAIL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broken_observation_triggers_three_errors() {
        let findings = lint_awq_observation(&build_broken_observation());
        let errs = findings.iter().filter(|f| f.severity == "error").count();
        assert_eq!(errs, 3, "findings: {:?}", findings);
    }

    #[test]
    fn rule_ids_are_stable() {
        let findings = lint_awq_observation(&build_broken_observation());
        let ids: Vec<&str> = findings.iter().map(|f| f.rule.as_str()).collect();
        assert!(ids.contains(&"AWQ-002"));
        assert!(ids.contains(&"AWQ-005"));
        assert!(ids.contains(&"AWQ-006"));
    }
}
