//! # Recipe: DRY-Sampling Lint — Happy Path
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr dry-sampling-lint observation.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example dry_sampling_lint_happy` exits 0
//! 2. [x] `cargo test --example dry_sampling_lint_happy` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Demonstrates the DRY (Don't Repeat Yourself) sampling lint pipeline. The DRY
//! sampler penalises tokens that participate in long repeated sequences. This
//! recipe emits a valid observation and runs the canonical rule-set: multiplier,
//! base, allowed-length bounds, and a sanity check on reported rep-count.
//!
//! ## Run Command
//! ```bash
//! cargo run --example dry_sampling_lint_happy
//! ```
//!
//! ## References
//! - Xu, C. et al. (2024). *DRY: A Modern Repetition Penalty That Preserves Creativity*. arXiv:2409.00509

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub rule: String,
    pub severity: &'static str,
    pub message: String,
}

pub fn lint_dry(obs: &Value) -> Vec<Finding> {
    let mut out = Vec::new();

    // DRY-001: multiplier in (0.0, 4.0]
    match obs.get("multiplier").and_then(Value::as_f64) {
        Some(m) if m > 0.0 && m <= 4.0 => {}
        _ => out.push(Finding {
            rule: "DRY-001".into(),
            severity: "error",
            message: "multiplier must be in (0.0, 4.0]".into(),
        }),
    }

    // DRY-002: base ≥ 1.0 (DRY uses base^(match_len − allowed_length))
    match obs.get("base").and_then(Value::as_f64) {
        Some(b) if b >= 1.0 => {}
        _ => out.push(Finding {
            rule: "DRY-002".into(),
            severity: "error",
            message: "base must be >= 1.0".into(),
        }),
    }

    // DRY-003: allowed_length ≥ 1
    match obs.get("allowed_length").and_then(Value::as_u64) {
        Some(l) if l >= 1 => {}
        _ => out.push(Finding {
            rule: "DRY-003".into(),
            severity: "error",
            message: "allowed_length must be >= 1".into(),
        }),
    }

    // DRY-004: penalized tokens ≤ emitted
    let pen = obs
        .get("penalized_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let emit = obs
        .get("emitted_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if pen > emit {
        out.push(Finding {
            rule: "DRY-004".into(),
            severity: "error",
            message: "penalized_tokens cannot exceed emitted_tokens".into(),
        });
    }

    out
}

fn build_happy() -> Value {
    json!({
        "schema_version": 1,
        "sampler": "dry",
        "multiplier": 0.8,
        "base": 1.75,
        "allowed_length": 2,
        "emitted_tokens": 256,
        "penalized_tokens": 12,
        "longest_match": 3
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("dry_sampling_lint_happy")?;
    let obs = build_happy();

    let p = ctx.path("dry.json");
    std::fs::write(
        &p,
        serde_json::to_vec_pretty(&obs).map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let findings = lint_dry(&obs);
    let errors = findings.iter().filter(|f| f.severity == "error").count();

    println!("=== Recipe: {} ===", ctx.name());
    println!("Observation: {}", p.display());
    println!("Findings: {}", findings.len());
    for f in &findings {
        println!("  [{}] {} — {}", f.severity, f.rule, f.message);
    }
    println!(
        "\nmultiplier={} base={} allowed_length={} — within canonical ranges",
        obs["multiplier"], obs["base"], obs["allowed_length"]
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
    fn happy_path_is_clean() {
        assert!(lint_dry(&build_happy()).is_empty());
    }

    #[test]
    fn missing_multiplier_flags_rule_001() {
        let mut obs = build_happy();
        obs["multiplier"] = json!(-0.1);
        let f = lint_dry(&obs);
        assert!(f.iter().any(|x| x.rule == "DRY-001"));
    }

    #[test]
    fn penalised_gt_emitted_flags_rule_004() {
        let mut obs = build_happy();
        obs["penalized_tokens"] = json!(1000);
        let f = lint_dry(&obs);
        assert!(f.iter().any(|x| x.rule == "DRY-004"));
    }
}
