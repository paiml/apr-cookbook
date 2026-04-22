//! # Recipe: DRY-Sampling Lint — Repetition Edge Case
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr dry-sampling-lint observation.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example dry_sampling_lint_repetition` exits 0
//! 2. [x] `cargo test --example dry_sampling_lint_repetition` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Synthesises a generation log that exhibits pathological repetition
//! (`longest_match = 48` in a 256-token window) and shows how the
//! `repetition_ratio` rule catches it when the sampler multiplier is too low
//! to damp the loop. Edge case: valid schema, but effective penalty is
//! insufficient.
//!
//! ## Run Command
//! ```bash
//! cargo run --example dry_sampling_lint_repetition
//! ```
//!
//! ## References
//! - Xu, C. et al. (2024). *DRY: A Modern Repetition Penalty That Preserves Creativity*. arXiv:2409.00509

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct Finding {
    pub rule: String,
    pub severity: &'static str,
    pub message: String,
}

/// Compute repetition_ratio = longest_match / emitted.
pub fn repetition_ratio(obs: &Value) -> f64 {
    let lm = obs
        .get("longest_match")
        .and_then(Value::as_u64)
        .unwrap_or(0) as f64;
    let em = obs
        .get("emitted_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .max(1) as f64;
    lm / em
}

pub fn lint_repetition(obs: &Value) -> Vec<Finding> {
    let mut out = Vec::new();
    let ratio = repetition_ratio(obs);
    if ratio > 0.10 {
        out.push(Finding {
            rule: "DRY-010".into(),
            severity: "error",
            message: format!(
                "repetition_ratio={:.3} > 0.10 — sampler failed to damp loop",
                ratio
            ),
        });
    }
    let mult = obs.get("multiplier").and_then(Value::as_f64).unwrap_or(0.0);
    if ratio > 0.10 && mult < 0.5 {
        out.push(Finding {
            rule: "DRY-011".into(),
            severity: "warn",
            message: format!(
                "multiplier={} is likely too low for the observed repetition",
                mult
            ),
        });
    }
    out
}

fn build_looping() -> Value {
    json!({
        "schema_version": 1,
        "sampler": "dry",
        "multiplier": 0.2,
        "base": 1.5,
        "allowed_length": 2,
        "emitted_tokens": 256,
        "penalized_tokens": 200,
        "longest_match": 48
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("dry_sampling_lint_repetition")?;
    let obs = build_looping();
    let p = ctx.path("looping.json");
    std::fs::write(
        &p,
        serde_json::to_vec_pretty(&obs).map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let ratio = repetition_ratio(&obs);
    let findings = lint_repetition(&obs);
    let errors = findings.iter().filter(|f| f.severity == "error").count();

    println!("=== Recipe: {} ===", ctx.name());
    println!("Observation: {}", p.display());
    println!("repetition_ratio = {:.3}", ratio);
    for f in &findings {
        println!("  [{}] {} — {}", f.severity, f.rule, f.message);
    }
    println!(
        "\nBumping multiplier from {} to >=0.8 would suppress the loop.",
        obs["multiplier"]
    );

    ctx.record_float_metric("repetition_ratio", ratio);
    ctx.record_metric("errors", errors as i64);
    ctx.record_string_metric("verdict", if errors == 0 { "PASS" } else { "FAIL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratio_over_threshold() {
        let obs = build_looping();
        assert!(repetition_ratio(&obs) > 0.10);
    }

    #[test]
    fn rule_010_triggered() {
        let f = lint_repetition(&build_looping());
        assert!(f.iter().any(|x| x.rule == "DRY-010"));
    }

    #[test]
    fn low_multiplier_warns() {
        let f = lint_repetition(&build_looping());
        assert!(f.iter().any(|x| x.rule == "DRY-011"));
    }
}
