//! # Recipe: AWQ Lint — Happy Path
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr awq-lint observation.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example awq_lint_happy` exits 0
//! 2. [x] `cargo test --example awq_lint_happy` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Demonstrates the AWQ (Activation-aware Weight Quantization) lint pipeline by
//! synthesizing a valid observation record and running each of the six invariants
//! the real `apr awq-lint` subcommand enforces. The happy-path observation passes
//! all checks cleanly.
//!
//! ## Run Command
//! ```bash
//! cargo run --example awq_lint_happy
//! ```
//!
//! ## References
//! - Lin, J. et al. (2024). *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration*. arXiv:2306.00978

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};

/// A single rule violation from AWQ lint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintFinding {
    pub rule: String,
    pub severity: &'static str,
    pub message: String,
}

/// Run the full AWQ lint rule-set against an observation.
pub fn lint_awq_observation(obs: &Value) -> Vec<LintFinding> {
    let mut out = Vec::new();

    // Rule 1: schema_version must be present and >= 1.
    match obs.get("schema_version").and_then(Value::as_u64) {
        Some(v) if v >= 1 => {}
        _ => out.push(LintFinding {
            rule: "AWQ-001".into(),
            severity: "error",
            message: "schema_version missing or < 1".into(),
        }),
    }

    // Rule 2: group_size must be a power of two in [32, 256].
    match obs.get("group_size").and_then(Value::as_u64) {
        Some(g) if (32..=256).contains(&g) && g.is_power_of_two() => {}
        _ => out.push(LintFinding {
            rule: "AWQ-002".into(),
            severity: "error",
            message: "group_size must be a power-of-two in [32, 256]".into(),
        }),
    }

    // Rule 3: bits must be 3 or 4 (INT3/INT4 are the AWQ target regimes).
    match obs.get("bits").and_then(Value::as_u64) {
        Some(3 | 4) => {}
        _ => out.push(LintFinding {
            rule: "AWQ-003".into(),
            severity: "error",
            message: "bits must be 3 or 4".into(),
        }),
    }

    // Rule 4: ppl_delta must be finite and within tolerated drift band (< 0.5).
    match obs.get("ppl_delta").and_then(Value::as_f64) {
        Some(d) if d.is_finite() && d.abs() < 0.5 => {}
        _ => out.push(LintFinding {
            rule: "AWQ-004".into(),
            severity: "warn",
            message: "ppl_delta must be finite and |delta| < 0.5".into(),
        }),
    }

    // Rule 5: clip_ratio ∈ (0, 1]
    match obs.get("clip_ratio").and_then(Value::as_f64) {
        Some(c) if c > 0.0 && c <= 1.0 => {}
        _ => out.push(LintFinding {
            rule: "AWQ-005".into(),
            severity: "error",
            message: "clip_ratio must be in (0, 1]".into(),
        }),
    }

    // Rule 6: flags.zero_point and flags.symmetric are mutually exclusive.
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
            message: "flags.zero_point and flags.symmetric are mutually exclusive".into(),
        });
    }

    out
}

fn build_happy_observation() -> Value {
    json!({
        "schema_version": 1,
        "model": "llama-7b",
        "group_size": 128,
        "bits": 4,
        "ppl_delta": 0.07,
        "clip_ratio": 0.85,
        "calibration_samples": 512,
        "flags": {
            "zero_point": false,
            "symmetric": true
        }
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("awq_lint_happy")?;
    let observation = build_happy_observation();

    let obs_path = ctx.path("awq_observation.json");
    std::fs::write(
        &obs_path,
        serde_json::to_vec_pretty(&observation)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let findings = lint_awq_observation(&observation);
    let errors = findings.iter().filter(|f| f.severity == "error").count();
    let warnings = findings.iter().filter(|f| f.severity == "warn").count();

    println!("=== Recipe: {} ===", ctx.name());
    println!("Observation: {}", obs_path.display());
    println!("Findings: {} errors, {} warnings", errors, warnings);
    for f in &findings {
        println!("  [{}] {} — {}", f.severity, f.rule, f.message);
    }

    ctx.record_metric("errors", errors as i64);
    ctx.record_metric("warnings", warnings as i64);
    ctx.record_string_metric("verdict", if errors == 0 { "PASS" } else { "FAIL" });

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_observation_has_no_findings() {
        let findings = lint_awq_observation(&build_happy_observation());
        assert!(findings.is_empty(), "expected clean: {:?}", findings);
    }

    #[test]
    fn detects_bad_group_size() {
        let mut obs = build_happy_observation();
        obs["group_size"] = json!(100);
        let findings = lint_awq_observation(&obs);
        assert!(findings.iter().any(|f| f.rule == "AWQ-002"));
    }

    #[test]
    fn detects_mutually_exclusive_flags() {
        let mut obs = build_happy_observation();
        obs["flags"]["zero_point"] = json!(true);
        obs["flags"]["symmetric"] = json!(true);
        let findings = lint_awq_observation(&obs);
        assert!(findings.iter().any(|f| f.rule == "AWQ-006"));
    }
}
