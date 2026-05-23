//! # Recipe: FP8 (E4M3) Lint — Happy Path
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr fp8-lint --observation-file observation.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the FP8 (E4M3) lint pipeline by synthesizing an observation
//! that records (a) Frobenius round-trip error per tensor, (b) GPU SM
//! capability reported by the producer, and (c) E4M3 saturation flags.
//! The happy-path observation passes the six rules `apr fp8-lint` enforces.
//!
//! ## Run Command
//! ```bash
//! cargo run --example fp8_lint_happy
//! ```
//!
//! ## References
//! - Micikevicius, P. et al. (2022). *FP8 Formats for Deep Learning*. arXiv:2209.05433
//! - NVIDIA Hopper Architecture Whitepaper (SM 9.0+ adds E4M3 hardware support).
//!
//! Added by PMAT-089 (expand-cookbooks followup — quantization lint coverage).

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintFinding {
    pub rule: String,
    pub severity: &'static str,
    pub message: String,
}

/// Run the FP8 (E4M3) lint rule-set against a captured observation.
pub fn lint_fp8_observation(obs: &Value) -> Vec<LintFinding> {
    let mut out = Vec::new();

    // Rule 1: schema_version present and >= 1.
    match obs.get("schema_version").and_then(Value::as_u64) {
        Some(v) if v >= 1 => {}
        _ => out.push(LintFinding {
            rule: "FP8-001".into(),
            severity: "error",
            message: "schema_version missing or < 1".into(),
        }),
    }

    // Rule 2: format must be exactly "E4M3" (E5M2 has its own lint).
    match obs.get("format").and_then(Value::as_str) {
        Some("E4M3") => {}
        _ => out.push(LintFinding {
            rule: "FP8-002".into(),
            severity: "error",
            message: "format must be \"E4M3\"".into(),
        }),
    }

    // Rule 3: SM capability >= 9.0 (Hopper is the first arch with native E4M3).
    let sm_major = obs
        .pointer("/capability/sm_major")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    if sm_major < 9 {
        out.push(LintFinding {
            rule: "FP8-003".into(),
            severity: "error",
            message: format!("sm_major {sm_major} < 9 — E4M3 needs Hopper SM 9.0+"),
        });
    }

    // Rule 4: frobenius_rel_err finite, < 0.05 (5% drift band).
    match obs.get("frobenius_rel_err").and_then(Value::as_f64) {
        Some(e) if e.is_finite() && (0.0..0.05).contains(&e) => {}
        _ => out.push(LintFinding {
            rule: "FP8-004".into(),
            severity: "warn",
            message: "frobenius_rel_err must be finite, in [0, 0.05)".into(),
        }),
    }

    // Rule 5: saturation_count must be 0 (any saturation = silent precision loss).
    match obs.get("saturation_count").and_then(Value::as_u64) {
        Some(0) => {}
        _ => out.push(LintFinding {
            rule: "FP8-005".into(),
            severity: "error",
            message: "saturation_count must be 0 — saturation indicates scale-factor mis-tune"
                .into(),
        }),
    }

    // Rule 6: scale_factor must be finite, > 0.
    match obs.get("scale_factor").and_then(Value::as_f64) {
        Some(s) if s.is_finite() && s > 0.0 => {}
        _ => out.push(LintFinding {
            rule: "FP8-006".into(),
            severity: "error",
            message: "scale_factor must be finite and > 0".into(),
        }),
    }

    out
}

pub fn build_happy_observation() -> Value {
    json!({
        "schema_version": 1,
        "format": "E4M3",
        "model": "llama-7b-fp8",
        "capability": { "sm_major": 9, "sm_minor": 0, "device": "H100" },
        "frobenius_rel_err": 0.012,
        "saturation_count": 0,
        "scale_factor": 448.0
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("fp8_lint_happy")?;
    let observation = build_happy_observation();

    let obs_path = ctx.path("fp8_observation.json");
    std::fs::write(
        &obs_path,
        serde_json::to_vec_pretty(&observation)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let findings = lint_fp8_observation(&observation);
    let errors = findings.iter().filter(|f| f.severity == "error").count();
    let warnings = findings.iter().filter(|f| f.severity == "warn").count();

    println!("=== Recipe: {} ===", ctx.name());
    println!("Observation: {}", obs_path.display());
    println!("Findings: {errors} errors, {warnings} warnings");
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
    fn happy_path_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_observation_has_no_errors() {
        let findings = lint_fp8_observation(&build_happy_observation());
        let errs: Vec<_> = findings.iter().filter(|f| f.severity == "error").collect();
        assert!(errs.is_empty(), "expected clean: {errs:?}");
    }

    #[test]
    fn rejects_e5m2_format() {
        let mut obs = build_happy_observation();
        obs["format"] = json!("E5M2");
        let findings = lint_fp8_observation(&obs);
        assert!(findings.iter().any(|f| f.rule == "FP8-002"));
    }

    #[test]
    fn rejects_pre_hopper_sm() {
        let mut obs = build_happy_observation();
        obs["capability"]["sm_major"] = json!(8);
        let findings = lint_fp8_observation(&obs);
        assert!(findings.iter().any(|f| f.rule == "FP8-003"));
    }
}
