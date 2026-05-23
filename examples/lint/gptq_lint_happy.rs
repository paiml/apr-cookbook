//! # Recipe: GPTQ Lint — Happy Path
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr gptq-lint --observation-file observation.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the GPTQ lint pipeline by synthesizing an observation
//! recording (a) per-block compression ratio, (b) cosine similarity vs
//! FP16 reference, and (c) the act_order / desc_act / sym_quant flag
//! consistency. The happy-path observation passes all six rules.
//!
//! ## Run Command
//! ```bash
//! cargo run --example gptq_lint_happy
//! ```
//!
//! ## References
//! - Frantar, E. et al. (2023). *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*. arXiv:2210.17323
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

pub fn lint_gptq_observation(obs: &Value) -> Vec<LintFinding> {
    let mut out = Vec::new();

    // Rule 1: schema_version present and >= 1.
    match obs.get("schema_version").and_then(Value::as_u64) {
        Some(v) if v >= 1 => {}
        _ => out.push(LintFinding {
            rule: "GPTQ-001".into(),
            severity: "error",
            message: "schema_version missing or < 1".into(),
        }),
    }

    // Rule 2: bits in {2, 3, 4, 8}.
    match obs.get("bits").and_then(Value::as_u64) {
        Some(2 | 3 | 4 | 8) => {}
        _ => out.push(LintFinding {
            rule: "GPTQ-002".into(),
            severity: "error",
            message: "bits must be 2, 3, 4, or 8".into(),
        }),
    }

    // Rule 3: group_size power of 2 in [-1, 32, 64, 128, 256, 512, 1024]; -1 = per-channel.
    match obs.get("group_size").and_then(Value::as_i64) {
        Some(-1) => {}
        Some(g) if (32..=1024).contains(&g) && (g as u64).is_power_of_two() => {}
        _ => out.push(LintFinding {
            rule: "GPTQ-003".into(),
            severity: "error",
            message: "group_size must be -1 (per-channel) or a power of 2 in [32, 1024]".into(),
        }),
    }

    // Rule 4: cosine_similarity finite, in (0.95, 1.0]. Below 0.95 = ship-blocker.
    match obs.get("cosine_similarity").and_then(Value::as_f64) {
        Some(c) if c.is_finite() && c > 0.95 && c <= 1.0 => {}
        _ => out.push(LintFinding {
            rule: "GPTQ-004".into(),
            severity: "error",
            message: "cosine_similarity must be finite, in (0.95, 1.0]".into(),
        }),
    }

    // Rule 5: compression_ratio > 1.0 (otherwise quantization wasn't applied).
    match obs.get("compression_ratio").and_then(Value::as_f64) {
        Some(r) if r.is_finite() && r > 1.0 => {}
        _ => out.push(LintFinding {
            rule: "GPTQ-005".into(),
            severity: "warn",
            message: "compression_ratio must be > 1.0".into(),
        }),
    }

    // Rule 6: act_order=true requires desc_act=true (kernels assume the pair).
    let act_order = obs
        .pointer("/flags/act_order")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let desc_act = obs
        .pointer("/flags/desc_act")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if act_order && !desc_act {
        out.push(LintFinding {
            rule: "GPTQ-006".into(),
            severity: "error",
            message: "act_order=true requires desc_act=true (kernel pairing)".into(),
        });
    }

    out
}

pub fn build_happy_observation() -> Value {
    json!({
        "schema_version": 1,
        "model": "llama-7b-gptq",
        "bits": 4,
        "group_size": 128,
        "cosine_similarity": 0.989,
        "compression_ratio": 3.91,
        "calibration_samples": 128,
        "flags": {
            "act_order": true,
            "desc_act": true,
            "sym_quant": false
        }
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("gptq_lint_happy")?;
    let observation = build_happy_observation();

    let obs_path = ctx.path("gptq_observation.json");
    std::fs::write(
        &obs_path,
        serde_json::to_vec_pretty(&observation)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let findings = lint_gptq_observation(&observation);
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
        let f = lint_gptq_observation(&build_happy_observation());
        let errs: Vec<_> = f.iter().filter(|x| x.severity == "error").collect();
        assert!(errs.is_empty(), "expected clean: {errs:?}");
    }

    #[test]
    fn rejects_invalid_bits() {
        let mut obs = build_happy_observation();
        obs["bits"] = json!(5);
        let f = lint_gptq_observation(&obs);
        assert!(f.iter().any(|x| x.rule == "GPTQ-002"));
    }

    #[test]
    fn rejects_act_order_without_desc_act() {
        let mut obs = build_happy_observation();
        obs["flags"]["desc_act"] = json!(false);
        let f = lint_gptq_observation(&obs);
        assert!(f.iter().any(|x| x.rule == "GPTQ-006"));
    }

    #[test]
    fn accepts_per_channel_group_size_minus_one() {
        let mut obs = build_happy_observation();
        obs["group_size"] = json!(-1);
        let f = lint_gptq_observation(&obs);
        let errs: Vec<_> = f.iter().filter(|x| x.severity == "error").collect();
        assert!(errs.is_empty());
    }
}
