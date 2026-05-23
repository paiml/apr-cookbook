//! # Recipe: imatrix Calibration Lint — Happy Path
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr imatrix-lint --observation-file observation.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the imatrix (importance-matrix) calibration lint pipeline
//! (CRUX-B-07). imatrix is the calibration phase used by k-quant
//! quantization (Q4_K_M, Q5_K_M, Q6_K) where activation magnitudes guide
//! per-channel scale selection. The lint enforces six rules: schema_version,
//! calibration_corpus_size ≥ 100, per-tensor entries cover ≥90% of
//! tensors, no NaN/inf entries, scale_dtype is f32, and dataset hash is
//! recorded so reruns are reproducible.
//!
//! ## Run Command
//! ```bash
//! cargo run --example imatrix_lint_happy
//! ```
//!
//! ## References
//! - llama.cpp imatrix-tool docs (k-quant calibration phase).
//! - aprender CRUX-B-07 contract (imatrix observation).
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

pub fn lint_imatrix_observation(obs: &Value) -> Vec<LintFinding> {
    let mut out = Vec::new();

    // Rule 1: schema_version present and >= 1.
    match obs.get("schema_version").and_then(Value::as_u64) {
        Some(v) if v >= 1 => {}
        _ => out.push(LintFinding {
            rule: "IMAT-001".into(),
            severity: "error",
            message: "schema_version missing or < 1".into(),
        }),
    }

    // Rule 2: calibration_corpus_size >= 100 (smaller calibration sets
    // produce noisy importance estimates per llama.cpp recommendations).
    match obs.get("calibration_corpus_size").and_then(Value::as_u64) {
        Some(n) if n >= 100 => {}
        _ => out.push(LintFinding {
            rule: "IMAT-002".into(),
            severity: "error",
            message: "calibration_corpus_size must be >= 100 chunks".into(),
        }),
    }

    // Rule 3: tensor coverage must be >= 90% (per-tensor entries / total tensors).
    let covered = obs.get("tensors_covered").and_then(Value::as_u64);
    let total = obs.get("tensors_total").and_then(Value::as_u64);
    if let (Some(c), Some(t)) = (covered, total) {
        if t == 0 || c * 100 < t * 90 {
            out.push(LintFinding {
                rule: "IMAT-003".into(),
                severity: "error",
                message: format!(
                    "tensor coverage {c}/{t} below 90% — re-run calibration with longer prompts"
                ),
            });
        }
    }

    // Rule 4: no per-tensor importance values are NaN or inf.
    if let Some(entries) = obs.get("per_tensor").and_then(Value::as_array) {
        for e in entries {
            let name = e.get("name").and_then(Value::as_str).unwrap_or("?");
            if let Some(arr) = e.get("importance").and_then(Value::as_array) {
                for (i, v) in arr.iter().enumerate() {
                    let f = v.as_f64().unwrap_or(f64::NAN);
                    if !f.is_finite() {
                        out.push(LintFinding {
                            rule: "IMAT-004".into(),
                            severity: "error",
                            message: format!("per_tensor.{name}.importance[{i}] = {f} (NaN/inf)"),
                        });
                    }
                }
            }
        }
    }

    // Rule 5: scale_dtype must be "f32" (k-quant kernels assume f32 scales).
    match obs.get("scale_dtype").and_then(Value::as_str) {
        Some("f32") => {}
        _ => out.push(LintFinding {
            rule: "IMAT-005".into(),
            severity: "error",
            message: "scale_dtype must be \"f32\"".into(),
        }),
    }

    // Rule 6: dataset_hash must be a non-empty hex string (reproducibility).
    match obs.get("dataset_hash").and_then(Value::as_str) {
        Some(h) if h.len() >= 16 && h.chars().all(|c| c.is_ascii_hexdigit()) => {}
        _ => out.push(LintFinding {
            rule: "IMAT-006".into(),
            severity: "error",
            message: "dataset_hash must be hex string of length >= 16".into(),
        }),
    }

    out
}

pub fn build_happy_observation() -> Value {
    json!({
        "schema_version": 1,
        "model": "llama-7b",
        "calibration_corpus_size": 512,
        "tensors_covered": 290,
        "tensors_total": 300,
        "scale_dtype": "f32",
        "dataset_hash": "abcdef1234567890",
        "per_tensor": [
            { "name": "blk.0.attn_q.weight", "importance": [0.42, 0.31, 0.18] },
            { "name": "blk.0.attn_k.weight", "importance": [0.51, 0.27, 0.12] }
        ]
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("imatrix_lint_happy")?;
    let obs = build_happy_observation();

    let obs_path = ctx.path("imatrix_observation.json");
    std::fs::write(
        &obs_path,
        serde_json::to_vec_pretty(&obs).map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let findings = lint_imatrix_observation(&obs);
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
        let f = lint_imatrix_observation(&build_happy_observation());
        assert!(f.is_empty(), "expected clean: {f:?}");
    }

    #[test]
    fn rejects_small_corpus() {
        let mut obs = build_happy_observation();
        obs["calibration_corpus_size"] = json!(50);
        let f = lint_imatrix_observation(&obs);
        assert!(f.iter().any(|x| x.rule == "IMAT-002"));
    }

    #[test]
    fn rejects_low_tensor_coverage() {
        let mut obs = build_happy_observation();
        obs["tensors_covered"] = json!(100);
        obs["tensors_total"] = json!(300);
        let f = lint_imatrix_observation(&obs);
        assert!(f.iter().any(|x| x.rule == "IMAT-003"));
    }

    #[test]
    fn rejects_non_hex_dataset_hash() {
        let mut obs = build_happy_observation();
        obs["dataset_hash"] = json!("not-a-hash!!");
        let f = lint_imatrix_observation(&obs);
        assert!(f.iter().any(|x| x.rule == "IMAT-006"));
    }
}
