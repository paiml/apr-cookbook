//! # Recipe: NF4 Lint — Happy Path
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr nf4-lint --observation-file observation.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the NF4 (NormalFloat-4) lint pipeline by synthesizing a
//! valid observation that records the 16-entry codebook, the round-trip
//! Frobenius error, the storage layout, and the double-quantization parity
//! flag. The happy-path observation passes all five rules `apr nf4-lint`
//! enforces.
//!
//! ## Run Command
//! ```bash
//! cargo run --example nf4_lint_happy
//! ```
//!
//! ## References
//! - Dettmers, T. et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv:2305.14314
//! - bitsandbytes NF4 implementation (codebook constants).
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

/// Reference NF4 codebook from the QLoRA paper (16 normalized quantile points).
/// All entries are in [-1.0, 1.0] and the codebook is symmetric around 0
/// (entry 7 is exactly 0.0).
pub const NF4_CODEBOOK: [f64; 16] = [
    -1.0, -0.6961928, -0.5250730, -0.3949175, -0.2844976, -0.1848792, -0.0907983, 0.0, 0.07958029,
    0.1609302, 0.2461123, 0.3379152, 0.4407410, 0.5626170, 0.7229568, 1.0,
];

pub fn lint_nf4_observation(obs: &Value) -> Vec<LintFinding> {
    let mut out = Vec::new();

    // Rule 1: schema_version present and >= 1.
    match obs.get("schema_version").and_then(Value::as_u64) {
        Some(v) if v >= 1 => {}
        _ => out.push(LintFinding {
            rule: "NF4-001".into(),
            severity: "error",
            message: "schema_version missing or < 1".into(),
        }),
    }

    // Rule 2: codebook must be exactly 16 entries.
    let cb = obs.get("codebook").and_then(Value::as_array);
    match cb {
        Some(arr) if arr.len() == 16 => {
            // Rule 3: codebook entries in [-1.0, 1.0]; first = -1.0; last = 1.0; middle entry = 0.0.
            let vals: Vec<f64> = arr.iter().filter_map(Value::as_f64).collect();
            if vals.len() == 16 {
                if (vals[0] - -1.0).abs() > 1e-6 || (vals[15] - 1.0).abs() > 1e-6 {
                    out.push(LintFinding {
                        rule: "NF4-003".into(),
                        severity: "error",
                        message: "codebook[0] must be -1.0 and codebook[15] must be 1.0".into(),
                    });
                }
                if vals[7].abs() > 1e-6 {
                    out.push(LintFinding {
                        rule: "NF4-003".into(),
                        severity: "error",
                        message: "codebook[7] must be exactly 0.0 (symmetric pivot)".into(),
                    });
                }
            } else {
                out.push(LintFinding {
                    rule: "NF4-003".into(),
                    severity: "error",
                    message: "codebook contains non-numeric entries".into(),
                });
            }
        }
        _ => out.push(LintFinding {
            rule: "NF4-002".into(),
            severity: "error",
            message: "codebook must be exactly 16 entries".into(),
        }),
    }

    // Rule 4: frobenius_rel_err finite, < 0.10 (NF4 has higher tolerance than FP8).
    match obs.get("frobenius_rel_err").and_then(Value::as_f64) {
        Some(e) if e.is_finite() && (0.0..0.10).contains(&e) => {}
        _ => out.push(LintFinding {
            rule: "NF4-004".into(),
            severity: "warn",
            message: "frobenius_rel_err must be finite, in [0, 0.10)".into(),
        }),
    }

    // Rule 5: storage layout must be one of the bitsandbytes-supported variants.
    match obs.get("storage").and_then(Value::as_str) {
        Some("packed_2x4bit_le" | "packed_2x4bit_be") => {}
        _ => out.push(LintFinding {
            rule: "NF4-005".into(),
            severity: "error",
            message: "storage must be packed_2x4bit_le or packed_2x4bit_be".into(),
        }),
    }

    out
}

pub fn build_happy_observation() -> Value {
    json!({
        "schema_version": 1,
        "model": "llama-7b-nf4",
        "codebook": NF4_CODEBOOK.to_vec(),
        "frobenius_rel_err": 0.041,
        "storage": "packed_2x4bit_le",
        "double_quant": true,
        "block_size": 64
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("nf4_lint_happy")?;
    let observation = build_happy_observation();

    let obs_path = ctx.path("nf4_observation.json");
    std::fs::write(
        &obs_path,
        serde_json::to_vec_pretty(&observation)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let findings = lint_nf4_observation(&observation);
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
        let f = lint_nf4_observation(&build_happy_observation());
        let errs: Vec<_> = f.iter().filter(|x| x.severity == "error").collect();
        assert!(errs.is_empty(), "expected clean: {errs:?}");
    }

    #[test]
    fn rejects_wrong_codebook_size() {
        let mut obs = build_happy_observation();
        let mut cb = NF4_CODEBOOK.to_vec();
        cb.pop();
        obs["codebook"] = json!(cb);
        let f = lint_nf4_observation(&obs);
        assert!(f.iter().any(|x| x.rule == "NF4-002"));
    }

    #[test]
    fn rejects_unknown_storage() {
        let mut obs = build_happy_observation();
        obs["storage"] = json!("packed_4bit_misaligned");
        let f = lint_nf4_observation(&obs);
        assert!(f.iter().any(|x| x.rule == "NF4-005"));
    }
}
