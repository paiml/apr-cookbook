//! # Recipe: imatrix Lint — NaN/Inf Detection
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr imatrix-lint --observation-file observation.json` (NaN path)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates NaN/Inf detection in importance vectors. A NaN here means
//! the calibration pass divided by a zero variance (an entire activation
//! channel collapsed); an inf means the running max overflowed (FP16
//! arithmetic on extreme outliers). Both poison every downstream k-quant
//! scale that consults this importance row, so the lint must report the
//! exact `(tensor, index, kind)` triple for surgical re-calibration.
//!
//! ## Run Command
//! ```bash
//! cargo run --example imatrix_lint_nan_detection
//! ```
//!
//! ## References
//! - llama.cpp imatrix-tool source (per-channel running stats).
//! - IEEE 754-2019 §6 (NaN/inf semantics).
//!
//! Added by PMAT-091 (expand-cookbooks followup — Ollama/sampling/imatrix lint).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumericKind {
    Nan,
    PosInf,
    NegInf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoisonFinding {
    pub tensor: String,
    pub index: usize,
    pub kind: NumericKind,
}

pub fn detect_poison(obs: &Value) -> Vec<PoisonFinding> {
    let mut out = Vec::new();
    let Some(entries) = obs.get("per_tensor").and_then(Value::as_array) else {
        return out;
    };
    for e in entries {
        let name = e
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string();
        let Some(arr) = e.get("importance").and_then(Value::as_array) else {
            continue;
        };
        for (i, v) in arr.iter().enumerate() {
            let f = v.as_f64().unwrap_or(f64::NAN);
            if f.is_nan() {
                out.push(PoisonFinding {
                    tensor: name.clone(),
                    index: i,
                    kind: NumericKind::Nan,
                });
            } else if f == f64::INFINITY {
                out.push(PoisonFinding {
                    tensor: name.clone(),
                    index: i,
                    kind: NumericKind::PosInf,
                });
            } else if f == f64::NEG_INFINITY {
                out.push(PoisonFinding {
                    tensor: name.clone(),
                    index: i,
                    kind: NumericKind::NegInf,
                });
            }
        }
    }
    out
}

fn build_poisoned_observation() -> Value {
    // serde_json represents NaN/inf as Null when rendered, so use a fixture
    // that exercises the f64::NAN / inf path through the as_f64 unwrap_or.
    json!({
        "per_tensor": [
            { "name": "blk.0.attn_q.weight", "importance": [0.42, 0.31, null] },         // NaN via null
            { "name": "blk.0.attn_k.weight", "importance": [0.51, 0.27, 0.12] },         // clean
            { "name": "blk.0.attn_v.weight", "importance": [1e308, 1e308, "infinity"] }  // bogus string
        ]
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("imatrix_lint_nan_detection")?;
    let obs = build_poisoned_observation();
    let findings = detect_poison(&obs);

    println!("=== Recipe: {} ===", ctx.name());
    println!("poisoned entries: {}", findings.len());
    for f in &findings {
        println!("  {} index={} {:?}", f.tensor, f.index, f.kind);
    }

    ctx.record_metric("poisoned_entries", findings.len() as i64);
    ctx.record_string_metric("verdict", if findings.is_empty() { "PASS" } else { "FAIL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nan_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn null_value_in_importance_is_nan() {
        // serde_json::Value::Null → as_f64() = None → unwrap_or(NaN).
        let obs = json!({
            "per_tensor": [{ "name": "x", "importance": [null] }]
        });
        let f = detect_poison(&obs);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].kind, NumericKind::Nan);
    }

    #[test]
    fn string_in_importance_is_nan() {
        // Producer bug: emit "infinity" string instead of f64. Caught as NaN.
        let obs = json!({
            "per_tensor": [{ "name": "x", "importance": ["infinity"] }]
        });
        let f = detect_poison(&obs);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].kind, NumericKind::Nan);
    }

    #[test]
    fn clean_observation_returns_empty() {
        let obs = json!({
            "per_tensor": [{ "name": "x", "importance": [0.1, 0.2, 0.3] }]
        });
        assert!(detect_poison(&obs).is_empty());
    }

    #[test]
    fn missing_per_tensor_array_returns_empty() {
        // Different rule (IMAT-001/002) covers this case — no double-flag.
        assert!(detect_poison(&json!({})).is_empty());
    }

    #[test]
    fn finding_preserves_tensor_and_index_for_surgical_recalibration() {
        // The whole point of this rule: tell the operator exactly which
        // (tensor, channel) needs re-calibration. Don't lose the locator.
        let obs = json!({
            "per_tensor": [
                { "name": "blk.5.ffn_up.weight", "importance": [0.0, null, 0.0] }
            ]
        });
        let f = detect_poison(&obs);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].tensor, "blk.5.ffn_up.weight");
        assert_eq!(f[0].index, 1);
    }
}
