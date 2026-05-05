//! # Recipe: Embeddings Lint — L2 Normalization Check
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr embeddings-lint --observation-file observation.json` (norm path)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates L2-norm verification. Embedding models advertise either
//! unit-normalized output (most retrieval models) or unbounded output
//! (some BGE variants). When the response declares `normalized=true`, the
//! lint must confirm `||v||₂ ∈ [1.0 - tol, 1.0 + tol]` for every entry,
//! tolerance defaulting to 1e-3 (FP32 round-trip noise). When
//! `normalized=false`, the lint instead checks `||v||₂ > 0` (degenerate
//! all-zeros embedding is always a producer bug).
//!
//! ## Run Command
//! ```bash
//! cargo run --example embeddings_lint_l2_norm_check
//! ```
//!
//! ## References
//! - Reimers & Gurevych (2019). *Sentence-BERT*. arXiv:1908.10084 (L2-normalized embeddings).
//! - aprender CRUX-C-13 (normalization invariant).
//!
//! Added by PMAT-092 (expand-cookbooks followup — embeddings/search/grad-norm lint).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub enum NormFinding {
    NotUnit { index: usize, norm: f64 },
    Degenerate { index: usize },
}

const UNIT_TOL: f64 = 1e-3;

pub fn check_norms(resp: &Value) -> Vec<NormFinding> {
    let mut out = Vec::new();
    let normalized = resp
        .get("normalized")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let Some(arr) = resp.get("data").and_then(Value::as_array) else {
        return out;
    };
    for (i, e) in arr.iter().enumerate() {
        let Some(v) = e.get("embedding").and_then(Value::as_array) else {
            continue;
        };
        let sq: f64 = v.iter().filter_map(Value::as_f64).map(|x| x * x).sum();
        let norm = sq.sqrt();
        if normalized {
            if (norm - 1.0).abs() > UNIT_TOL {
                out.push(NormFinding::NotUnit { index: i, norm });
            }
        } else if norm == 0.0 {
            out.push(NormFinding::Degenerate { index: i });
        }
    }
    out
}

fn build_unit_response() -> Value {
    // [0.6, 0.8] → norm = 1.0
    json!({
        "normalized": true,
        "data": [
            { "index": 0, "embedding": [0.6, 0.8] },
            { "index": 1, "embedding": [0.0, 1.0] }
        ]
    })
}

fn build_off_unit_response() -> Value {
    json!({
        "normalized": true,
        "data": [
            { "index": 0, "embedding": [1.0, 1.0] }, // norm = sqrt(2) ≠ 1
            { "index": 1, "embedding": [0.6, 0.8] }
        ]
    })
}

fn build_degenerate_response() -> Value {
    json!({
        "normalized": false,
        "data": [
            { "index": 0, "embedding": [0.0, 0.0, 0.0] },
            { "index": 1, "embedding": [1.0, 2.0, 3.0] }
        ]
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("embeddings_lint_l2_norm_check")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("unit:        {:?}", check_norms(&build_unit_response()));
    println!("off-unit:    {:?}", check_norms(&build_off_unit_response()));
    println!(
        "degenerate:  {:?}",
        check_norms(&build_degenerate_response())
    );

    ctx.record_string_metric("verdict", "matrix_printed");
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn unit_response_has_no_findings() {
        let f = check_norms(&build_unit_response());
        assert!(f.is_empty(), "expected clean: {f:?}");
    }

    #[test]
    fn off_unit_flagged_when_normalized() {
        let f = check_norms(&build_off_unit_response());
        assert_eq!(f.len(), 1);
        assert!(matches!(f[0], NormFinding::NotUnit { index: 0, .. }));
    }

    #[test]
    fn degenerate_flagged_only_when_unnormalized() {
        let f = check_norms(&build_degenerate_response());
        assert_eq!(f.len(), 1);
        assert!(matches!(f[0], NormFinding::Degenerate { index: 0 }));
    }

    #[test]
    fn off_unit_unnormalized_not_flagged() {
        // When `normalized=false`, [1.0, 1.0] is fine — norm just isn't 1.
        let resp = json!({
            "normalized": false,
            "data": [{ "index": 0, "embedding": [1.0, 1.0] }]
        });
        assert!(check_norms(&resp).is_empty());
    }

    #[test]
    fn within_tolerance_passes() {
        // norm = 1.0005 is within ±1e-3 — must not flag.
        let resp = json!({
            "normalized": true,
            "data": [{ "index": 0, "embedding": [0.6003, 0.8004] }]
        });
        let f = check_norms(&resp);
        assert!(f.is_empty(), "near-unit should pass: {f:?}");
    }
}
