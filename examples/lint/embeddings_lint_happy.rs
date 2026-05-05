//! # Recipe: /v1/embeddings Lint — Happy Path
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr embeddings-lint --observation-file observation.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the OpenAI-compatible `/v1/embeddings` lint pipeline
//! (CRUX-C-13). The response must satisfy seven invariants: top-level
//! `object == "list"`, `data[]` non-empty, each entry's
//! `object == "embedding"`, `index` matches array position, `embedding`
//! is a numeric vector with declared dimension, `model` echoed back, and
//! `usage.prompt_tokens` is a non-negative integer.
//!
//! ## Run Command
//! ```bash
//! cargo run --example embeddings_lint_happy
//! ```
//!
//! ## References
//! - OpenAI Embeddings API spec (platform.openai.com/docs/api-reference/embeddings).
//! - aprender CRUX-C-13 contract.
//!
//! Added by PMAT-092 (expand-cookbooks followup — embeddings/search/grad-norm lint).

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LintFinding {
    pub rule: String,
    pub severity: &'static str,
    pub message: String,
}

pub fn lint_embeddings_response(resp: &Value) -> Vec<LintFinding> {
    let mut out = Vec::new();

    // Rule 1: object == "list".
    match resp.get("object").and_then(Value::as_str) {
        Some("list") => {}
        _ => out.push(LintFinding {
            rule: "EMB-001".into(),
            severity: "error",
            message: "object must be \"list\"".into(),
        }),
    }

    // Rule 2: data[] present and non-empty.
    let data = resp.get("data").and_then(Value::as_array);
    let Some(arr) = data else {
        out.push(LintFinding {
            rule: "EMB-002".into(),
            severity: "error",
            message: "data must be an array".into(),
        });
        return out;
    };
    if arr.is_empty() {
        out.push(LintFinding {
            rule: "EMB-002".into(),
            severity: "error",
            message: "data array must be non-empty".into(),
        });
    }

    let declared_dim = resp.get("dim").and_then(Value::as_u64);

    for (i, e) in arr.iter().enumerate() {
        // Rule 3: each entry object == "embedding"
        match e.get("object").and_then(Value::as_str) {
            Some("embedding") => {}
            _ => out.push(LintFinding {
                rule: "EMB-003".into(),
                severity: "error",
                message: format!("data[{i}].object must be \"embedding\""),
            }),
        }
        // Rule 4: each entry index matches array position
        match e.get("index").and_then(Value::as_u64) {
            Some(idx) if idx as usize == i => {}
            _ => out.push(LintFinding {
                rule: "EMB-004".into(),
                severity: "error",
                message: format!("data[{i}].index must equal {i}"),
            }),
        }
        // Rule 5: embedding is a numeric vector matching declared dim
        let emb = e.get("embedding").and_then(Value::as_array);
        match emb {
            Some(v) => {
                if v.iter().any(|x| x.as_f64().is_none()) {
                    out.push(LintFinding {
                        rule: "EMB-005".into(),
                        severity: "error",
                        message: format!("data[{i}].embedding contains non-numeric entries"),
                    });
                }
                if let Some(d) = declared_dim {
                    if v.len() as u64 != d {
                        out.push(LintFinding {
                            rule: "EMB-005".into(),
                            severity: "error",
                            message: format!(
                                "data[{i}].embedding length {} != declared dim {d}",
                                v.len()
                            ),
                        });
                    }
                }
            }
            None => out.push(LintFinding {
                rule: "EMB-005".into(),
                severity: "error",
                message: format!("data[{i}].embedding must be a numeric array"),
            }),
        }
    }

    // Rule 6: top-level "model" string non-empty.
    match resp.get("model").and_then(Value::as_str) {
        Some(m) if !m.is_empty() => {}
        _ => out.push(LintFinding {
            rule: "EMB-006".into(),
            severity: "error",
            message: "model must be a non-empty string".into(),
        }),
    }

    // Rule 7: usage.prompt_tokens is a non-negative integer.
    match resp.pointer("/usage/prompt_tokens").and_then(Value::as_i64) {
        Some(n) if n >= 0 => {}
        _ => out.push(LintFinding {
            rule: "EMB-007".into(),
            severity: "error",
            message: "usage.prompt_tokens must be a non-negative integer".into(),
        }),
    }

    out
}

pub fn build_happy_response() -> Value {
    json!({
        "object": "list",
        "model": "nomic-embed-text",
        "dim": 4,
        "data": [
            { "object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4] },
            { "object": "embedding", "index": 1, "embedding": [0.5, 0.6, 0.7, 0.8] }
        ],
        "usage": { "prompt_tokens": 12, "total_tokens": 12 }
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("embeddings_lint_happy")?;
    let resp = build_happy_response();

    let resp_path = ctx.path("embeddings_response.json");
    std::fs::write(
        &resp_path,
        serde_json::to_vec_pretty(&resp)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let findings = lint_embeddings_response(&resp);
    let errors = findings.iter().filter(|f| f.severity == "error").count();

    println!("=== Recipe: {} ===", ctx.name());
    println!("Response: {}", resp_path.display());
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
    fn happy_response_has_no_errors() {
        let f = lint_embeddings_response(&build_happy_response());
        assert!(f.is_empty(), "expected clean: {f:?}");
    }

    #[test]
    fn rejects_index_mismatch() {
        let mut resp = build_happy_response();
        resp["data"][1]["index"] = json!(5);
        let f = lint_embeddings_response(&resp);
        assert!(f.iter().any(|x| x.rule == "EMB-004"));
    }

    #[test]
    fn rejects_dim_mismatch() {
        let mut resp = build_happy_response();
        resp["data"][0]["embedding"] = json!([0.1, 0.2, 0.3]); // dim=3 but declared 4
        let f = lint_embeddings_response(&resp);
        assert!(f.iter().any(|x| x.rule == "EMB-005"));
    }

    #[test]
    fn rejects_negative_prompt_tokens() {
        let mut resp = build_happy_response();
        resp["usage"]["prompt_tokens"] = json!(-1);
        let f = lint_embeddings_response(&resp);
        assert!(f.iter().any(|x| x.rule == "EMB-007"));
    }
}
