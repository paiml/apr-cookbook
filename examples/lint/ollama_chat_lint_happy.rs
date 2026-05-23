//! # Recipe: Ollama-Chat Lint — Happy Path
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr ollama-chat-lint response.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example ollama_chat_lint_happy` exits 0
//! 2. [x] `cargo test --example ollama_chat_lint_happy` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Validates a non-streaming `/api/chat` response from Ollama against its
//! documented schema: the envelope (`model`, `created_at`, `done`,
//! `total_duration`) and the message payload (`role`, `content`). This is the
//! happy-path — one response, everything populated correctly.
//!
//! ## Run Command
//! ```bash
//! cargo run --example ollama_chat_lint_happy
//! ```
//!
//! ## References
//! - Ollama. *API Reference: /api/chat*. <https://github.com/ollama/ollama/blob/main/docs/api.md>
//! - Vaswani, A. et al. (2017). *Attention Is All You Need*. arXiv:1706.03762

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub rule: &'static str,
    pub severity: &'static str,
    pub message: String,
}

pub fn lint_response(resp: &Value) -> Vec<Finding> {
    let mut out = Vec::new();
    for field in ["model", "created_at", "message", "done"] {
        if resp.get(field).is_none() {
            out.push(Finding {
                rule: "OLL-001",
                severity: "error",
                message: format!("missing required field `{field}`"),
            });
        }
    }
    if let Some(role) = resp.pointer("/message/role").and_then(Value::as_str) {
        if role != "assistant" {
            out.push(Finding {
                rule: "OLL-002",
                severity: "error",
                message: format!("message.role must be `assistant` (got `{role}`)"),
            });
        }
    } else {
        out.push(Finding {
            rule: "OLL-002",
            severity: "error",
            message: "message.role missing".into(),
        });
    }
    if !resp
        .pointer("/message/content")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.is_empty())
    {
        out.push(Finding {
            rule: "OLL-003",
            severity: "error",
            message: "message.content must be a non-empty string".into(),
        });
    }
    if let Some(d) = resp.get("total_duration").and_then(Value::as_u64) {
        if d == 0 {
            out.push(Finding {
                rule: "OLL-004",
                severity: "warn",
                message: "total_duration == 0 is suspicious".into(),
            });
        }
    }
    out
}

fn build_happy() -> Value {
    json!({
        "model": "llama3.1:8b",
        "created_at": "2026-04-22T12:00:00Z",
        "message": {
            "role": "assistant",
            "content": "The Toyota Production System emphasises muda elimination."
        },
        "done": true,
        "total_duration": 123_456_789u64,
        "eval_count": 42,
        "prompt_eval_count": 17
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("ollama_chat_lint_happy")?;
    let r = build_happy();
    let p = ctx.path("response.json");
    std::fs::write(
        &p,
        serde_json::to_vec_pretty(&r).map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let findings = lint_response(&r);
    let errors = findings.iter().filter(|f| f.severity == "error").count();

    println!("=== Recipe: {} ===", ctx.name());
    println!("Response: {}", p.display());
    println!("Findings: {} (errors: {})", findings.len(), errors);
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
    fn happy_is_clean() {
        assert!(lint_response(&build_happy()).is_empty());
    }

    #[test]
    fn missing_model_flags_oll_001() {
        let mut r = build_happy();
        r.as_object_mut().unwrap().remove("model");
        assert!(lint_response(&r).iter().any(|f| f.rule == "OLL-001"));
    }

    #[test]
    fn wrong_role_flags_oll_002() {
        let mut r = build_happy();
        r["message"]["role"] = json!("user");
        assert!(lint_response(&r).iter().any(|f| f.rule == "OLL-002"));
    }
}
