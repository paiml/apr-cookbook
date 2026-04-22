//! # Recipe: Ollama-Chat Lint — Schema Violation
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr ollama-chat-lint response.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example ollama_chat_lint_schema_violation` exits 0
//! 2. [x] `cargo test --example ollama_chat_lint_schema_violation` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Covers the deliberate-violation edge case: a proxy rewrites `/api/chat`
//! output into OpenAI-chat shape (`choices[0].message` rather than
//! `message`). The lint catches the missing fields and the `role: "system"`
//! anti-pattern that sometimes leaks through buggy proxies.
//!
//! ## Run Command
//! ```bash
//! cargo run --example ollama_chat_lint_schema_violation
//! ```
//!
//! ## References
//! - Ollama. *API Reference: /api/chat*. <https://github.com/ollama/ollama/blob/main/docs/api.md>

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct Finding {
    pub rule: &'static str,
    pub severity: &'static str,
    pub message: String,
}

pub fn lint_response(resp: &Value) -> Vec<Finding> {
    let mut out = Vec::new();
    for field in ["model", "message", "done"] {
        if resp.get(field).is_none() {
            out.push(Finding {
                rule: "OLL-001",
                severity: "error",
                message: format!("missing required field `{field}`"),
            });
        }
    }
    match resp.pointer("/message/role").and_then(Value::as_str) {
        Some("assistant") => {}
        Some(other) => out.push(Finding {
            rule: "OLL-002",
            severity: "error",
            message: format!("message.role must be `assistant` (got `{other}`)"),
        }),
        None => {
            // Only emit if `message` itself is present (otherwise OLL-001 covers it).
            if resp.get("message").is_some() {
                out.push(Finding {
                    rule: "OLL-002",
                    severity: "error",
                    message: "message.role missing".into(),
                });
            }
        }
    }
    if resp
        .pointer("/message/content")
        .and_then(Value::as_str)
        .is_some_and(str::is_empty)
    {
        out.push(Finding {
            rule: "OLL-003",
            severity: "error",
            message: "message.content is empty".into(),
        });
    }
    // OpenAI-shaped leak: has `choices` but no `message`.
    if resp.get("choices").is_some() && resp.get("message").is_none() {
        out.push(Finding {
            rule: "OLL-005",
            severity: "error",
            message: "response looks OpenAI-shaped (has `choices`, missing `message`)".into(),
        });
    }
    out
}

fn build_openai_shaped() -> Value {
    json!({
        "model": "llama3.1:8b",
        "created_at": "2026-04-22T12:00:00Z",
        "done": true,
        // BUG: proxy rewrote the envelope into OpenAI shape.
        "choices": [{
            "index": 0,
            "message": {"role": "system", "content": ""},
            "finish_reason": "stop"
        }]
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("ollama_chat_lint_schema_violation")?;
    let r = build_openai_shaped();
    let p = ctx.path("response.json");
    std::fs::write(
        &p,
        serde_json::to_vec_pretty(&r).map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let findings = lint_response(&r);
    let errors = findings.iter().filter(|f| f.severity == "error").count();

    println!("=== Recipe: {} ===", ctx.name());
    println!("Response: {}", p.display());
    println!("Expected: proxy-shape rewrite triggers OLL-001 + OLL-005");
    for f in &findings {
        println!("  [{}] {} — {}", f.severity, f.rule, f.message);
    }
    println!("Total errors: {}", errors);

    ctx.record_metric("errors", errors as i64);
    ctx.record_string_metric("verdict", if errors >= 2 { "DETECTED" } else { "MISSED" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_shape_detected() {
        let f = lint_response(&build_openai_shaped());
        assert!(f.iter().any(|x| x.rule == "OLL-005"));
    }

    #[test]
    fn missing_message_emits_oll_001() {
        let f = lint_response(&build_openai_shaped());
        assert!(f
            .iter()
            .any(|x| x.rule == "OLL-001" && x.message.contains("message")));
    }

    #[test]
    fn valid_response_clean() {
        let v = json!({
            "model": "llama3.1",
            "message": {"role": "assistant", "content": "hi"},
            "done": true
        });
        assert!(lint_response(&v).is_empty());
    }
}
