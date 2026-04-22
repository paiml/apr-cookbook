//! # Recipe: Tool-Use Lint — Happy Path
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr tool-use-lint response.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example tool_use_lint_happy` exits 0
//! 2. [x] `cargo test --example tool_use_lint_happy` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Validates an OpenAI-shape chat completion containing a tool call. The
//! response must be valid JSON, have `finish_reason == "tool_calls"`, and
//! every `tool_calls[*]` entry must have a parseable-JSON `arguments`
//! payload that matches a declared `parameters` schema.
//!
//! ## Run Command
//! ```bash
//! cargo run --example tool_use_lint_happy
//! ```
//!
//! ## References
//! - Schick, T. et al. (2023). *Toolformer: Language Models Can Teach Themselves to Use Tools*. arXiv:2302.04761

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    pub rule: &'static str,
    pub severity: &'static str,
    pub message: String,
}

pub fn lint_tool_use(resp: &Value, required_args: &[&str]) -> Vec<Finding> {
    let mut out = Vec::new();
    let Some(choice) = resp.pointer("/choices/0") else {
        out.push(Finding {
            rule: "TOOL-001",
            severity: "error",
            message: "missing choices[0]".into(),
        });
        return out;
    };
    match choice.get("finish_reason").and_then(Value::as_str) {
        Some("tool_calls") => {}
        other => out.push(Finding {
            rule: "TOOL-002",
            severity: "error",
            message: format!("finish_reason must be `tool_calls` (got {:?})", other),
        }),
    }
    let calls = match choice
        .pointer("/message/tool_calls")
        .and_then(Value::as_array)
    {
        Some(c) if !c.is_empty() => c,
        _ => {
            out.push(Finding {
                rule: "TOOL-003",
                severity: "error",
                message: "message.tool_calls missing or empty".into(),
            });
            return out;
        }
    };
    for (i, c) in calls.iter().enumerate() {
        let args_raw = c.pointer("/function/arguments").and_then(Value::as_str);
        let Some(s) = args_raw else {
            out.push(Finding {
                rule: "TOOL-004",
                severity: "error",
                message: format!(
                    "tool_calls[{i}].function.arguments missing (must be JSON-encoded string)"
                ),
            });
            continue;
        };
        let parsed: std::result::Result<Value, _> = serde_json::from_str(s);
        let Ok(obj) = parsed else {
            out.push(Finding {
                rule: "TOOL-005",
                severity: "error",
                message: format!("tool_calls[{i}].arguments is not valid JSON"),
            });
            continue;
        };
        for arg in required_args {
            if obj.get(arg).is_none() {
                out.push(Finding {
                    rule: "TOOL-006",
                    severity: "error",
                    message: format!("tool_calls[{i}] missing required arg `{arg}`"),
                });
            }
        }
    }
    out
}

fn build_happy() -> Value {
    json!({
        "id": "chatcmpl-1",
        "model": "gpt-4o-mini",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": "{\"city\":\"Tokyo\",\"units\":\"metric\"}"
                    }
                }]
            }
        }]
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("tool_use_lint_happy")?;
    let r = build_happy();
    let p = ctx.path("tool_use.json");
    std::fs::write(
        &p,
        serde_json::to_vec_pretty(&r).map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let findings = lint_tool_use(&r, &["city", "units"]);
    let errors = findings.iter().filter(|f| f.severity == "error").count();

    println!("=== Recipe: {} ===", ctx.name());
    println!("Response: {}", p.display());
    println!("Findings: {}", findings.len());
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
        assert!(lint_tool_use(&build_happy(), &["city", "units"]).is_empty());
    }

    #[test]
    fn missing_required_arg_flags_tool_006() {
        let f = lint_tool_use(&build_happy(), &["city", "temperature"]);
        assert!(f.iter().any(|x| x.rule == "TOOL-006"));
    }

    #[test]
    fn rejects_wrong_finish_reason() {
        let mut r = build_happy();
        r["choices"][0]["finish_reason"] = json!("stop");
        let f = lint_tool_use(&r, &["city", "units"]);
        assert!(f.iter().any(|x| x.rule == "TOOL-002"));
    }
}
