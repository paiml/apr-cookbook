//! # Recipe: Ollama Tools Lint — Happy Path
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr ollama-tools-lint --response-file response.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the Ollama function-calling lint pipeline (CRUX-I-04). The
//! response from `/api/chat` is checked for the seven invariants the real
//! `apr ollama-tools-lint` enforces: top-level `model`, `created_at`,
//! `done` boolean, `message.role == "assistant"`, optional `tool_calls`
//! array structure, each tool call has `function.name` + `function.arguments`,
//! and arguments parse as a JSON object.
//!
//! ## Run Command
//! ```bash
//! cargo run --example ollama_tools_lint_happy
//! ```
//!
//! ## References
//! - aprender CRUX-I-04 contract (Ollama tools observation).
//! - Ollama API docs (github.com/ollama/ollama/blob/main/docs/api.md#chat).
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

pub fn lint_ollama_tools_response(resp: &Value) -> Vec<LintFinding> {
    let mut out = Vec::new();

    // Rule 1: top-level "model" string non-empty.
    match resp.get("model").and_then(Value::as_str) {
        Some(m) if !m.is_empty() => {}
        _ => out.push(LintFinding {
            rule: "OLT-001".into(),
            severity: "error",
            message: "model must be a non-empty string".into(),
        }),
    }

    // Rule 2: top-level "created_at" string present.
    if resp.get("created_at").and_then(Value::as_str).is_none() {
        out.push(LintFinding {
            rule: "OLT-002".into(),
            severity: "error",
            message: "created_at must be present (RFC 3339 timestamp)".into(),
        });
    }

    // Rule 3: top-level "done" boolean present.
    if resp.get("done").and_then(Value::as_bool).is_none() {
        out.push(LintFinding {
            rule: "OLT-003".into(),
            severity: "error",
            message: "done must be a boolean".into(),
        });
    }

    // Rule 4: message.role == "assistant".
    match resp.pointer("/message/role").and_then(Value::as_str) {
        Some("assistant") => {}
        _ => out.push(LintFinding {
            rule: "OLT-004".into(),
            severity: "error",
            message: "message.role must be \"assistant\"".into(),
        }),
    }

    // Rule 5: if tool_calls present, must be an array.
    if let Some(tc) = resp.pointer("/message/tool_calls") {
        if !tc.is_array() {
            out.push(LintFinding {
                rule: "OLT-005".into(),
                severity: "error",
                message: "message.tool_calls must be an array".into(),
            });
        } else if let Some(arr) = tc.as_array() {
            // Rule 6: each tool call has function.name + function.arguments.
            for (i, call) in arr.iter().enumerate() {
                let name = call.pointer("/function/name").and_then(Value::as_str);
                if name.is_none() {
                    out.push(LintFinding {
                        rule: "OLT-006".into(),
                        severity: "error",
                        message: format!("tool_calls[{i}].function.name missing"),
                    });
                }
                // Rule 7: function.arguments must be a JSON object (not stringified).
                let args = call.pointer("/function/arguments");
                match args {
                    Some(v) if v.is_object() => {}
                    Some(_) => out.push(LintFinding {
                        rule: "OLT-007".into(),
                        severity: "error",
                        message: format!(
                            "tool_calls[{i}].function.arguments must be an object, not stringified"
                        ),
                    }),
                    None => out.push(LintFinding {
                        rule: "OLT-007".into(),
                        severity: "error",
                        message: format!("tool_calls[{i}].function.arguments missing"),
                    }),
                }
            }
        }
    }

    out
}

pub fn build_happy_response() -> Value {
    json!({
        "model": "llama3.1:8b",
        "created_at": "2026-05-05T12:00:00.000Z",
        "done": true,
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_current_weather",
                        "arguments": { "city": "Berkeley", "unit": "fahrenheit" }
                    }
                }
            ]
        }
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("ollama_tools_lint_happy")?;
    let resp = build_happy_response();

    let resp_path = ctx.path("ollama_tools_response.json");
    std::fs::write(
        &resp_path,
        serde_json::to_vec_pretty(&resp)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let findings = lint_ollama_tools_response(&resp);
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
        let f = lint_ollama_tools_response(&build_happy_response());
        assert!(f.is_empty(), "expected clean: {f:?}");
    }

    #[test]
    fn rejects_stringified_arguments() {
        // Common producer bug: emit "arguments" as a JSON string instead of an
        // inline object. Ollama clients then double-decode and lose type info.
        let mut resp = build_happy_response();
        resp["message"]["tool_calls"][0]["function"]["arguments"] =
            json!("{\"city\": \"Berkeley\"}");
        let f = lint_ollama_tools_response(&resp);
        assert!(f.iter().any(|x| x.rule == "OLT-007"));
    }

    #[test]
    fn rejects_user_role() {
        let mut resp = build_happy_response();
        resp["message"]["role"] = json!("user");
        let f = lint_ollama_tools_response(&resp);
        assert!(f.iter().any(|x| x.rule == "OLT-004"));
    }

    #[test]
    fn missing_function_name_flagged_per_call() {
        let mut resp = build_happy_response();
        resp["message"]["tool_calls"][0]["function"]
            .as_object_mut()
            .unwrap()
            .remove("name");
        let f = lint_ollama_tools_response(&resp);
        assert!(f.iter().any(|x| x.rule == "OLT-006"));
    }
}
