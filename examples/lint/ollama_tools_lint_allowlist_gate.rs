//! # Recipe: Ollama Tools Lint — Allowlist Gate
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr ollama-tools-lint --response-file resp.json --request-file req.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates the `--request-file` allowlist gate. When the operator
//! provides the original `/api/chat` request alongside the response, the
//! lint cross-checks every called tool name against the
//! `request.tools[*].function.name` allowlist. A model that hallucinates a
//! tool name not declared in the request is a security/safety regression
//! and the lint elevates it to error severity.
//!
//! ## Run Command
//! ```bash
//! cargo run --example ollama_tools_lint_allowlist_gate
//! ```
//!
//! ## References
//! - aprender CRUX-I-04 (allowlist invariant).
//! - Ollama tools API (`tools` field accepted at request time).
//!
//! Added by PMAT-091 (expand-cookbooks followup — Ollama/sampling/imatrix lint).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};
use std::collections::HashSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllowlistFinding {
    pub call_index: usize,
    pub called_name: String,
    pub allowed: HashSet<String>,
}

pub fn check_allowlist(req: &Value, resp: &Value) -> Vec<AllowlistFinding> {
    let allowed: HashSet<String> = req
        .get("tools")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|t| t.pointer("/function/name").and_then(Value::as_str))
                .map(String::from)
                .collect()
        })
        .unwrap_or_default();

    let calls = resp
        .pointer("/message/tool_calls")
        .and_then(Value::as_array);
    let Some(calls) = calls else {
        return Vec::new();
    };

    let mut out = Vec::new();
    for (i, c) in calls.iter().enumerate() {
        let Some(name) = c.pointer("/function/name").and_then(Value::as_str) else {
            continue;
        };
        if !allowed.contains(name) {
            out.push(AllowlistFinding {
                call_index: i,
                called_name: name.into(),
                allowed: allowed.clone(),
            });
        }
    }
    out
}

fn build_request_with_two_tools() -> Value {
    json!({
        "model": "llama3.1:8b",
        "messages": [{ "role": "user", "content": "What is the weather in Berkeley?" }],
        "tools": [
            { "type": "function", "function": { "name": "get_current_weather", "description": "..." } },
            { "type": "function", "function": { "name": "get_forecast",        "description": "..." } }
        ]
    })
}

fn build_response_with_hallucinated_tool() -> Value {
    json!({
        "model": "llama3.1:8b",
        "created_at": "2026-05-05T12:00:00.000Z",
        "done": true,
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                { "function": { "name": "get_current_weather", "arguments": { "city": "Berkeley" } } },
                { "function": { "name": "drop_database",        "arguments": {} } } // ⚠ not allowed
            ]
        }
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("ollama_tools_lint_allowlist_gate")?;
    let req = build_request_with_two_tools();
    let resp = build_response_with_hallucinated_tool();

    let findings = check_allowlist(&req, &resp);

    println!("=== Recipe: {} ===", ctx.name());
    println!("hallucinated tool calls: {}", findings.len());
    for f in &findings {
        println!(
            "  call[{}] called {:?} not in allowlist {:?}",
            f.call_index, f.called_name, f.allowed
        );
    }
    ctx.record_metric("hallucinations", findings.len() as i64);
    ctx.record_string_metric("verdict", if findings.is_empty() { "PASS" } else { "FAIL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowlist_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn allowed_tool_passes() {
        let req = build_request_with_two_tools();
        let resp = json!({
            "message": { "tool_calls": [
                { "function": { "name": "get_current_weather" } }
            ]}
        });
        assert!(check_allowlist(&req, &resp).is_empty());
    }

    #[test]
    fn hallucinated_tool_flagged() {
        let req = build_request_with_two_tools();
        let resp = build_response_with_hallucinated_tool();
        let f = check_allowlist(&req, &resp);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].called_name, "drop_database");
    }

    #[test]
    fn missing_tool_calls_array_yields_no_findings() {
        // No tool calls = no allowlist work.
        let req = build_request_with_two_tools();
        let resp = json!({ "message": { "role": "assistant", "content": "" } });
        assert!(check_allowlist(&req, &resp).is_empty());
    }

    #[test]
    fn empty_allowlist_flags_all_calls() {
        // Request with no `tools` field → every called tool is hallucinated.
        let req = json!({});
        let resp = build_response_with_hallucinated_tool();
        let f = check_allowlist(&req, &resp);
        assert_eq!(f.len(), 2);
    }
}
