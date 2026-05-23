//! # Recipe: MCP Tool Discovery
//!
//! **Category**: mcp
//! **CLI Equivalent**: `apr mcp` (after client sends `tools/list`)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example mcp_tool_discovery` exits 0
//! 2. [x] `cargo test --example mcp_tool_discovery` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Simulates the `tools/list` + `tools/call` exchange. The `apr mcp` server
//! exposes tools like `apr.inspect`, `apr.bench`, `apr.qa`. This recipe
//! shows the expected tool metadata schema, validates JSON-schema-shaped
//! `inputSchema` fragments, and demonstrates an edge-case tool-call failure.
//!
//! ## Run Command
//! ```bash
//! cargo run --example mcp_tool_discovery
//! ```
//!
//! ## References
//! - Anthropic. *Model Context Protocol: Tools*. <https://modelcontextprotocol.io>

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name: &'static str,
    pub description: &'static str,
    pub required: Vec<&'static str>,
}

pub fn tool_catalog() -> Vec<ToolDef> {
    vec![
        ToolDef {
            name: "apr.inspect",
            description: "Inspect an .apr model and return metadata.",
            required: vec!["model_path"],
        },
        ToolDef {
            name: "apr.bench",
            description: "Benchmark a model (tokens/sec, p99 latency).",
            required: vec!["model_path"],
        },
        ToolDef {
            name: "apr.qa",
            description: "Run the falsifiable QA checklist on a model.",
            required: vec!["model_path"],
        },
    ]
}

pub fn tools_list_response(id: &Value, catalog: &[ToolDef]) -> Value {
    let tools: Vec<Value> = catalog
        .iter()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "inputSchema": {
                    "type": "object",
                    "properties": t.required.iter().map(|r| ((*r).to_string(), json!({"type":"string"}))).collect::<serde_json::Map<String, Value>>(),
                    "required": t.required,
                }
            })
        })
        .collect();
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": { "tools": tools }
    })
}

pub fn validate_tool_call(req: &Value, catalog: &[ToolDef]) -> std::result::Result<(), String> {
    let name = req
        .pointer("/params/name")
        .and_then(Value::as_str)
        .ok_or("missing params.name")?;
    let def = catalog
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| format!("unknown tool: {name}"))?;
    let args = req
        .pointer("/params/arguments")
        .cloned()
        .unwrap_or(Value::Null);
    for r in &def.required {
        if args.get(*r).is_none() {
            return Err(format!("missing required argument `{}`", r));
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("mcp_tool_discovery")?;
    let catalog = tool_catalog();

    let list_req = json!({"jsonrpc":"2.0","id":1,"method":"tools/list"});
    let list_resp = tools_list_response(&list_req["id"], &catalog);

    let ok_call = json!({
        "jsonrpc":"2.0","id":2,"method":"tools/call",
        "params":{"name":"apr.inspect","arguments":{"model_path":"mistral-7b.apr"}}
    });
    let bad_call = json!({
        "jsonrpc":"2.0","id":3,"method":"tools/call",
        "params":{"name":"apr.bench","arguments":{}}
    });

    println!("=== Recipe: {} ===", ctx.name());
    println!("Tools exposed: {}", catalog.len());
    for t in &catalog {
        println!("  {:<12} — {}", t.name, t.description);
    }

    let ok_result = validate_tool_call(&ok_call, &catalog);
    let bad_result = validate_tool_call(&bad_call, &catalog);
    println!("\n>> valid call (apr.inspect) -> {:?}", ok_result.is_ok());
    println!(
        ">> invalid call (apr.bench, no model_path) -> {:?}",
        bad_result
    );

    // Save artefacts
    std::fs::write(
        ctx.path("tools_list_resp.json"),
        serde_json::to_vec_pretty(&list_resp)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;
    std::fs::write(
        ctx.path("tools_call_ok.json"),
        serde_json::to_vec_pretty(&ok_call)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;
    std::fs::write(
        ctx.path("tools_call_bad.json"),
        serde_json::to_vec_pretty(&bad_call)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    ctx.record_metric("tool_count", catalog.len() as i64);
    ctx.record_string_metric(
        "verdict",
        if ok_result.is_ok() && bad_result.is_err() {
            "PASS"
        } else {
            "FAIL"
        },
    );
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tools_list_has_three() {
        let c = tool_catalog();
        assert_eq!(c.len(), 3);
    }

    #[test]
    fn valid_call_passes() {
        let c = tool_catalog();
        let req = json!({
            "params":{"name":"apr.inspect","arguments":{"model_path":"m.apr"}}
        });
        assert!(validate_tool_call(&req, &c).is_ok());
    }

    #[test]
    fn unknown_tool_fails() {
        let c = tool_catalog();
        let req = json!({"params":{"name":"apr.unknown","arguments":{}}});
        assert!(validate_tool_call(&req, &c).is_err());
    }

    #[test]
    fn missing_required_arg_fails() {
        let c = tool_catalog();
        let req = json!({"params":{"name":"apr.bench","arguments":{}}});
        let err = validate_tool_call(&req, &c).unwrap_err();
        assert!(err.contains("model_path"));
    }
}
