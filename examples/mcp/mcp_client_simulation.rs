//! # Recipe: MCP Client Simulation — Full Session Transcript
//!
//! **Category**: mcp
//! **CLI Equivalent**: `apr mcp < session.jsonl`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example mcp_client_simulation` exits 0
//! 2. [x] `cargo test --example mcp_client_simulation` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Composition: simulates a full MCP client session in-process. Drives an
//! `initialize → tools/list → tools/call → shutdown` sequence, writes the
//! transcript as NDJSON, and verifies the ID-correlation invariant (every
//! response id matches a prior request id).
//!
//! ## Run Command
//! ```bash
//! cargo run --example mcp_client_simulation
//! ```
//!
//! ## References
//! - Anthropic. *Model Context Protocol Specification*. <https://modelcontextprotocol.io>

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct Exchange {
    pub request: Value,
    pub response: Value,
}

pub fn simulate_server(req: &Value) -> Value {
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let method = req.get("method").and_then(Value::as_str).unwrap_or("");
    match method {
        "initialize" => json!({
            "jsonrpc":"2.0","id":id,
            "result":{
                "protocolVersion":"2024-11-05",
                "serverInfo":{"name":"apr-cookbook-mcp","version":"0.1.0"},
                "capabilities":{"tools":{"listChanged":false}}
            }
        }),
        "tools/list" => json!({
            "jsonrpc":"2.0","id":id,
            "result":{"tools":[
                {"name":"apr.inspect","description":"Inspect a model","inputSchema":{"type":"object"}}
            ]}
        }),
        "tools/call" => {
            let name = req
                .pointer("/params/name")
                .and_then(Value::as_str)
                .unwrap_or("");
            json!({
                "jsonrpc":"2.0","id":id,
                "result":{"content":[{"type":"text","text":format!("executed {name}")}]}
            })
        }
        "shutdown" => json!({"jsonrpc":"2.0","id":id,"result":{}}),
        _ => json!({
            "jsonrpc":"2.0","id":id,
            "error":{"code":-32601,"message":format!("method not found: {method}")}
        }),
    }
}

pub fn run_session() -> Vec<Exchange> {
    let requests = vec![
        json!({"jsonrpc":"2.0","id":1,"method":"initialize"}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
        json!({
            "jsonrpc":"2.0","id":3,"method":"tools/call",
            "params":{"name":"apr.inspect","arguments":{"model_path":"m.apr"}}
        }),
        json!({"jsonrpc":"2.0","id":4,"method":"shutdown"}),
    ];
    requests
        .into_iter()
        .map(|req| {
            let resp = simulate_server(&req);
            Exchange {
                request: req,
                response: resp,
            }
        })
        .collect()
}

pub fn verify_id_correlation(exchanges: &[Exchange]) -> std::result::Result<(), String> {
    let mut seen = HashSet::new();
    for e in exchanges {
        let rid = e.request.get("id").cloned().unwrap_or(Value::Null);
        if !seen.insert(rid.clone()) {
            return Err(format!("duplicate request id: {rid}"));
        }
        if e.response.get("id") != Some(&rid) {
            return Err(format!(
                "response id mismatch: request {rid} vs response {:?}",
                e.response.get("id")
            ));
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("mcp_client_simulation")?;
    let exchanges = run_session();

    let mut ndjson = String::new();
    for e in &exchanges {
        ndjson.push_str(
            &serde_json::to_string(&e.request)
                .map_err(|err| CookbookError::Serialization(err.to_string()))?,
        );
        ndjson.push('\n');
        ndjson.push_str(
            &serde_json::to_string(&e.response)
                .map_err(|err| CookbookError::Serialization(err.to_string()))?,
        );
        ndjson.push('\n');
    }
    let p = ctx.path("session.ndjson");
    std::fs::write(&p, &ndjson)?;

    let ok = verify_id_correlation(&exchanges);
    println!("=== Recipe: {} ===", ctx.name());
    println!("Transcript: {}", p.display());
    for e in &exchanges {
        let method = e
            .request
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or("?");
        let status = if e.response.get("error").is_some() {
            "err"
        } else {
            "ok"
        };
        println!(
            "  [{}] {:<12} id={}",
            status,
            method,
            e.request.get("id").cloned().unwrap_or(Value::Null)
        );
    }
    println!(
        "\nid-correlation: {}",
        ok.as_ref().map_or_else(String::as_str, |()| "OK")
    );

    ctx.record_metric("exchanges", exchanges.len() as i64);
    ctx.record_string_metric("verdict", if ok.is_ok() { "PASS" } else { "FAIL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_has_four_exchanges() {
        assert_eq!(run_session().len(), 4);
    }

    #[test]
    fn id_correlation_valid() {
        let ex = run_session();
        assert!(verify_id_correlation(&ex).is_ok());
    }

    #[test]
    fn unknown_method_returns_rpc_error() {
        let req = json!({"jsonrpc":"2.0","id":99,"method":"does.not.exist"});
        let resp = simulate_server(&req);
        assert_eq!(resp["error"]["code"], -32601);
    }

    #[test]
    fn mismatched_ids_detected() {
        let mut ex = run_session();
        ex[0].response["id"] = json!(999);
        assert!(verify_id_correlation(&ex).is_err());
    }
}
