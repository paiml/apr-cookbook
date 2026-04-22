//! # Recipe: MCP stdio Server Handshake
//!
//! **Category**: mcp
//! **CLI Equivalent**: `apr mcp`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example mcp_stdio_server` exits 0
//! 2. [x] `cargo test --example mcp_stdio_server` passes
//! 3. [x] Deterministic output (same seed → same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Simulates an MCP (Model Context Protocol) server running over stdio. The
//! server framing is line-delimited JSON-RPC 2.0 messages. This recipe
//! synthesises a client `initialize` request and emits the server reply,
//! matching the format `apr mcp` uses over real stdin/stdout.
//!
//! ## Run Command
//! ```bash
//! cargo run --example mcp_stdio_server
//! ```
//!
//! ## References
//! - Anthropic. *Model Context Protocol Specification*. <https://modelcontextprotocol.io>

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::{json, Value};

pub const SERVER_NAME: &str = "apr-cookbook-mcp";
pub const SERVER_VERSION: &str = "0.1.0";
pub const PROTOCOL_VERSION: &str = "2024-11-05";

/// Handle a single JSON-RPC 2.0 request and produce the reply.
pub fn handle_request(req: &Value) -> Value {
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    let method = req.get("method").and_then(Value::as_str).unwrap_or("");
    match method {
        "initialize" => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {
                    "tools": { "listChanged": false },
                    "resources": { "subscribe": false },
                    "prompts": { "listChanged": false }
                },
                "serverInfo": {
                    "name": SERVER_NAME,
                    "version": SERVER_VERSION
                }
            }
        }),
        "ping" => json!({ "jsonrpc": "2.0", "id": id, "result": {} }),
        _ => json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": -32601, "message": format!("method not found: {method}") }
        }),
    }
}

/// Validate a JSON-RPC 2.0 envelope (but not the method-specific payload).
pub fn validate_envelope(v: &Value) -> std::result::Result<(), &'static str> {
    if v.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err("jsonrpc field missing or not \"2.0\"");
    }
    if v.get("method").is_none() && v.get("result").is_none() && v.get("error").is_none() {
        return Err("must have one of: method, result, error");
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("mcp_stdio_server")?;

    let request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": PROTOCOL_VERSION,
            "clientInfo": {"name": "claude-code", "version": "1.0"}
        }
    });

    let response = handle_request(&request);

    // Persist request + response so other tools can diff the exchange.
    let rp = ctx.path("request.json");
    let sp = ctx.path("response.json");
    std::fs::write(
        &rp,
        serde_json::to_vec_pretty(&request)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;
    std::fs::write(
        &sp,
        serde_json::to_vec_pretty(&response)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    println!("=== Recipe: {} ===", ctx.name());
    println!(">>> client -> server (stdin line):");
    println!(
        "{}",
        serde_json::to_string(&request).map_err(|e| CookbookError::Serialization(e.to_string()))?
    );
    println!("<<< server -> client (stdout line):");
    println!(
        "{}",
        serde_json::to_string(&response)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?
    );

    let ok = validate_envelope(&response).is_ok();
    let name = response
        .pointer("/result/serverInfo/name")
        .and_then(Value::as_str)
        .unwrap_or("<missing>");
    ctx.record_string_metric("server_name", name);
    ctx.record_string_metric("verdict", if ok { "PASS" } else { "FAIL" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialize_returns_server_info() {
        let req = json!({"jsonrpc":"2.0","id":1,"method":"initialize"});
        let resp = handle_request(&req);
        assert_eq!(resp["result"]["serverInfo"]["name"], SERVER_NAME);
        assert_eq!(resp["result"]["protocolVersion"], PROTOCOL_VERSION);
    }

    #[test]
    fn unknown_method_returns_error() {
        let req = json!({"jsonrpc":"2.0","id":9,"method":"nope"});
        let resp = handle_request(&req);
        assert_eq!(resp["error"]["code"], -32601);
    }

    #[test]
    fn envelope_validation_rejects_missing_version() {
        let v = json!({"id":1,"method":"initialize"});
        assert!(validate_envelope(&v).is_err());
    }
}
