//! # MCP Embedded Protocol Invariants
//!
//! Exercises the two protocol-level invariants the aprender-mcp dispatcher
//! enforces before routing any request:
//!
//! - **FALSIFY-MCP-005**: `jsonrpc` must be exactly `"2.0"` or the response
//!   is `-32600 Invalid Request`.
//! - **FALSIFY-MCP-007**: an `initialize` whose `params.protocolVersion`
//!   mismatches the server's version returns `-32602 Invalid Params` instead
//!   of advancing to `tools/list`.
//!
//! Together these pin the JSON-RPC 2.0 + MCP 2024-11-05 spec compliance gate.
//! External Rust apps embedding aprender-mcp should run these as part of
//! their integration smoke-test.
//!
//! Demonstrates the **MCP-EMB.3** recipe — protocol-invariant gates as the
//! third aprender-mcp embed example (per subcrate-coverage.md).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: JSON-RPC 2.0 Specification + MCP v2024-11-05 (https://spec.modelcontextprotocol.io)
//!
//! Run with: cargo run --example mcp_embedded_protocol_invariants
//!
//! Added by PMAT-079 (expand-cookbooks: aprender-mcp coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use aprender_mcp::types::JsonRpcRequest;
use aprender_mcp::AprMcpServer;
use serde_json::json;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mcp_embedded_protocol_invariants")?;

    // FALSIFY-MCP-005 — bad jsonrpc version → -32600.
    let mut server = AprMcpServer::new();
    let bad_jsonrpc = JsonRpcRequest {
        jsonrpc: "1.0".to_string(),
        id: Some(json!(1)),
        method: "initialize".to_string(),
        params: json!({}),
    };
    let resp = server.handle_request(&bad_jsonrpc);
    let err = resp
        .error
        .as_ref()
        .expect("FALSIFY-MCP-005: bad jsonrpc must error");
    println!(
        "FALSIFY-MCP-005 ok: bad jsonrpc rejected with code={} message={:?}",
        err.code, err.message
    );

    // FALSIFY-MCP-007 — mismatched protocolVersion → -32602.
    let bad_proto = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(2)),
        method: "initialize".to_string(),
        params: json!({"protocolVersion": "1900-01-01"}),
    };
    let resp = server.handle_request(&bad_proto);
    let err = resp
        .error
        .as_ref()
        .expect("FALSIFY-MCP-007: bad protocolVersion must error");
    println!(
        "FALSIFY-MCP-007 ok: bad protocolVersion rejected with code={} message={:?}",
        err.code, err.message
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invariants_run() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn falsify_mcp_005_bad_jsonrpc_returns_invalid_request() {
        let mut server = AprMcpServer::new();
        let req = JsonRpcRequest {
            jsonrpc: "1.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: json!({}),
        };
        let resp = server.handle_request(&req);
        assert_eq!(resp.error.unwrap().code, -32600);
    }

    #[test]
    fn falsify_mcp_007_bad_protocol_version_returns_invalid_params() {
        let mut server = AprMcpServer::new();
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: json!({"protocolVersion": "9999-12-31"}),
        };
        let resp = server.handle_request(&req);
        assert_eq!(resp.error.unwrap().code, -32602);
    }

    #[test]
    fn missing_protocol_version_is_permitted() {
        // The dispatcher allows initialize without an explicit protocolVersion
        // (some clients omit it on first handshake) — only a *mismatch* is
        // rejected.
        let mut server = AprMcpServer::new();
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: json!({}),
        };
        let resp = server.handle_request(&req);
        assert!(resp.error.is_none());
    }
}
