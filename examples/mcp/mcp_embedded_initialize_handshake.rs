//! # MCP Embedded Initialize Handshake
//!
//! Smallest possible aprender-mcp embed: construct an `AprMcpServer`, send a
//! single `initialize` JSON-RPC request through the in-process dispatcher,
//! assert the response carries the expected `protocolVersion` and `serverInfo`
//! shape. No stdin/stdout — purely in-memory `JsonRpcRequest` → `JsonRpcResponse`
//! roundtrip.
//!
//! Demonstrates the **MCP-EMB.1** recipe from
//! `docs/specifications/expand-cookbooks/subcrate-coverage.md` — the
//! lightest possible aprender-mcp integration that an external Rust app
//! can copy verbatim.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Anthropic (2024). Model Context Protocol Specification v2024-11-05. https://spec.modelcontextprotocol.io
//!
//! Run with: cargo run --example mcp_embedded_initialize_handshake
//!
//! Added by PMAT-079 (expand-cookbooks: aprender-mcp coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use aprender_mcp::types::{JsonRpcRequest, JsonRpcResponse};
use aprender_mcp::AprMcpServer;
use serde_json::json;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mcp_embedded_initialize_handshake")?;

    let mut server = AprMcpServer::new();
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "initialize".to_string(),
        params: json!({}),
    };

    let resp: JsonRpcResponse = server.handle_request(&req);

    assert_eq!(resp.jsonrpc, "2.0");
    assert!(resp.error.is_none(), "initialize should not error");
    let result = resp.result.as_ref().expect("initialize must return result");
    let proto = result["protocolVersion"]
        .as_str()
        .expect("protocolVersion must be a string");
    let server_name = result["serverInfo"]["name"]
        .as_str()
        .expect("serverInfo.name must be a string");

    println!("MCP server initialized: name={server_name} protocolVersion={proto}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handshake_returns_protocol_version() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn handshake_rejects_bad_jsonrpc() {
        // FALSIFY-MCP-005 — non-2.0 jsonrpc → -32600 Invalid Request.
        let mut server = AprMcpServer::new();
        let req = JsonRpcRequest {
            jsonrpc: "1.0".to_string(),
            id: Some(json!(99)),
            method: "initialize".to_string(),
            params: json!({}),
        };
        let resp = server.handle_request(&req);
        let err = resp.error.expect("must error on bad jsonrpc");
        assert_eq!(err.code, -32600);
    }
}
