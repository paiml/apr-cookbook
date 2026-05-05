//! # MCP — FALSIFY-MCP-009 Byte-Parity Across Dispatcher Swap
//!
//! M5 introduces a `pmcp-dispatcher` feature flag that swaps aprender-mcp's
//! hand-rolled stdio dispatcher for the pmcp 2.3 implementation. The
//! FALSIFY-MCP-009 invariant is that the wire output for the same input
//! request is **byte-identical** under both dispatchers. This recipe
//! demonstrates the byte-parity test pattern: feed the same request
//! through both dispatchers (here simulated by two identically-implemented
//! handlers), assert the response bytes match exactly.
//!
//! Demonstrates the **MCP+.4** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PR #908 (MCP M5 scaffold) + FALSIFY-MCP-009 byte-identical parity test
//!
//! Run with: cargo run --example mcp_byte_parity_dispatcher_swap
//!
//! Added by PMAT-078 (expand-cookbooks: MCP M5 transports + notifications).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use aprender_mcp::types::JsonRpcRequest;
use aprender_mcp::AprMcpServer;
use serde_json::json;

/// Send the same request through two AprMcpServer instances (proxy for
/// hand-rolled vs pmcp dispatcher) and return both serialized responses.
/// Real FALSIFY-MCP-009 wires this up with `cfg(feature = "pmcp-dispatcher")`.
fn dispatch_through_both(req: &JsonRpcRequest) -> (Vec<u8>, Vec<u8>) {
    let mut server_a = AprMcpServer::new();
    let mut server_b = AprMcpServer::new();
    let resp_a = server_a.handle_request(req);
    let resp_b = server_b.handle_request(req);
    let bytes_a = serde_json::to_vec(&resp_a).unwrap();
    let bytes_b = serde_json::to_vec(&resp_b).unwrap();
    (bytes_a, bytes_b)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mcp_byte_parity_dispatcher_swap")?;
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "tools/list".to_string(),
        params: json!({}),
    };

    let (a, b) = dispatch_through_both(&req);
    println!("hand-rolled dispatcher response: {} bytes", a.len());
    println!("pmcp dispatcher response:        {} bytes", b.len());
    if a == b {
        println!("FALSIFY-MCP-009 ok: byte-identical responses");
    } else {
        println!("FALSIFY-MCP-009 FAIL: dispatchers disagree at byte level (drift detector fired)");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn falsify_mcp_009_identical_responses_for_initialize() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: json!({}),
        };
        let (a, b) = dispatch_through_both(&req);
        assert_eq!(
            a, b,
            "FALSIFY-MCP-009: initialize responses must be byte-identical"
        );
    }

    #[test]
    fn falsify_mcp_009_identical_responses_for_tools_list() {
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(99)),
            method: "tools/list".to_string(),
            params: json!({}),
        };
        let (a, b) = dispatch_through_both(&req);
        assert_eq!(a, b);
    }

    #[test]
    fn falsify_mcp_009_identical_for_error_path() {
        // Bad jsonrpc → -32600. Both dispatchers must produce the same error envelope.
        let req = JsonRpcRequest {
            jsonrpc: "1.0".to_string(),
            id: Some(json!(1)),
            method: "initialize".to_string(),
            params: json!({}),
        };
        let (a, b) = dispatch_through_both(&req);
        assert_eq!(a, b);
    }
}
