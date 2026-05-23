//! # MCP Embedded Tools List Discovery
//!
//! Send a `tools/list` JSON-RPC request through the in-process aprender-mcp
//! dispatcher, assert all 9 expected MCP tools are present, and verify each
//! tool carries a JSON Schema Draft 7 `inputSchema` (FALSIFY-MCP-002 strict
//! invariant). External Rust apps embedding aprender-mcp can use this as a
//! pre-flight smoke-test for the build.rs codegen pipeline.
//!
//! Demonstrates the **MCP-EMB.2** recipe — pre-flight smoke for the
//! `apr-mcp-tool-schemas-v1.yaml` codegen contract.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: JSON Schema Draft 7 (RFC 8259-adjacent). https://json-schema.org/draft-07/json-schema-release-notes
//!
//! Run with: cargo run --example mcp_embedded_tools_list_discovery
//!
//! Added by PMAT-079 (expand-cookbooks: aprender-mcp coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use aprender_mcp::types::JsonRpcRequest;
use aprender_mcp::AprMcpServer;
use serde_json::json;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mcp_embedded_tools_list_discovery")?;

    let mut server = AprMcpServer::new();
    let req = JsonRpcRequest {
        jsonrpc: "2.0".to_string(),
        id: Some(json!(1)),
        method: "tools/list".to_string(),
        params: json!({}),
    };

    let resp = server.handle_request(&req);
    let result = resp.result.as_ref().expect("tools/list must return result");
    let tools = result["tools"].as_array().expect("tools must be an array");

    println!("aprender-mcp v0.31.2 advertises {} tools", tools.len());
    for tool in tools {
        let name = tool["name"].as_str().unwrap_or("?");
        let has_input_schema = tool.get("inputSchema").is_some();
        println!("  - {name} (inputSchema present: {has_input_schema})");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 9 tools aprender-mcp 0.31.2 exposes per apr-mcp-tool-schemas-v1.
    /// A contract-side rename surfaces as a recipe-test failure.
    const EXPECTED_TOOLS: &[&str] = &[
        "apr.version",
        "apr.validate",
        "apr.tensors",
        "apr.bench",
        "apr.qa",
        "apr.trace",
        "apr.run",
        "apr.serve",
        "apr.finetune",
    ];

    #[test]
    fn discovery_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn all_expected_tools_present() {
        let mut server = AprMcpServer::new();
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "tools/list".to_string(),
            params: json!({}),
        };
        let resp = server.handle_request(&req);
        let result = resp.result.unwrap();
        let tools: Vec<String> = result["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["name"].as_str().unwrap().to_string())
            .collect();
        for expected in EXPECTED_TOOLS {
            assert!(
                tools.iter().any(|t| t == expected),
                "expected tool {expected} not in tools/list response: {tools:?}"
            );
        }
    }

    #[test]
    fn every_tool_carries_input_schema() {
        // FALSIFY-MCP-002 strict — every tool's inputSchema must be a valid
        // JSON Schema Draft 7 object (we only check structural presence here;
        // upstream CI runs the full meta-validation).
        let mut server = AprMcpServer::new();
        let req = JsonRpcRequest {
            jsonrpc: "2.0".to_string(),
            id: Some(json!(1)),
            method: "tools/list".to_string(),
            params: json!({}),
        };
        let resp = server.handle_request(&req);
        let result = resp.result.unwrap();
        for tool in result["tools"].as_array().unwrap() {
            let name = tool["name"].as_str().unwrap();
            assert!(
                tool.get("inputSchema").is_some(),
                "tool {name} missing inputSchema"
            );
            let schema = &tool["inputSchema"];
            assert_eq!(
                schema["type"].as_str(),
                Some("object"),
                "tool {name} inputSchema.type must be \"object\""
            );
        }
    }
}
