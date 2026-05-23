//! # apr mcp — Tool Manifest Envelope
//!
//! `apr mcp` exposes apr subcommands as MCP tools. This recipe builds
//! the manifest envelope that the server emits during the MCP
//! `initialize` handshake: per-tool name, description, and JSON schema
//! for arguments. Required keys: `tools` array, each tool with `name`,
//! `description`, `inputSchema`.
//!
//! Demonstrates the **MCP.5** recipe for PMAT-107 (apr mcp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender MCP-001 + Model Context Protocol spec
//!
//! Run with: cargo run --example cli_mcp_tool_manifest_envelope
//!
//! Added by PMAT-107 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct McpTool {
    pub name: &'static str,
    pub description: &'static str,
    pub input_schema: Value,
}

pub fn build_manifest(tools: &[McpTool]) -> Value {
    json!({
        "tools": tools
            .iter()
            .map(|t| json!({
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema.clone(),
            }))
            .collect::<Vec<_>>()
    })
}

pub fn validate_manifest(v: &Value) -> Result<usize> {
    let tools = v.get("tools").and_then(Value::as_array).ok_or_else(|| {
        apr_cookbook::CookbookError::Validation("manifest missing 'tools' array".into())
    })?;
    for (i, t) in tools.iter().enumerate() {
        for key in ["name", "description", "inputSchema"] {
            if t.get(key).is_none() {
                return Err(apr_cookbook::CookbookError::Validation(format!(
                    "tool[{i}] missing key '{key}'"
                )));
            }
        }
    }
    Ok(tools.len())
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_mcp_tool_manifest_envelope")?;

    let tools = vec![
        McpTool {
            name: "apr_inspect",
            description: "Inspect APR model metadata, vocab, and structure",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file": { "type": "string", "description": "Path to .apr file" }
                },
                "required": ["file"]
            }),
        },
        McpTool {
            name: "apr_bench",
            description: "Run inference benchmark",
            input_schema: json!({
                "type": "object",
                "properties": {
                    "file": { "type": "string" },
                    "iterations": { "type": "integer", "default": 10 }
                },
                "required": ["file"]
            }),
        },
    ];

    let manifest = build_manifest(&tools);
    println!("{}", serde_json::to_string_pretty(&manifest).unwrap());
    println!(
        "\nvalidation: {} tools",
        validate_manifest(&manifest).unwrap()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tool() -> McpTool {
        McpTool {
            name: "apr_inspect",
            description: "Inspect a model",
            input_schema: json!({ "type": "object", "properties": {} }),
        }
    }

    #[test]
    fn manifest_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn manifest_has_tools_array() {
        let m = build_manifest(&[sample_tool()]);
        assert!(m["tools"].is_array());
        assert_eq!(m["tools"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn each_tool_has_required_keys() {
        let m = build_manifest(&[sample_tool()]);
        let t = &m["tools"][0];
        for key in ["name", "description", "inputSchema"] {
            assert!(t.get(key).is_some(), "missing key {key}");
        }
    }

    #[test]
    fn empty_tools_yield_empty_array() {
        let m = build_manifest(&[]);
        assert!(m["tools"].as_array().unwrap().is_empty());
        assert_eq!(validate_manifest(&m).unwrap(), 0);
    }

    #[test]
    fn validate_rejects_missing_tools_key() {
        let bad = json!({ "wrong_key": [] });
        assert!(validate_manifest(&bad).is_err());
    }

    #[test]
    fn validate_rejects_tool_missing_input_schema() {
        let bad = json!({
            "tools": [{ "name": "x", "description": "y" }]
        });
        assert!(validate_manifest(&bad).is_err());
    }

    #[test]
    fn json_serialization_round_trips() {
        let m = build_manifest(&[sample_tool()]);
        let s = serde_json::to_string(&m).unwrap();
        let parsed: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed, m);
    }
}
