//! # apr code — MCP Client Config (.mcp.json)
//!
//! `apr code` discovers external MCP servers via a `.mcp.json` manifest at
//! the project root. Each entry declares a server name, transport (stdio /
//! sse / http), command, args, and env. This recipe writes a sample
//! `.mcp.json` to a tempdir, parses it, and asserts the schema (name +
//! transport-specific fields) without invoking any actual server process.
//!
//! Demonstrates the **C.1** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: apr-code-parity-v1.yaml row PMAT-CODE-MCP-CLIENT-001 (SHIPPED v4)
//!
//! Run with: cargo run --example code_mcp_client_config
//!
//! Added by PMAT-074 (expand-cookbooks: apr code agentic surface).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::fs;
use std::io::Write;

const SAMPLE_MCP_JSON: &str = r#"{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/sandbox"],
      "transport": "stdio"
    },
    "github-api": {
      "url": "https://mcp-bridge.example.com/sse",
      "transport": "sse"
    }
  }
}
"#;

fn parse_mcp_manifest(json: &str) -> Result<serde_json::Value> {
    serde_json::from_str(json)
        .map_err(|e| apr_cookbook::CookbookError::Validation(format!("mcp.json parse error: {e}")))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("code_mcp_client_config")?;
    let dir = tempfile::tempdir()?;
    let path = dir.path().join(".mcp.json");
    let mut file = fs::File::create(&path)?;
    file.write_all(SAMPLE_MCP_JSON.as_bytes())?;
    drop(file);

    let content = fs::read_to_string(&path)?;
    let manifest = parse_mcp_manifest(&content)?;
    let servers = manifest["mcpServers"].as_object().ok_or_else(|| {
        apr_cookbook::CookbookError::Validation("mcpServers must be an object".into())
    })?;

    println!(".mcp.json declares {} MCP servers:", servers.len());
    for (name, cfg) in servers {
        let transport = cfg["transport"].as_str().unwrap_or("?");
        println!("  {name}: transport={transport}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_parser_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn parses_two_servers() {
        let m = parse_mcp_manifest(SAMPLE_MCP_JSON).unwrap();
        let servers = m["mcpServers"].as_object().unwrap();
        assert_eq!(servers.len(), 2);
        assert!(servers.contains_key("filesystem"));
        assert!(servers.contains_key("github-api"));
    }

    #[test]
    fn stdio_server_has_command() {
        let m = parse_mcp_manifest(SAMPLE_MCP_JSON).unwrap();
        let fs_srv = &m["mcpServers"]["filesystem"];
        assert_eq!(fs_srv["transport"], "stdio");
        assert_eq!(fs_srv["command"], "npx");
        assert!(fs_srv["args"].is_array());
    }

    #[test]
    fn sse_server_has_url() {
        let m = parse_mcp_manifest(SAMPLE_MCP_JSON).unwrap();
        let gh = &m["mcpServers"]["github-api"];
        assert_eq!(gh["transport"], "sse");
        assert!(gh["url"].as_str().unwrap().starts_with("https://"));
    }
}
