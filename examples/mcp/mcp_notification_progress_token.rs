//! # MCP — `notifications/progress` for Long-Running Jobs
//!
//! `apr.finetune` and other long-running MCP tools opt-in to per-line
//! progress notifications via `params._meta.progressToken`. The server
//! emits `notifications/progress` events tagged with the same token so
//! the client can correlate. This recipe simulates the full lifecycle:
//! request with progressToken → server emits 5 progress notifications
//! → final response.
//!
//! Demonstrates the **MCP+.3** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: apr-mcp-tool-schemas-v1.yaml FALSIFY-MCP-PROGRESS-001 (per-line progress notifications opt-in via progressToken)
//!
//! Run with: cargo run --example mcp_notification_progress_token
//!
//! Added by PMAT-078 (expand-cookbooks: MCP M5 transports + notifications).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use serde_json::{json, Value};

fn build_request_with_progress_token(method: &str, token: &str) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": {
            "_meta": {"progressToken": token}
        }
    })
}

fn extract_progress_token(req: &Value) -> Option<String> {
    req["params"]["_meta"]["progressToken"]
        .as_str()
        .map(String::from)
}

fn build_progress_notification(token: &str, progress: f64, total: f64) -> Value {
    json!({
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {
            "progressToken": token,
            "progress": progress,
            "total": total
        }
    })
}

fn build_final_response(id: i64) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": {"status": "complete"}
    })
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mcp_notification_progress_token")?;

    let token = "finetune-job-2026-05";
    let req = build_request_with_progress_token("apr.finetune", token);
    println!("request opts in via _meta.progressToken={}", token);

    let extracted = extract_progress_token(&req).unwrap();
    println!("server extracts token: {extracted}");

    println!("\nserver emits 5 progress notifications:");
    for i in 1..=5 {
        let n = build_progress_notification(&extracted, f64::from(i), 5.0);
        println!(
            "  notifications/progress {}/5 (token={})",
            n["params"]["progress"], n["params"]["progressToken"]
        );
    }

    let resp = build_final_response(1);
    println!("\nfinal response: {}", resp["result"]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn token_roundtrip_through_request() {
        let req = build_request_with_progress_token("apr.finetune", "tok-42");
        assert_eq!(extract_progress_token(&req), Some("tok-42".into()));
    }

    #[test]
    fn missing_progress_token_returns_none() {
        let req = json!({"jsonrpc": "2.0", "id": 1, "method": "apr.bench", "params": {}});
        assert_eq!(extract_progress_token(&req), None);
    }

    #[test]
    fn progress_notification_carries_token_progress_total() {
        let n = build_progress_notification("tok-7", 3.0, 10.0);
        assert_eq!(n["method"], "notifications/progress");
        assert_eq!(n["params"]["progressToken"], "tok-7");
        assert_eq!(n["params"]["progress"], 3.0);
        assert_eq!(n["params"]["total"], 10.0);
        // Notifications MUST NOT carry an `id` per JSON-RPC 2.0.
        assert!(n.get("id").is_none());
    }
}
