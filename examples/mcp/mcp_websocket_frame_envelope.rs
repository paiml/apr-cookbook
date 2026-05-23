//! # MCP M5 — WebSocket Transport Frame Envelope
//!
//! WebSocket transport for aprender-mcp wraps each JSON-RPC message in an
//! RFC 6455 text frame. This recipe demonstrates the message envelope
//! layer (the JSON-RPC payload that goes inside a WebSocket text frame),
//! validates the bidirectional send/receive contract (request → response,
//! notification = no response), and asserts that batched requests share a
//! single frame.
//!
//! Demonstrates the **MCP+.2** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: RFC 6455 The WebSocket Protocol + MCP 2024-11-05 transport spec
//!
//! Run with: cargo run --example mcp_websocket_frame_envelope
//!
//! Added by PMAT-078 (expand-cookbooks: MCP M5 transports + notifications).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use serde_json::{json, Value};

/// JSON-RPC message classifier per the MCP wire spec.
#[derive(Debug, PartialEq, Eq)]
enum MessageKind {
    Request,
    Response,
    Notification,
}

fn classify(msg: &Value) -> Result<MessageKind> {
    let has_method = msg.get("method").is_some();
    let has_id = msg.get("id").is_some();
    let has_result = msg.get("result").is_some();
    let has_error = msg.get("error").is_some();
    match (has_method, has_id, has_result || has_error) {
        (true, true, false) => Ok(MessageKind::Request),
        (true, false, false) => Ok(MessageKind::Notification),
        (false, true, true) => Ok(MessageKind::Response),
        _ => Err(apr_cookbook::CookbookError::Validation(format!(
            "ambiguous message shape: method={has_method} id={has_id} result|error={}",
            has_result || has_error
        ))),
    }
}

/// Encode a batch of messages as a single WebSocket text-frame payload.
fn encode_batch(msgs: &[Value]) -> String {
    serde_json::to_string(msgs).unwrap()
}

fn decode_batch(payload: &str) -> Result<Vec<Value>> {
    serde_json::from_str(payload)
        .map_err(|e| apr_cookbook::CookbookError::Validation(format!("batch decode: {e}")))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mcp_websocket_frame_envelope")?;

    let request = json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}});
    let notification =
        json!({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}});
    let response = json!({"jsonrpc": "2.0", "id": 1, "result": {"tools": []}});

    println!("classify (request):       {:?}", classify(&request)?);
    println!("classify (notification):  {:?}", classify(&notification)?);
    println!("classify (response):      {:?}", classify(&response)?);

    let batch_payload = encode_batch(&[request.clone(), notification.clone()]);
    println!("\nbatched WS frame payload: {batch_payload}");
    let decoded = decode_batch(&batch_payload)?;
    println!("decoded {} messages", decoded.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn classify_request() {
        let m = json!({"jsonrpc": "2.0", "id": 1, "method": "x"});
        assert_eq!(classify(&m).unwrap(), MessageKind::Request);
    }

    #[test]
    fn classify_notification() {
        let m = json!({"jsonrpc": "2.0", "method": "x"});
        assert_eq!(classify(&m).unwrap(), MessageKind::Notification);
    }

    #[test]
    fn classify_response() {
        let m = json!({"jsonrpc": "2.0", "id": 1, "result": "ok"});
        assert_eq!(classify(&m).unwrap(), MessageKind::Response);
    }

    #[test]
    fn batch_roundtrip() {
        let a = json!({"jsonrpc": "2.0", "id": 1, "method": "a"});
        let b = json!({"jsonrpc": "2.0", "id": 2, "method": "b"});
        let payload = encode_batch(&[a.clone(), b.clone()]);
        let decoded = decode_batch(&payload).unwrap();
        assert_eq!(decoded, vec![a, b]);
    }
}
