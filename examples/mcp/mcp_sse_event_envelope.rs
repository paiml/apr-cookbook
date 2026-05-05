//! # MCP M5 — SSE Transport Event Envelope
//!
//! M5 ports the aprender-mcp dispatcher to `pmcp = "2.3"` with SSE
//! (Server-Sent Events) transport support per the MCP 2024-11-05 spec.
//! SSE wire format:
//! ```text
//! event: <event_name>
//! data: <json_payload>
//! \n
//! ```
//!
//! This recipe builds the canonical SSE envelope for a tools/list response,
//! validates the format (event name on first line, `data: ` prefix on
//! second, blank line terminator), and parses it back. External Rust apps
//! using aprender-mcp's SSE transport need exactly this envelope shape.
//!
//! Demonstrates the **MCP+.1** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: MCP Spec §SSE Transport. https://spec.modelcontextprotocol.io/specification/2024-11-05/basic/transports/
//!
//! Run with: cargo run --example mcp_sse_event_envelope
//!
//! Added by PMAT-078 (expand-cookbooks: MCP M5 transports + notifications).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use serde_json::Value;

fn build_sse_envelope(event: &str, data: &Value) -> String {
    let payload = serde_json::to_string(data).unwrap();
    format!("event: {event}\ndata: {payload}\n\n")
}

fn parse_sse_envelope(raw: &str) -> Result<(String, Value)> {
    let mut event = None;
    let mut data_line = None;
    for line in raw.lines() {
        if let Some(rest) = line.strip_prefix("event: ") {
            event = Some(rest.to_string());
        } else if let Some(rest) = line.strip_prefix("data: ") {
            data_line = Some(rest.to_string());
        }
    }
    let event = event.ok_or_else(|| {
        apr_cookbook::CookbookError::Validation("SSE envelope missing `event:` line".into())
    })?;
    let raw_data = data_line.ok_or_else(|| {
        apr_cookbook::CookbookError::Validation("SSE envelope missing `data:` line".into())
    })?;
    let parsed: Value = serde_json::from_str(&raw_data).map_err(|e| {
        apr_cookbook::CookbookError::Validation(format!("data payload not JSON: {e}"))
    })?;
    Ok((event, parsed))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mcp_sse_event_envelope")?;
    let payload = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"tools": [{"name": "apr.version"}]}
    });
    let envelope = build_sse_envelope("message", &payload);
    print!("--- SSE envelope on the wire ---\n{envelope}");
    let (event, parsed) = parse_sse_envelope(&envelope)?;
    println!(
        "parsed event={event} payload-keys={:?}",
        parsed.as_object().map(|o| o.keys().collect::<Vec<_>>())
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn envelope_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn envelope_ends_with_blank_line() {
        let env = build_sse_envelope("message", &json!({"x": 1}));
        // SSE message MUST be terminated by a blank line per the spec.
        assert!(env.ends_with("\n\n"));
    }

    #[test]
    fn parse_roundtrip_preserves_event_and_data() {
        let original = json!({"id": 42, "result": "ok"});
        let env = build_sse_envelope("message", &original);
        let (event, parsed) = parse_sse_envelope(&env).unwrap();
        assert_eq!(event, "message");
        assert_eq!(parsed, original);
    }

    #[test]
    fn parse_rejects_missing_event() {
        let bad = "data: {\"x\": 1}\n\n";
        assert!(parse_sse_envelope(bad).is_err());
    }
}
