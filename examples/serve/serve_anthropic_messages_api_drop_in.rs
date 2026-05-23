//! # apr serve anthropic — Claude Messages API Drop-In
//!
//! Demonstrates the request/response shape of `apr serve anthropic` — the
//! Claude Messages-API-compatible serving mode that lets you drop your
//! `ANTHROPIC_API_KEY=foo` and have a sovereign apr backend answer instead.
//! The serve mode is in aprender's Unreleased section behind
//! `apr-claude-proxy-v1.yaml` (DRAFT).
//!
//! Recipe builds a sample Messages-API request envelope, validates its
//! schema, then constructs the canonical SSE-streaming response sequence
//! (`message_start` → `content_block_delta` × N → `message_stop`) that
//! the proxy must emit per FALSIFY-CLAUDE-PROXY-002 (SSE event sequence).
//!
//! Demonstrates the **SRV+.1** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Anthropic (2024). Messages API Reference. https://docs.anthropic.com/en/api/messages
//!
//! Run with: cargo run --example serve_anthropic_messages_api_drop_in
//!
//! Added by PMAT-077 (expand-cookbooks: apr serve anthropic + plan hf://).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use serde_json::{json, Value};

/// Validate a Claude Messages-API request envelope per the public schema.
/// Returns Ok with model + message count on success.
fn validate_messages_request(req: &Value) -> Result<(String, usize)> {
    let model = req["model"].as_str().ok_or_else(|| {
        apr_cookbook::CookbookError::Validation("missing required `model` (string)".into())
    })?;
    let max_tokens = req["max_tokens"].as_u64().ok_or_else(|| {
        apr_cookbook::CookbookError::Validation(
            "missing required `max_tokens` (positive integer)".into(),
        )
    })?;
    if max_tokens == 0 {
        return Err(apr_cookbook::CookbookError::Validation(
            "max_tokens must be > 0".into(),
        ));
    }
    let messages = req["messages"].as_array().ok_or_else(|| {
        apr_cookbook::CookbookError::Validation("missing required `messages` (array)".into())
    })?;
    if messages.is_empty() {
        return Err(apr_cookbook::CookbookError::Validation(
            "messages array must be non-empty".into(),
        ));
    }
    for (i, m) in messages.iter().enumerate() {
        let role = m["role"].as_str().ok_or_else(|| {
            apr_cookbook::CookbookError::Validation(format!(
                "messages[{i}].role missing or not a string"
            ))
        })?;
        if !matches!(role, "user" | "assistant") {
            return Err(apr_cookbook::CookbookError::Validation(format!(
                "messages[{i}].role must be \"user\" or \"assistant\", got {role:?}"
            )));
        }
        if m["content"].is_null() {
            return Err(apr_cookbook::CookbookError::Validation(format!(
                "messages[{i}].content missing"
            )));
        }
    }
    Ok((model.to_string(), messages.len()))
}

/// Construct the canonical SSE event sequence that `apr serve anthropic`
/// emits for a streaming response. Per FALSIFY-CLAUDE-PROXY-002 the sequence
/// must start with `message_start`, contain N `content_block_delta` events,
/// and end with `message_stop`. We build the events as named JSON values
/// (in real wire-format each is prefixed with `event: <name>\ndata: <json>`).
fn build_sse_event_sequence(text_chunks: &[&str]) -> Vec<(String, Value)> {
    let mut events = Vec::with_capacity(text_chunks.len() + 2);
    events.push((
        "message_start".to_string(),
        json!({"type": "message_start", "message": {"id": "msg_demo", "role": "assistant"}}),
    ));
    for (i, chunk) in text_chunks.iter().enumerate() {
        events.push((
            "content_block_delta".to_string(),
            json!({
                "type": "content_block_delta",
                "index": i,
                "delta": {"type": "text_delta", "text": chunk}
            }),
        ));
    }
    events.push(("message_stop".to_string(), json!({"type": "message_stop"})));
    events
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("serve_anthropic_messages_api_drop_in")?;

    let request = json!({
        "model": "claude-opus-4-7",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": "Summarize the IIUR principle in one sentence."}
        ]
    });
    let (model, n_messages) = validate_messages_request(&request)?;
    println!("validated request: model={model} messages={n_messages}");

    let chunks = [
        "IIUR ",
        "means ",
        "Isolated, ",
        "Idempotent, ",
        "Useful, ",
        "Reproducible.",
    ];
    let events = build_sse_event_sequence(&chunks);
    println!("emitted {} SSE events:", events.len());
    for (name, _) in &events {
        println!("  event: {name}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drop_in_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn missing_model_rejected() {
        let bad = json!({"max_tokens": 100, "messages": [{"role": "user", "content": "hi"}]});
        assert!(validate_messages_request(&bad).is_err());
    }

    #[test]
    fn zero_max_tokens_rejected() {
        let bad =
            json!({"model": "x", "max_tokens": 0, "messages": [{"role": "user", "content": "hi"}]});
        assert!(validate_messages_request(&bad).is_err());
    }

    #[test]
    fn invalid_role_rejected() {
        let bad = json!({"model": "x", "max_tokens": 100, "messages": [{"role": "system", "content": "hi"}]});
        assert!(validate_messages_request(&bad).is_err());
    }

    #[test]
    fn sse_sequence_brackets_with_start_stop() {
        let events = build_sse_event_sequence(&["a", "b"]);
        assert_eq!(
            events.first().map(|(n, _)| n.as_str()),
            Some("message_start")
        );
        assert_eq!(events.last().map(|(n, _)| n.as_str()), Some("message_stop"));
        // 1 start + 2 deltas + 1 stop = 4 events
        assert_eq!(events.len(), 4);
    }
}
