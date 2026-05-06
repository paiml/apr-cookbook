//! # apr mcp — JSON-RPC 2.0 Request Validator
//!
//! `apr mcp` speaks JSON-RPC 2.0 over stdio. Per the spec, every request
//! must carry `jsonrpc: "2.0"`, a `method` string, and an `id` (number,
//! string, or null for notifications). This recipe builds the validator
//! and asserts the contract against malformed inputs.
//!
//! Demonstrates the **MCP.6** recipe for PMAT-107 (apr mcp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender MCP-002 + JSON-RPC 2.0 spec
//!
//! Run with: cargo run --example cli_mcp_jsonrpc_request_validator
//!
//! Added by PMAT-107 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use serde_json::Value;

#[derive(Debug, PartialEq)]
pub enum RequestVerdict {
    OkRequest { id: String },
    OkNotification, // request without id
    MissingJsonRpc,
    WrongJsonRpcVersion(String),
    MissingMethod,
    InvalidIdType,
}

pub fn validate_request(v: &Value) -> RequestVerdict {
    let jsonrpc = v.get("jsonrpc").and_then(Value::as_str);
    match jsonrpc {
        None => return RequestVerdict::MissingJsonRpc,
        Some("2.0") => {}
        Some(other) => return RequestVerdict::WrongJsonRpcVersion(other.into()),
    }
    if v.get("method").and_then(Value::as_str).is_none() {
        return RequestVerdict::MissingMethod;
    }
    match v.get("id") {
        None => RequestVerdict::OkNotification,
        Some(Value::Number(n)) => RequestVerdict::OkRequest { id: n.to_string() },
        Some(Value::String(s)) => RequestVerdict::OkRequest { id: s.clone() },
        Some(Value::Null) => RequestVerdict::OkNotification,
        _ => RequestVerdict::InvalidIdType,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_mcp_jsonrpc_request_validator")?;

    let cases = [
        (
            "happy numeric id",
            serde_json::json!({"jsonrpc":"2.0","method":"tools/list","id":1}),
        ),
        (
            "happy string id",
            serde_json::json!({"jsonrpc":"2.0","method":"tools/list","id":"req-001"}),
        ),
        (
            "notification (no id)",
            serde_json::json!({"jsonrpc":"2.0","method":"notify"}),
        ),
        (
            "notification (null id)",
            serde_json::json!({"jsonrpc":"2.0","method":"x","id":null}),
        ),
        ("missing jsonrpc", serde_json::json!({"method":"x","id":1})),
        (
            "wrong version",
            serde_json::json!({"jsonrpc":"1.0","method":"x","id":1}),
        ),
        (
            "missing method",
            serde_json::json!({"jsonrpc":"2.0","id":1}),
        ),
        (
            "array id (invalid)",
            serde_json::json!({"jsonrpc":"2.0","method":"x","id":[1,2,3]}),
        ),
    ];
    for (label, v) in cases {
        println!("{label:>22}  →  {:?}", validate_request(&v));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_request_with_int_id() {
        let v = serde_json::json!({"jsonrpc":"2.0","method":"tools/list","id":1});
        let r = validate_request(&v);
        assert!(matches!(r, RequestVerdict::OkRequest { .. }));
    }

    #[test]
    fn string_id_accepted() {
        let v = serde_json::json!({"jsonrpc":"2.0","method":"tools/list","id":"req-001"});
        if let RequestVerdict::OkRequest { id } = validate_request(&v) {
            assert_eq!(id, "req-001");
        } else {
            panic!("expected OkRequest");
        }
    }

    #[test]
    fn missing_id_is_notification() {
        let v = serde_json::json!({"jsonrpc":"2.0","method":"notify"});
        assert_eq!(validate_request(&v), RequestVerdict::OkNotification);
    }

    #[test]
    fn null_id_is_notification() {
        // Per JSON-RPC 2.0 §1.2: null id signals notification (no response expected).
        let v = serde_json::json!({"jsonrpc":"2.0","method":"x","id":null});
        assert_eq!(validate_request(&v), RequestVerdict::OkNotification);
    }

    #[test]
    fn missing_jsonrpc_field_rejected() {
        let v = serde_json::json!({"method":"x","id":1});
        assert_eq!(validate_request(&v), RequestVerdict::MissingJsonRpc);
    }

    #[test]
    fn wrong_jsonrpc_version_rejected() {
        let v = serde_json::json!({"jsonrpc":"1.0","method":"x","id":1});
        assert!(matches!(
            validate_request(&v),
            RequestVerdict::WrongJsonRpcVersion(_)
        ));
    }

    #[test]
    fn missing_method_rejected() {
        let v = serde_json::json!({"jsonrpc":"2.0","id":1});
        assert_eq!(validate_request(&v), RequestVerdict::MissingMethod);
    }

    #[test]
    fn array_id_invalid() {
        let v = serde_json::json!({"jsonrpc":"2.0","method":"x","id":[1,2,3]});
        assert_eq!(validate_request(&v), RequestVerdict::InvalidIdType);
    }

    #[test]
    fn object_id_invalid() {
        let v = serde_json::json!({"jsonrpc":"2.0","method":"x","id":{"a":1}});
        assert_eq!(validate_request(&v), RequestVerdict::InvalidIdType);
    }
}
