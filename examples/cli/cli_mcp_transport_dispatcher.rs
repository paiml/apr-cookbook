//! # apr mcp --transport — Transport Mode Dispatcher
//!
//! MCP supports {stdio, sse, websocket}. Constraints: stdio for local
//! subprocess pairing only; sse and websocket require host + port;
//! websocket prefers wss:// (TLS) over ws://. This recipe builds the
//! dispatcher.
//!
//! Demonstrates the **MCP.5** recipe for PMAT-120 (apr mcp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender MCP-001 + JSON-RPC 2.0 + SSE/WebSocket RFCs
//!
//! Run with: cargo run --example cli_mcp_transport_dispatcher
//!
//! Added by PMAT-120 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    Stdio,
    Sse,
    WebSocket,
}

impl Transport {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "stdio" => Some(Transport::Stdio),
            "sse" => Some(Transport::Sse),
            "websocket" | "ws" | "wss" => Some(Transport::WebSocket),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum DispatchVerdict {
    Ok,
    UnknownTransport,
    StdioRejectsHost,
    RemoteMissingHost,
    InsecureWsWarning,
}

pub fn dispatch(transport_str: &str, host: Option<&str>, prefer_tls: bool) -> DispatchVerdict {
    let Some(transport) = Transport::from_str_strict(transport_str) else {
        return DispatchVerdict::UnknownTransport;
    };
    match transport {
        Transport::Stdio => {
            if host.is_some() {
                DispatchVerdict::StdioRejectsHost
            } else {
                DispatchVerdict::Ok
            }
        }
        Transport::Sse | Transport::WebSocket => {
            if host.is_none() || host == Some("") {
                return DispatchVerdict::RemoteMissingHost;
            }
            if transport == Transport::WebSocket && transport_str == "ws" && prefer_tls {
                return DispatchVerdict::InsecureWsWarning;
            }
            DispatchVerdict::Ok
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_mcp_transport_dispatcher")?;

    let cases = [
        ("stdio", None, false),
        ("stdio", Some("api.x.com"), false),
        ("sse", Some("api.x.com"), false),
        ("ws", Some("api.x.com"), true),
        ("wss", Some("api.x.com"), true),
        ("websocket", None, false),
        ("typo", None, false),
    ];
    for (t, h, tls) in cases {
        println!("{t:<10} host={h:?} tls={tls}  →  {:?}", dispatch(t, h, tls));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn stdio_no_host_passes() {
        assert_eq!(dispatch("stdio", None, false), DispatchVerdict::Ok);
    }

    #[test]
    fn stdio_with_host_rejected() {
        assert_eq!(
            dispatch("stdio", Some("x.com"), false),
            DispatchVerdict::StdioRejectsHost
        );
    }

    #[test]
    fn sse_with_host_passes() {
        assert_eq!(
            dispatch("sse", Some("api.x.com"), false),
            DispatchVerdict::Ok
        );
    }

    #[test]
    fn sse_without_host_rejected() {
        assert_eq!(
            dispatch("sse", None, false),
            DispatchVerdict::RemoteMissingHost
        );
    }

    #[test]
    fn websocket_with_host_passes() {
        assert_eq!(
            dispatch("websocket", Some("api.x.com"), false),
            DispatchVerdict::Ok
        );
    }

    #[test]
    fn ws_warns_when_tls_preferred() {
        assert_eq!(
            dispatch("ws", Some("x.com"), true),
            DispatchVerdict::InsecureWsWarning
        );
    }

    #[test]
    fn wss_passes_when_tls_preferred() {
        assert_eq!(dispatch("wss", Some("x.com"), true), DispatchVerdict::Ok);
    }

    #[test]
    fn unknown_transport_rejected() {
        assert_eq!(
            dispatch("typo", None, false),
            DispatchVerdict::UnknownTransport
        );
    }

    #[test]
    fn empty_host_treated_as_missing() {
        assert_eq!(
            dispatch("sse", Some(""), false),
            DispatchVerdict::RemoteMissingHost
        );
    }
}
