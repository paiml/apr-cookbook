//! # MCP Session Lifecycle State Machine
//!
//! MCP session: Init → HandshakeSent → Ready → ToolCallActive → Ready
//! ↔ ToolCallActive → Closed. This recipe codifies allowed transitions
//! + the handshake-version negotiation.
//!
//! Demonstrates the **MCP.13** recipe for PMAT-132 (mcp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: MCP spec § Session Initialization.
//!
//! Run with: cargo run --example mcp_session_lifecycle
//!
//! Added by PMAT-132 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Init,
    HandshakeSent,
    Ready,
    ToolCallActive,
    Closed,
}

#[derive(Debug, PartialEq)]
pub enum TransitionVerdict {
    Ok(SessionState),
    InvalidTransition {
        from: SessionState,
        to: SessionState,
    },
    AlreadyClosed,
}

pub fn transition(from: SessionState, to: SessionState) -> TransitionVerdict {
    if from == SessionState::Closed {
        return TransitionVerdict::AlreadyClosed;
    }
    let valid = matches!(
        (from, to),
        (SessionState::Init, SessionState::HandshakeSent)
            | (
                SessionState::HandshakeSent | SessionState::ToolCallActive,
                SessionState::Ready
            )
            | (SessionState::Ready, SessionState::ToolCallActive)
            | (
                SessionState::HandshakeSent | SessionState::Ready | SessionState::ToolCallActive,
                SessionState::Closed
            )
    );
    if valid {
        TransitionVerdict::Ok(to)
    } else {
        TransitionVerdict::InvalidTransition { from, to }
    }
}

#[derive(Debug, PartialEq)]
pub enum NegotiateVerdict {
    Ok { agreed_version: u32 },
    NoOverlap,
    InvalidVersion,
}

pub fn negotiate_version(client_supported: &[u32], server_supported: &[u32]) -> NegotiateVerdict {
    if client_supported.is_empty() || server_supported.is_empty() {
        return NegotiateVerdict::InvalidVersion;
    }
    let client_set: std::collections::BTreeSet<u32> = client_supported.iter().copied().collect();
    let highest_common = server_supported
        .iter()
        .copied()
        .filter(|v| client_set.contains(v))
        .max();
    match highest_common {
        Some(v) => NegotiateVerdict::Ok { agreed_version: v },
        None => NegotiateVerdict::NoOverlap,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mcp_session_lifecycle")?;

    let path = [
        SessionState::Init,
        SessionState::HandshakeSent,
        SessionState::Ready,
        SessionState::ToolCallActive,
        SessionState::Ready,
        SessionState::Closed,
    ];
    for w in path.windows(2) {
        println!("{:?} → {:?}: {:?}", w[0], w[1], transition(w[0], w[1]));
    }

    println!(
        "negotiate [1,2,3] vs [2,3,4]: {:?}",
        negotiate_version(&[1, 2, 3], &[2, 3, 4])
    );
    println!(
        "no overlap [1,2] vs [3,4]: {:?}",
        negotiate_version(&[1, 2], &[3, 4])
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lifecycle_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn init_to_handshake_allowed() {
        let v = transition(SessionState::Init, SessionState::HandshakeSent);
        assert!(matches!(v, TransitionVerdict::Ok(_)));
    }

    #[test]
    fn handshake_to_ready_allowed() {
        let v = transition(SessionState::HandshakeSent, SessionState::Ready);
        assert!(matches!(v, TransitionVerdict::Ok(_)));
    }

    #[test]
    fn ready_to_tool_call_allowed() {
        let v = transition(SessionState::Ready, SessionState::ToolCallActive);
        assert!(matches!(v, TransitionVerdict::Ok(_)));
    }

    #[test]
    fn tool_call_back_to_ready_allowed() {
        let v = transition(SessionState::ToolCallActive, SessionState::Ready);
        assert!(matches!(v, TransitionVerdict::Ok(_)));
    }

    #[test]
    fn skipping_handshake_rejected() {
        let v = transition(SessionState::Init, SessionState::Ready);
        assert!(matches!(v, TransitionVerdict::InvalidTransition { .. }));
    }

    #[test]
    fn closed_session_no_transitions() {
        let v = transition(SessionState::Closed, SessionState::Init);
        assert_eq!(v, TransitionVerdict::AlreadyClosed);
    }

    #[test]
    fn any_state_can_close() {
        for from in [
            SessionState::HandshakeSent,
            SessionState::Ready,
            SessionState::ToolCallActive,
        ] {
            let v = transition(from, SessionState::Closed);
            assert!(matches!(v, TransitionVerdict::Ok(_)), "from {from:?}");
        }
    }

    #[test]
    fn negotiate_picks_highest_common_version() {
        let v = negotiate_version(&[1, 2, 3, 4], &[2, 4, 5]);
        assert_eq!(v, NegotiateVerdict::Ok { agreed_version: 4 });
    }

    #[test]
    fn negotiate_no_overlap_rejected() {
        assert_eq!(
            negotiate_version(&[1, 2], &[3, 4]),
            NegotiateVerdict::NoOverlap
        );
    }

    #[test]
    fn negotiate_empty_lists_invalid() {
        assert_eq!(
            negotiate_version(&[], &[1]),
            NegotiateVerdict::InvalidVersion
        );
        assert_eq!(
            negotiate_version(&[1], &[]),
            NegotiateVerdict::InvalidVersion
        );
    }
}
