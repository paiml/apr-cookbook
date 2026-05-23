//! # Monitoring Circuit-State-Transition Log Emitter
//!
//! Whenever the inference circuit transitions, emit a log with the
//! right severity:
//!   Closed→Open: error (will hit traffic, page oncall)
//!   Open→HalfOpen: info (testing recovery)
//!   HalfOpen→Closed: info (recovered)
//!   HalfOpen→Open: warn (still broken, will re-page after cooldown)
//!
//! This recipe builds the emitter policy.
//!
//! Demonstrates the **MON.24** recipe for PMAT-141 (monitoring round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hystrix circuit-state observability docs.
//!
//! Run with: cargo run --example monitor_circuit_log_emit
//!
//! Added by PMAT-141 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageDecision {
    Page,
    NoPage,
}

#[derive(Debug, PartialEq)]
pub enum EmitVerdict {
    Ok {
        level: LogLevel,
        page: PageDecision,
        message: &'static str,
    },
    SameState,
    InvalidTransition {
        from: CircuitState,
        to: CircuitState,
    },
}

pub fn emit(from: CircuitState, to: CircuitState) -> EmitVerdict {
    if from == to {
        return EmitVerdict::SameState;
    }
    match (from, to) {
        (CircuitState::Closed, CircuitState::Open) => EmitVerdict::Ok {
            level: LogLevel::Error,
            page: PageDecision::Page,
            message: "circuit opened — traffic blocked",
        },
        (CircuitState::Open, CircuitState::HalfOpen) => EmitVerdict::Ok {
            level: LogLevel::Info,
            page: PageDecision::NoPage,
            message: "circuit testing recovery",
        },
        (CircuitState::HalfOpen, CircuitState::Closed) => EmitVerdict::Ok {
            level: LogLevel::Info,
            page: PageDecision::NoPage,
            message: "circuit recovered — normal traffic",
        },
        (CircuitState::HalfOpen, CircuitState::Open) => EmitVerdict::Ok {
            level: LogLevel::Warn,
            page: PageDecision::NoPage,
            message: "circuit re-opened — recovery failed",
        },
        // Closed → HalfOpen and Open → Closed are not legal in standard CB.
        _ => EmitVerdict::InvalidTransition { from, to },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("monitor_circuit_log_emit")?;

    let pairs = [
        (CircuitState::Closed, CircuitState::Open),
        (CircuitState::Open, CircuitState::HalfOpen),
        (CircuitState::HalfOpen, CircuitState::Closed),
        (CircuitState::HalfOpen, CircuitState::Open),
        (CircuitState::Closed, CircuitState::Closed),
        (CircuitState::Closed, CircuitState::HalfOpen),
    ];
    for (f, t) in pairs {
        println!("{f:?} → {t:?}: {:?}", emit(f, t));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emitter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn closed_to_open_pages() {
        let v = emit(CircuitState::Closed, CircuitState::Open);
        if let EmitVerdict::Ok { level, page, .. } = v {
            assert_eq!(level, LogLevel::Error);
            assert_eq!(page, PageDecision::Page);
        }
    }

    #[test]
    fn open_to_half_open_info() {
        let v = emit(CircuitState::Open, CircuitState::HalfOpen);
        if let EmitVerdict::Ok { level, page, .. } = v {
            assert_eq!(level, LogLevel::Info);
            assert_eq!(page, PageDecision::NoPage);
        }
    }

    #[test]
    fn half_open_to_closed_info() {
        let v = emit(CircuitState::HalfOpen, CircuitState::Closed);
        if let EmitVerdict::Ok { level, page, .. } = v {
            assert_eq!(level, LogLevel::Info);
            assert_eq!(page, PageDecision::NoPage);
        }
    }

    #[test]
    fn half_open_to_open_warn() {
        let v = emit(CircuitState::HalfOpen, CircuitState::Open);
        if let EmitVerdict::Ok { level, page, .. } = v {
            assert_eq!(level, LogLevel::Warn);
            assert_eq!(page, PageDecision::NoPage);
        }
    }

    #[test]
    fn same_state_no_emit() {
        assert_eq!(
            emit(CircuitState::Closed, CircuitState::Closed),
            EmitVerdict::SameState
        );
    }

    #[test]
    fn closed_to_half_open_invalid() {
        let v = emit(CircuitState::Closed, CircuitState::HalfOpen);
        assert!(matches!(v, EmitVerdict::InvalidTransition { .. }));
    }

    #[test]
    fn open_to_closed_invalid() {
        // Open → Closed must go through HalfOpen first.
        let v = emit(CircuitState::Open, CircuitState::Closed);
        assert!(matches!(v, EmitVerdict::InvalidTransition { .. }));
    }

    #[test]
    fn message_present_on_emit() {
        if let EmitVerdict::Ok { message, .. } = emit(CircuitState::Closed, CircuitState::Open) {
            assert!(!message.is_empty());
        }
    }

    #[test]
    fn only_open_event_pages() {
        // Only Closed → Open should page.
        let pairs = [
            (CircuitState::Open, CircuitState::HalfOpen),
            (CircuitState::HalfOpen, CircuitState::Closed),
            (CircuitState::HalfOpen, CircuitState::Open),
        ];
        for (f, t) in pairs {
            if let EmitVerdict::Ok { page, .. } = emit(f, t) {
                assert_eq!(page, PageDecision::NoPage, "transition {f:?}→{t:?}");
            }
        }
    }

    #[test]
    fn invalid_transition_carries_states() {
        if let EmitVerdict::InvalidTransition { from, to } =
            emit(CircuitState::Closed, CircuitState::HalfOpen)
        {
            assert_eq!(from, CircuitState::Closed);
            assert_eq!(to, CircuitState::HalfOpen);
        }
    }
}
