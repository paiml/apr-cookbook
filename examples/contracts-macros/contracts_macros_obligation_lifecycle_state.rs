//! # Contracts-Macros Obligation Lifecycle State
//!
//! Validate obligation state transitions through a strict lifecycle:
//! Draft → Open → Implemented → Closed. Reject backward transitions.
//!
//! Demonstrates the **CMM.109** recipe for PMAT-194 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GitHub issue lifecycle conventions; ITIL incident
//!  state-machine.
//!
//! Run with: cargo run --example contracts_macros_obligation_lifecycle_state
//!
//! Added by PMAT-194 (catalog 1369→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy, PartialOrd, Ord, Eq)]
pub enum LifecycleState {
    Draft,
    Open,
    Implemented,
    Closed,
}

#[derive(Debug, PartialEq)]
pub enum LifecycleVerdict {
    Ok { allowed: bool, reason: String },
    InvalidConfig,
}

pub fn check(from: LifecycleState, to: LifecycleState) -> LifecycleVerdict {
    if from == to {
        return LifecycleVerdict::Ok {
            allowed: true,
            reason: "same state".to_string(),
        };
    }
    let allowed = to > from;
    let reason = if allowed {
        format!("forward {from:?} → {to:?}")
    } else {
        format!("backward {from:?} → {to:?} not allowed")
    };
    LifecycleVerdict::Ok { allowed, reason }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_lifecycle_state")?;

    println!(
        "draft → open: {:?}",
        check(LifecycleState::Draft, LifecycleState::Open)
    );
    println!(
        "closed → open: {:?}",
        check(LifecycleState::Closed, LifecycleState::Open)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn forward_transition_allowed() {
        let v = check(LifecycleState::Draft, LifecycleState::Open);
        if let LifecycleVerdict::Ok { allowed, .. } = v {
            assert!(allowed);
        }
    }

    #[test]
    fn backward_transition_disallowed() {
        let v = check(LifecycleState::Closed, LifecycleState::Open);
        if let LifecycleVerdict::Ok { allowed, .. } = v {
            assert!(!allowed);
        }
    }

    #[test]
    fn same_state_allowed() {
        let v = check(LifecycleState::Open, LifecycleState::Open);
        if let LifecycleVerdict::Ok { allowed, .. } = v {
            assert!(allowed);
        }
    }

    #[test]
    fn skip_transition_allowed() {
        let v = check(LifecycleState::Draft, LifecycleState::Closed);
        if let LifecycleVerdict::Ok { allowed, .. } = v {
            assert!(allowed);
        }
    }

    #[test]
    fn closed_to_implemented_disallowed() {
        let v = check(LifecycleState::Closed, LifecycleState::Implemented);
        if let LifecycleVerdict::Ok { allowed, .. } = v {
            assert!(!allowed);
        }
    }

    #[test]
    fn ordering_chain() {
        assert!(LifecycleState::Open > LifecycleState::Draft);
        assert!(LifecycleState::Implemented > LifecycleState::Open);
        assert!(LifecycleState::Closed > LifecycleState::Implemented);
    }

    #[test]
    fn deterministic() {
        let r1 = check(LifecycleState::Draft, LifecycleState::Open);
        let r2 = check(LifecycleState::Draft, LifecycleState::Open);
        assert_eq!(r1, r2);
    }

    #[test]
    fn reason_field_non_empty() {
        let v = check(LifecycleState::Draft, LifecycleState::Open);
        if let LifecycleVerdict::Ok { reason, .. } = v {
            assert!(!reason.is_empty());
        }
    }

    #[test]
    fn implemented_to_closed_allowed() {
        let v = check(LifecycleState::Implemented, LifecycleState::Closed);
        if let LifecycleVerdict::Ok { allowed, .. } = v {
            assert!(allowed);
        }
    }

    #[test]
    fn open_to_draft_disallowed() {
        let v = check(LifecycleState::Open, LifecycleState::Draft);
        if let LifecycleVerdict::Ok { allowed, .. } = v {
            assert!(!allowed);
        }
    }

    #[test]
    fn implemented_to_open_disallowed() {
        let v = check(LifecycleState::Implemented, LifecycleState::Open);
        if let LifecycleVerdict::Ok { allowed, .. } = v {
            assert!(!allowed);
        }
    }
}
