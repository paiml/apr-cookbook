//! # apr pipeline status — Resource State Machine
//!
//! `apr pipeline status` reports each resource as one of: `pending`,
//! `running`, `converged`, `failed`, `skipped`. This recipe encodes the
//! valid state transitions and asserts the contract: `failed` is
//! terminal, `converged` is terminal, `pending → running → converged|failed`,
//! invalid transitions reject.
//!
//! Demonstrates the **PIPELINE.14** recipe for PMAT-107 (apr pipeline coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PIPELINE-002 + finite-state-machine convention
//!
//! Run with: cargo run --example cli_pipeline_status_state_machine
//!
//! Added by PMAT-107 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceState {
    Pending,
    Running,
    Converged,
    Failed,
    Skipped,
}

#[derive(Debug, PartialEq)]
pub enum TransitionVerdict {
    Ok,
    InvalidTransition {
        from: ResourceState,
        to: ResourceState,
    },
    TerminalState {
        state: ResourceState,
    },
}

pub fn validate_transition(from: ResourceState, to: ResourceState) -> TransitionVerdict {
    if matches!(
        from,
        ResourceState::Converged | ResourceState::Failed | ResourceState::Skipped
    ) {
        return TransitionVerdict::TerminalState { state: from };
    }
    let allowed = matches!(
        (from, to),
        (
            ResourceState::Pending,
            ResourceState::Running | ResourceState::Skipped
        ) | (
            ResourceState::Running,
            ResourceState::Converged | ResourceState::Failed
        )
    );
    if allowed {
        TransitionVerdict::Ok
    } else {
        TransitionVerdict::InvalidTransition { from, to }
    }
}

pub fn is_terminal(state: ResourceState) -> bool {
    matches!(
        state,
        ResourceState::Converged | ResourceState::Failed | ResourceState::Skipped
    )
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_pipeline_status_state_machine")?;

    let cases = [
        (ResourceState::Pending, ResourceState::Running),
        (ResourceState::Pending, ResourceState::Skipped),
        (ResourceState::Running, ResourceState::Converged),
        (ResourceState::Running, ResourceState::Failed),
        (ResourceState::Converged, ResourceState::Pending),
        (ResourceState::Failed, ResourceState::Running),
        (ResourceState::Pending, ResourceState::Converged),
    ];
    for (from, to) in cases {
        println!("{from:?} → {to:?}  →  {:?}", validate_transition(from, to));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn machine_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn pending_to_running_allowed() {
        assert_eq!(
            validate_transition(ResourceState::Pending, ResourceState::Running),
            TransitionVerdict::Ok
        );
    }

    #[test]
    fn pending_to_skipped_allowed() {
        // Skipping a never-started resource is fine.
        assert_eq!(
            validate_transition(ResourceState::Pending, ResourceState::Skipped),
            TransitionVerdict::Ok
        );
    }

    #[test]
    fn running_to_converged_allowed() {
        assert_eq!(
            validate_transition(ResourceState::Running, ResourceState::Converged),
            TransitionVerdict::Ok
        );
    }

    #[test]
    fn running_to_failed_allowed() {
        assert_eq!(
            validate_transition(ResourceState::Running, ResourceState::Failed),
            TransitionVerdict::Ok
        );
    }

    #[test]
    fn pending_directly_to_converged_rejected() {
        // Must transition through Running first.
        let v = validate_transition(ResourceState::Pending, ResourceState::Converged);
        assert!(matches!(v, TransitionVerdict::InvalidTransition { .. }));
    }

    #[test]
    fn converged_is_terminal() {
        for to in [
            ResourceState::Pending,
            ResourceState::Running,
            ResourceState::Failed,
        ] {
            let v = validate_transition(ResourceState::Converged, to);
            assert!(matches!(v, TransitionVerdict::TerminalState { .. }));
        }
    }

    #[test]
    fn failed_is_terminal() {
        let v = validate_transition(ResourceState::Failed, ResourceState::Running);
        assert!(matches!(v, TransitionVerdict::TerminalState { .. }));
    }

    #[test]
    fn is_terminal_identifies_terminal_states() {
        assert!(!is_terminal(ResourceState::Pending));
        assert!(!is_terminal(ResourceState::Running));
        assert!(is_terminal(ResourceState::Converged));
        assert!(is_terminal(ResourceState::Failed));
        assert!(is_terminal(ResourceState::Skipped));
    }
}
