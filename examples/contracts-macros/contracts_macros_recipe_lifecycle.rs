//! # Contracts-Macros Recipe Lifecycle State Machine
//!
//! Track lifecycle transitions: Draft → Beta → Stable → Deprecated.
//! Validates legal transitions; rejects skipping or going backwards.
//!
//! Demonstrates the **CMM.53** recipe for PMAT-175 (catalog crosses 1200).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: rust-lang RFC stability annotations.
//!
//! Run with: cargo run --example contracts_macros_recipe_lifecycle
//!
//! Added by PMAT-175 (catalog 1198→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifecycle {
    Draft,
    Beta,
    Stable,
    Deprecated,
}

#[derive(Debug, PartialEq)]
pub enum TransitionVerdict {
    Promoted { new_state: Lifecycle },
    InvalidTransition { from: Lifecycle, to: Lifecycle },
    NoChange,
}

pub fn transition(current: Lifecycle, target: Lifecycle) -> TransitionVerdict {
    if current == target {
        return TransitionVerdict::NoChange;
    }
    let allowed = matches!(
        (current, target),
        (Lifecycle::Draft, Lifecycle::Beta | Lifecycle::Deprecated)
            | (Lifecycle::Beta, Lifecycle::Stable | Lifecycle::Deprecated)
            | (Lifecycle::Stable, Lifecycle::Deprecated)
    );
    if allowed {
        TransitionVerdict::Promoted { new_state: target }
    } else {
        TransitionVerdict::InvalidTransition {
            from: current,
            to: target,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_lifecycle")?;

    println!(
        "draft→beta: {:?}",
        transition(Lifecycle::Draft, Lifecycle::Beta)
    );
    println!(
        "beta→stable: {:?}",
        transition(Lifecycle::Beta, Lifecycle::Stable)
    );
    println!(
        "stable→draft: {:?}",
        transition(Lifecycle::Stable, Lifecycle::Draft)
    );
    println!(
        "draft→stable (skip): {:?}",
        transition(Lifecycle::Draft, Lifecycle::Stable)
    );
    println!(
        "no change: {:?}",
        transition(Lifecycle::Beta, Lifecycle::Beta)
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
    fn draft_to_beta_ok() {
        let v = transition(Lifecycle::Draft, Lifecycle::Beta);
        if let TransitionVerdict::Promoted { new_state } = v {
            assert_eq!(new_state, Lifecycle::Beta);
        }
    }

    #[test]
    fn beta_to_stable_ok() {
        assert!(matches!(
            transition(Lifecycle::Beta, Lifecycle::Stable),
            TransitionVerdict::Promoted { .. }
        ));
    }

    #[test]
    fn stable_to_deprecated_ok() {
        assert!(matches!(
            transition(Lifecycle::Stable, Lifecycle::Deprecated),
            TransitionVerdict::Promoted { .. }
        ));
    }

    #[test]
    fn beta_to_deprecated_ok() {
        // Allowed: drop a beta directly.
        assert!(matches!(
            transition(Lifecycle::Beta, Lifecycle::Deprecated),
            TransitionVerdict::Promoted { .. }
        ));
    }

    #[test]
    fn draft_to_deprecated_ok() {
        assert!(matches!(
            transition(Lifecycle::Draft, Lifecycle::Deprecated),
            TransitionVerdict::Promoted { .. }
        ));
    }

    #[test]
    fn draft_to_stable_blocked() {
        let v = transition(Lifecycle::Draft, Lifecycle::Stable);
        assert!(matches!(v, TransitionVerdict::InvalidTransition { .. }));
    }

    #[test]
    fn stable_to_draft_blocked() {
        let v = transition(Lifecycle::Stable, Lifecycle::Draft);
        assert!(matches!(v, TransitionVerdict::InvalidTransition { .. }));
    }

    #[test]
    fn deprecated_no_outgoing() {
        let v = transition(Lifecycle::Deprecated, Lifecycle::Stable);
        assert!(matches!(v, TransitionVerdict::InvalidTransition { .. }));
    }

    #[test]
    fn no_change_recognized() {
        assert_eq!(
            transition(Lifecycle::Beta, Lifecycle::Beta),
            TransitionVerdict::NoChange
        );
    }

    #[test]
    fn beta_to_draft_blocked() {
        let v = transition(Lifecycle::Beta, Lifecycle::Draft);
        assert!(matches!(v, TransitionVerdict::InvalidTransition { .. }));
    }

    #[test]
    fn deterministic() {
        let a = transition(Lifecycle::Draft, Lifecycle::Beta);
        let b = transition(Lifecycle::Draft, Lifecycle::Beta);
        assert_eq!(a, b);
    }
}
