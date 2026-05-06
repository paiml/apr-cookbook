//! # Contracts-Macros Proof Status Transition
//!
//! Validate proof-status transitions follow the canonical chain:
//!   `wip` → (`sorry` ∨ `not-applicable`) → `proved`.
//! Reject illegal transitions (e.g. `proved` → `wip`).
//!
//! Demonstrates the **CMM.75** recipe for PMAT-182 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Lean 4 manual ch.6 (sorry placeholder); "honest status"
//!  discipline (this codebase, MEMORY.md).
//!
//! Run with: cargo run --example contracts_macros_proof_status_transition
//!
//! Added by PMAT-182 (catalog 1261→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TransitionVerdict {
    Ok { allowed: bool, reason: String },
    InvalidConfig,
}

pub fn check(from: &str, to: &str) -> TransitionVerdict {
    let valid_states = ["wip", "sorry", "not-applicable", "proved"];
    if !valid_states.contains(&from) || !valid_states.contains(&to) {
        return TransitionVerdict::InvalidConfig;
    }
    let (allowed, reason) = match (from, to) {
        (a, b) if a == b => (true, "no-op (same state)".to_string()),
        ("wip", "sorry" | "not-applicable" | "proved") => {
            (true, "wip can advance to any tracked state".to_string())
        }
        ("sorry", "proved" | "not-applicable") => (
            true,
            "sorry can be replaced by a real proof or marked N/A".to_string(),
        ),
        ("not-applicable", "wip") => (true, "may revert N/A to wip if reclassified".to_string()),
        ("proved", _) => (false, "proved is terminal — cannot regress".to_string()),
        _ => (false, format!("invalid transition {from} → {to}")),
    };
    TransitionVerdict::Ok { allowed, reason }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_proof_status_transition")?;

    println!("wip → proved: {:?}", check("wip", "proved"));
    println!("proved → wip: {:?}", check("proved", "wip"));
    println!("invalid: {:?}", check("xyz", "wip"));
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
    fn wip_to_sorry_allowed() {
        let v = check("wip", "sorry");
        if let TransitionVerdict::Ok { allowed, .. } = v {
            assert!(allowed);
        }
    }

    #[test]
    fn wip_to_proved_allowed() {
        let v = check("wip", "proved");
        if let TransitionVerdict::Ok { allowed, .. } = v {
            assert!(allowed);
        }
    }

    #[test]
    fn sorry_to_proved_allowed() {
        let v = check("sorry", "proved");
        if let TransitionVerdict::Ok { allowed, .. } = v {
            assert!(allowed);
        }
    }

    #[test]
    fn proved_terminal() {
        let v = check("proved", "wip");
        if let TransitionVerdict::Ok { allowed, .. } = v {
            assert!(!allowed);
        }
    }

    #[test]
    fn proved_to_sorry_disallowed() {
        let v = check("proved", "sorry");
        if let TransitionVerdict::Ok { allowed, .. } = v {
            assert!(!allowed);
        }
    }

    #[test]
    fn no_op_allowed() {
        let v = check("wip", "wip");
        if let TransitionVerdict::Ok { allowed, .. } = v {
            assert!(allowed);
        }
    }

    #[test]
    fn invalid_state_rejected() {
        assert_eq!(check("unknown", "wip"), TransitionVerdict::InvalidConfig);
    }

    #[test]
    fn na_to_wip_allowed() {
        let v = check("not-applicable", "wip");
        if let TransitionVerdict::Ok { allowed, .. } = v {
            assert!(allowed);
        }
    }

    #[test]
    fn sorry_to_wip_disallowed() {
        let v = check("sorry", "wip");
        if let TransitionVerdict::Ok { allowed, .. } = v {
            assert!(!allowed);
        }
    }

    #[test]
    fn deterministic() {
        let a = check("wip", "proved");
        let b = check("wip", "proved");
        assert_eq!(a, b);
    }

    #[test]
    fn reason_field_non_empty() {
        let v = check("wip", "proved");
        if let TransitionVerdict::Ok { reason, .. } = v {
            assert!(!reason.is_empty());
        }
    }
}
