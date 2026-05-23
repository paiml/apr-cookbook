//! # Contracts-Macros Recipe Archive-Safe Check
//!
//! Verify that archiving a recipe is safe: no live dependents, no
//! pending PRs, no quarantined invariants. Returns categorical
//! verdict.
//!
//! Demonstrates the **CMM.179** recipe for PMAT-217 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: cargo yank pre-flight policy; PEP 541 frozen-package
//!  archive rules.
//!
//! Run with: cargo run --example contracts_macros_recipe_archive_safe
//!
//! Added by PMAT-217 (catalog 1576→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ArchiveSafeVerdict {
    Safe,
    Blocked { reasons: Vec<String> },
    InvalidConfig,
}

pub fn check(
    live_dependents: u32,
    pending_prs: u32,
    quarantined_invariants: u32,
) -> ArchiveSafeVerdict {
    let mut reasons: Vec<String> = Vec::new();
    if live_dependents > 0 {
        reasons.push(format!("live_dependents:{live_dependents}"));
    }
    if pending_prs > 0 {
        reasons.push(format!("pending_prs:{pending_prs}"));
    }
    if quarantined_invariants > 0 {
        reasons.push(format!("quarantined_invariants:{quarantined_invariants}"));
    }
    if reasons.is_empty() {
        ArchiveSafeVerdict::Safe
    } else {
        ArchiveSafeVerdict::Blocked { reasons }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_archive_safe")?;

    println!("safe: {:?}", check(0, 0, 0));
    println!("blocked: {:?}", check(2, 1, 0));
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
    fn all_zero_safe() {
        assert_eq!(check(0, 0, 0), ArchiveSafeVerdict::Safe);
    }

    #[test]
    fn live_dependents_blocked() {
        let v = check(2, 0, 0);
        if let ArchiveSafeVerdict::Blocked { reasons } = v {
            assert!(reasons.iter().any(|r| r.starts_with("live_dependents")));
        }
    }

    #[test]
    fn pending_prs_blocked() {
        let v = check(0, 1, 0);
        if let ArchiveSafeVerdict::Blocked { reasons } = v {
            assert!(reasons.iter().any(|r| r.starts_with("pending_prs")));
        }
    }

    #[test]
    fn quarantined_invariants_blocked() {
        let v = check(0, 0, 3);
        if let ArchiveSafeVerdict::Blocked { reasons } = v {
            assert!(reasons
                .iter()
                .any(|r| r.starts_with("quarantined_invariants")));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check(0, 0, 0);
        let r2 = check(0, 0, 0);
        assert_eq!(r1, r2);
    }

    #[test]
    fn multiple_reasons_listed() {
        let v = check(2, 1, 1);
        if let ArchiveSafeVerdict::Blocked { reasons } = v {
            assert_eq!(reasons.len(), 3);
        }
    }

    #[test]
    fn one_dependent_blocks() {
        let v = check(1, 0, 0);
        assert!(matches!(v, ArchiveSafeVerdict::Blocked { .. }));
    }

    #[test]
    fn one_pr_blocks() {
        let v = check(0, 1, 0);
        assert!(matches!(v, ArchiveSafeVerdict::Blocked { .. }));
    }

    #[test]
    fn boundary_at_zero_safe() {
        assert_eq!(check(0, 0, 0), ArchiveSafeVerdict::Safe);
    }

    #[test]
    fn high_counts_blocked() {
        let v = check(1000, 100, 50);
        if let ArchiveSafeVerdict::Blocked { reasons } = v {
            assert_eq!(reasons.len(), 3);
        }
    }

    #[test]
    fn reason_includes_count() {
        let v = check(5, 0, 0);
        if let ArchiveSafeVerdict::Blocked { reasons } = v {
            assert_eq!(reasons[0], "live_dependents:5");
        }
    }

    #[test]
    fn reasons_in_priority_order() {
        // dependents → prs → quarantine
        let v = check(1, 1, 1);
        if let ArchiveSafeVerdict::Blocked { reasons } = v {
            assert!(reasons[0].starts_with("live_dependents"));
            assert!(reasons[1].starts_with("pending_prs"));
            assert!(reasons[2].starts_with("quarantined"));
        }
    }
}
