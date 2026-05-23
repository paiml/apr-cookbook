//! # Contracts-Macros Recipe Release Blocker
//!
//! Identify P0 obligations in `unresolved` state as release blockers.
//! Returns blocking ids and total blocker count.
//!
//! Demonstrates the **CMM.129** recipe for PMAT-200 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GitLab/Jira P0 release-blocker conventions; ITIL Major
//!  Incident Process.
//!
//! Run with: cargo run --example contracts_macros_recipe_release_blocker
//!
//! Added by PMAT-200 (catalog 1423→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Status {
    Resolved,
    Unresolved,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Priority {
    P0,
    P1,
    P2,
    P3,
}

#[derive(Debug, PartialEq)]
pub enum BlockerVerdict {
    Ok {
        blockers: Vec<String>,
        blocker_count: u32,
    },
    InvalidConfig,
}

pub fn audit(items: &[(&str, Priority, Status)]) -> BlockerVerdict {
    if items.is_empty() {
        return BlockerVerdict::InvalidConfig;
    }
    let mut blockers: Vec<String> = items
        .iter()
        .filter(|(_, p, s)| *p == Priority::P0 && *s == Status::Unresolved)
        .map(|(id, _, _)| (*id).to_string())
        .collect();
    blockers.sort();
    let blocker_count = blockers.len() as u32;
    BlockerVerdict::Ok {
        blockers,
        blocker_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_release_blocker")?;

    let items = [
        ("o1", Priority::P0, Status::Unresolved),
        ("o2", Priority::P1, Status::Unresolved),
        ("o3", Priority::P0, Status::Resolved),
    ];
    println!("audit: {:?}", audit(&items));
    println!("invalid: {:?}", audit(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auditor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn p0_unresolved_is_blocker() {
        let items = [("o", Priority::P0, Status::Unresolved)];
        let v = audit(&items);
        if let BlockerVerdict::Ok {
            blockers,
            blocker_count,
        } = v
        {
            assert_eq!(blockers, vec!["o".to_string()]);
            assert_eq!(blocker_count, 1);
        }
    }

    #[test]
    fn p0_resolved_not_blocker() {
        let items = [("o", Priority::P0, Status::Resolved)];
        let v = audit(&items);
        if let BlockerVerdict::Ok { blockers, .. } = v {
            assert!(blockers.is_empty());
        }
    }

    #[test]
    fn p1_unresolved_not_blocker() {
        let items = [("o", Priority::P1, Status::Unresolved)];
        let v = audit(&items);
        if let BlockerVerdict::Ok { blockers, .. } = v {
            assert!(blockers.is_empty());
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), BlockerVerdict::InvalidConfig);
    }

    #[test]
    fn blockers_sorted() {
        let items = [
            ("zeta", Priority::P0, Status::Unresolved),
            ("alpha", Priority::P0, Status::Unresolved),
        ];
        let v = audit(&items);
        if let BlockerVerdict::Ok { blockers, .. } = v {
            assert_eq!(blockers, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let items = [("o", Priority::P0, Status::Unresolved)];
        let r1 = audit(&items);
        let r2 = audit(&items);
        assert_eq!(r1, r2);
    }

    #[test]
    fn count_matches_blockers_len() {
        let items = [
            ("a", Priority::P0, Status::Unresolved),
            ("b", Priority::P0, Status::Unresolved),
            ("c", Priority::P1, Status::Unresolved),
        ];
        let v = audit(&items);
        if let BlockerVerdict::Ok {
            blockers,
            blocker_count,
        } = v
        {
            assert_eq!(blockers.len() as u32, blocker_count);
        }
    }

    #[test]
    fn no_blockers_zero_count() {
        let items = [
            ("a", Priority::P1, Status::Unresolved),
            ("b", Priority::P0, Status::Resolved),
        ];
        let v = audit(&items);
        if let BlockerVerdict::Ok { blocker_count, .. } = v {
            assert_eq!(blocker_count, 0);
        }
    }

    #[test]
    fn p2_p3_not_blocker() {
        let items = [
            ("p2", Priority::P2, Status::Unresolved),
            ("p3", Priority::P3, Status::Unresolved),
        ];
        let v = audit(&items);
        if let BlockerVerdict::Ok { blockers, .. } = v {
            assert!(blockers.is_empty());
        }
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<(&str, Priority, Status)> = (0..20)
            .map(|_| ("o", Priority::P0, Status::Unresolved))
            .collect();
        let v = audit(&items);
        if let BlockerVerdict::Ok { blocker_count, .. } = v {
            assert_eq!(blocker_count, 20);
        }
    }

    #[test]
    fn mixed_blockers_filtered() {
        let items = [
            ("good", Priority::P0, Status::Resolved),
            ("blocker", Priority::P0, Status::Unresolved),
            ("low", Priority::P3, Status::Unresolved),
        ];
        let v = audit(&items);
        if let BlockerVerdict::Ok { blockers, .. } = v {
            assert_eq!(blockers, vec!["blocker".to_string()]);
        }
    }
}
