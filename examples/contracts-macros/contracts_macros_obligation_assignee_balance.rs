//! # Contracts-Macros Obligation Assignee Balance
//!
//! Audit obligation assignments: per-owner counts must be within
//! `tolerance` of mean. Returns over/under-loaded owners.
//!
//! Demonstrates the **CMM.82** recipe for PMAT-185 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: load balancing fairness theorems (Mitzenmacher
//!  "Power of Two Choices", IEEE TPDS 12, 2001).
//!
//! Run with: cargo run --example contracts_macros_obligation_assignee_balance
//!
//! Added by PMAT-185 (catalog 1288→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum BalanceVerdict {
    Ok {
        per_owner: BTreeMap<String, u32>,
        overloaded: Vec<String>,
        underloaded: Vec<String>,
    },
    InvalidConfig,
}

pub fn audit(assignments: &[(&str, &str)], tolerance_factor: f64) -> BalanceVerdict {
    if assignments.is_empty() || tolerance_factor < 1.0 {
        return BalanceVerdict::InvalidConfig;
    }
    let mut per_owner: BTreeMap<String, u32> = BTreeMap::new();
    for (_, owner) in assignments {
        *per_owner.entry((*owner).to_string()).or_insert(0) += 1;
    }
    let mean = assignments.len() as f64 / per_owner.len() as f64;
    let max = mean * tolerance_factor;
    let min = mean / tolerance_factor;
    let mut overloaded: Vec<String> = Vec::new();
    let mut underloaded: Vec<String> = Vec::new();
    for (owner, count) in &per_owner {
        if f64::from(*count) > max {
            overloaded.push(owner.clone());
        } else if f64::from(*count) < min {
            underloaded.push(owner.clone());
        }
    }
    BalanceVerdict::Ok {
        per_owner,
        overloaded,
        underloaded,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_assignee_balance")?;

    let assignments = [
        ("o1", "alice"),
        ("o2", "alice"),
        ("o3", "bob"),
        ("o4", "alice"),
        ("o5", "alice"),
    ];
    println!("audit: {:?}", audit(&assignments, 1.5));
    println!("invalid: {:?}", audit(&[], 1.5));
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
    fn balanced_no_offenders() {
        let assignments = [("o1", "a"), ("o2", "b")];
        let v = audit(&assignments, 1.5);
        if let BalanceVerdict::Ok {
            overloaded,
            underloaded,
            ..
        } = v
        {
            assert!(overloaded.is_empty());
            assert!(underloaded.is_empty());
        }
    }

    #[test]
    fn overloaded_owner_flagged() {
        let assignments = [
            ("o1", "alice"),
            ("o2", "alice"),
            ("o3", "alice"),
            ("o4", "alice"),
            ("o5", "bob"),
        ];
        let v = audit(&assignments, 1.5);
        if let BalanceVerdict::Ok { overloaded, .. } = v {
            assert!(overloaded.contains(&"alice".to_string()));
        }
    }

    #[test]
    fn underloaded_owner_flagged() {
        let assignments = [
            ("o1", "alice"),
            ("o2", "alice"),
            ("o3", "alice"),
            ("o4", "bob"),
        ];
        // mean=2, tol=1.5, max=3.0, min=1.33 → bob (1) is below min.
        let v = audit(&assignments, 1.5);
        if let BalanceVerdict::Ok { underloaded, .. } = v {
            assert!(underloaded.contains(&"bob".to_string()));
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[], 1.5), BalanceVerdict::InvalidConfig);
    }

    #[test]
    fn tolerance_below_one_rejected() {
        let assignments = [("o", "a")];
        assert_eq!(audit(&assignments, 0.5), BalanceVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let assignments = [("o1", "a"), ("o2", "b")];
        let r1 = audit(&assignments, 1.5);
        let r2 = audit(&assignments, 1.5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn single_owner_no_offenders() {
        let assignments = [("o1", "alice"), ("o2", "alice")];
        let v = audit(&assignments, 1.5);
        if let BalanceVerdict::Ok {
            overloaded,
            underloaded,
            ..
        } = v
        {
            assert!(overloaded.is_empty());
            assert!(underloaded.is_empty());
        }
    }

    #[test]
    fn count_per_owner_correct() {
        let assignments = [("o1", "a"), ("o2", "a"), ("o3", "b")];
        let v = audit(&assignments, 1.5);
        if let BalanceVerdict::Ok { per_owner, .. } = v {
            assert_eq!(per_owner.get("a"), Some(&2));
            assert_eq!(per_owner.get("b"), Some(&1));
        }
    }

    #[test]
    fn loose_tolerance_no_offenders() {
        let assignments = [("o1", "a"), ("o2", "a"), ("o3", "b")];
        let v = audit(&assignments, 100.0);
        if let BalanceVerdict::Ok {
            overloaded,
            underloaded,
            ..
        } = v
        {
            assert!(overloaded.is_empty());
            assert!(underloaded.is_empty());
        }
    }

    #[test]
    fn very_tight_tolerance_flags_minor_imbalance() {
        let assignments = [("o1", "a"), ("o2", "a"), ("o3", "b")];
        let v = audit(&assignments, 1.01);
        if let BalanceVerdict::Ok {
            overloaded,
            underloaded,
            ..
        } = v
        {
            assert!(!overloaded.is_empty() || !underloaded.is_empty());
        }
    }

    #[test]
    fn three_owners_balanced() {
        let assignments = [
            ("o1", "a"),
            ("o2", "a"),
            ("o3", "b"),
            ("o4", "b"),
            ("o5", "c"),
            ("o6", "c"),
        ];
        let v = audit(&assignments, 1.2);
        if let BalanceVerdict::Ok {
            overloaded,
            underloaded,
            ..
        } = v
        {
            assert!(overloaded.is_empty());
            assert!(underloaded.is_empty());
        }
    }
}
