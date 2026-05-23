//! # Contracts-Macros Obligation Subset Check
//!
//! Verify obligation set A is a subset of B (every element in A
//! appears in B). Returns missing-from-B elements + subset flag.
//!
//! Demonstrates the **CMM.124** recipe for PMAT-199 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: subset relation (set theory ZFC); contract refinement
//!  (Liskov substitution principle).
//!
//! Run with: cargo run --example contracts_macros_obligation_subset_check
//!
//! Added by PMAT-199 (catalog 1414→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum SubsetVerdict {
    Ok {
        is_subset: bool,
        missing_from_super: Vec<String>,
    },
    InvalidConfig,
}

pub fn check(subset: &[&str], superset: &[&str]) -> SubsetVerdict {
    if subset.is_empty() && superset.is_empty() {
        return SubsetVerdict::InvalidConfig;
    }
    let super_set: BTreeSet<&str> = superset.iter().copied().collect();
    let mut missing: Vec<String> = subset
        .iter()
        .filter(|s| !super_set.contains(*s))
        .map(|s| (*s).to_string())
        .collect();
    missing.sort();
    missing.dedup();
    SubsetVerdict::Ok {
        is_subset: missing.is_empty(),
        missing_from_super: missing,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_subset_check")?;

    println!("subset: {:?}", check(&["a", "b"], &["a", "b", "c"]));
    println!("not subset: {:?}", check(&["a", "x"], &["a", "b"]));
    println!("invalid: {:?}", check(&[], &[]));
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
    fn proper_subset_passes() {
        let v = check(&["a", "b"], &["a", "b", "c"]);
        if let SubsetVerdict::Ok { is_subset, .. } = v {
            assert!(is_subset);
        }
    }

    #[test]
    fn missing_element_flagged() {
        let v = check(&["a", "x"], &["a", "b"]);
        if let SubsetVerdict::Ok {
            is_subset,
            missing_from_super,
        } = v
        {
            assert!(!is_subset);
            assert_eq!(missing_from_super, vec!["x".to_string()]);
        }
    }

    #[test]
    fn empty_subset_is_subset() {
        let v = check(&[], &["a"]);
        if let SubsetVerdict::Ok { is_subset, .. } = v {
            assert!(is_subset);
        }
    }

    #[test]
    fn equal_sets_subset() {
        let v = check(&["a", "b"], &["a", "b"]);
        if let SubsetVerdict::Ok { is_subset, .. } = v {
            assert!(is_subset);
        }
    }

    #[test]
    fn both_empty_rejected() {
        assert_eq!(check(&[], &[]), SubsetVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = check(&["a"], &["a"]);
        let r2 = check(&["a"], &["a"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn missing_sorted() {
        let v = check(&["zeta", "alpha"], &[]);
        if let SubsetVerdict::Ok {
            missing_from_super, ..
        } = v
        {
            assert_eq!(
                missing_from_super,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn case_sensitive() {
        let v = check(&["A"], &["a"]);
        if let SubsetVerdict::Ok { is_subset, .. } = v {
            assert!(!is_subset);
        }
    }

    #[test]
    fn duplicate_subset_dedup() {
        let v = check(&["a", "a", "b"], &["a"]);
        if let SubsetVerdict::Ok {
            missing_from_super, ..
        } = v
        {
            assert_eq!(missing_from_super, vec!["b".to_string()]);
        }
    }

    #[test]
    fn supeset_superset_of_subset() {
        let v = check(&["a"], &["a", "b", "c", "d"]);
        if let SubsetVerdict::Ok { is_subset, .. } = v {
            assert!(is_subset);
        }
    }

    #[test]
    fn many_elements_handled() {
        let sub: Vec<&str> = vec!["x"; 30];
        let sup = ["x"];
        let v = check(&sub, &sup);
        if let SubsetVerdict::Ok { is_subset, .. } = v {
            assert!(is_subset);
        }
    }
}
