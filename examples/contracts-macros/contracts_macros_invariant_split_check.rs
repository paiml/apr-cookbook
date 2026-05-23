//! # Contracts-Macros Invariant Split Check
//!
//! Verify each compound invariant decomposes into atomic sub-invariants
//! whose IDs all exist in the registry. Returns sorted unresolved
//! sub-IDs.
//!
//! Demonstrates the **CMM.177** recipe for PMAT-216 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: BDD given/when/then atomic-step decomposition; Coq tactic
//!  decomposition (`split`/`destruct`).
//!
//! Run with: cargo run --example contracts_macros_invariant_split_check
//!
//! Added by PMAT-216 (catalog 1567→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum SplitVerdict {
    Ok {
        unresolved_subs: Vec<String>,
        resolved_count: u32,
    },
    InvalidConfig,
}

/// `compounds`: list of (compound_id, list of sub-ids).
/// `registry`: known atomic sub-ids.
pub fn check(compounds: &[(&str, Vec<&str>)], registry: &[&str]) -> SplitVerdict {
    if compounds.is_empty() || registry.is_empty() {
        return SplitVerdict::InvalidConfig;
    }
    let known: BTreeSet<&str> = registry.iter().copied().collect();
    let mut unresolved: BTreeSet<String> = BTreeSet::new();
    let mut resolved = 0u32;
    for (_, subs) in compounds {
        for s in subs {
            if known.contains(*s) {
                resolved += 1;
            } else {
                unresolved.insert((*s).to_string());
            }
        }
    }
    SplitVerdict::Ok {
        unresolved_subs: unresolved.into_iter().collect(),
        resolved_count: resolved,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_invariant_split_check")?;

    let compounds = vec![("c1", vec!["a", "b"]), ("c2", vec!["a", "x"])];
    let registry = ["a", "b"];
    println!("check: {:?}", check(&compounds, &registry));
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
    fn all_known_no_unresolved() {
        let compounds = vec![("c", vec!["a", "b"])];
        let v = check(&compounds, &["a", "b"]);
        if let SplitVerdict::Ok {
            unresolved_subs, ..
        } = v
        {
            assert!(unresolved_subs.is_empty());
        }
    }

    #[test]
    fn unknown_sub_flagged() {
        let compounds = vec![("c", vec!["x"])];
        let v = check(&compounds, &["a"]);
        if let SplitVerdict::Ok {
            unresolved_subs, ..
        } = v
        {
            assert_eq!(unresolved_subs, vec!["x".to_string()]);
        }
    }

    #[test]
    fn empty_compounds_rejected() {
        assert_eq!(check(&[], &["a"]), SplitVerdict::InvalidConfig);
    }

    #[test]
    fn empty_registry_rejected() {
        let compounds: Vec<(&str, Vec<&str>)> = vec![("c", vec!["a"])];
        assert_eq!(check(&compounds, &[]), SplitVerdict::InvalidConfig);
    }

    #[test]
    fn resolved_count_correct() {
        let compounds = vec![("c1", vec!["a", "x"]), ("c2", vec!["a", "b"])];
        let v = check(&compounds, &["a", "b"]);
        if let SplitVerdict::Ok { resolved_count, .. } = v {
            assert_eq!(resolved_count, 3);
        }
    }

    #[test]
    fn deterministic() {
        let compounds = vec![("c", vec!["a"])];
        let r1 = check(&compounds, &["a"]);
        let r2 = check(&compounds, &["a"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn unresolved_sorted_dedup() {
        let compounds = vec![("c1", vec!["zeta"]), ("c2", vec!["alpha", "zeta"])];
        let v = check(&compounds, &["a"]);
        if let SplitVerdict::Ok {
            unresolved_subs, ..
        } = v
        {
            assert_eq!(
                unresolved_subs,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn empty_subs_handled() {
        let compounds: Vec<(&str, Vec<&str>)> = vec![("c", vec![])];
        let v = check(&compounds, &["a"]);
        if let SplitVerdict::Ok { resolved_count, .. } = v {
            assert_eq!(resolved_count, 0);
        }
    }

    #[test]
    fn many_compounds_handled() {
        let compounds: Vec<(&str, Vec<&str>)> = (0..30).map(|_| ("c", vec!["a"])).collect();
        let v = check(&compounds, &["a"]);
        if let SplitVerdict::Ok { resolved_count, .. } = v {
            assert_eq!(resolved_count, 30);
        }
    }

    #[test]
    fn case_sensitive_sub() {
        let compounds = vec![("c", vec!["A"])];
        let v = check(&compounds, &["a"]);
        if let SplitVerdict::Ok {
            unresolved_subs, ..
        } = v
        {
            assert_eq!(unresolved_subs, vec!["A".to_string()]);
        }
    }

    #[test]
    fn unicode_sub_supported() {
        let compounds = vec![("c", vec!["café"])];
        let v = check(&compounds, &["café"]);
        if let SplitVerdict::Ok { resolved_count, .. } = v {
            assert_eq!(resolved_count, 1);
        }
    }
}
