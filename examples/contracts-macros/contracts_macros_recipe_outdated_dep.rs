//! # Contracts-Macros Recipe Outdated Dependency
//!
//! Flag recipes referencing deprecated dependencies. Returns
//! offending recipes + per-dep deprecation count.
//!
//! Demonstrates the **CMM.132** recipe for PMAT-201 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: cargo-deny advisory checks; npm audit deprecated-pkg
//!  warnings.
//!
//! Run with: cargo run --example contracts_macros_recipe_outdated_dep
//!
//! Added by PMAT-201 (catalog 1432→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum OutdatedVerdict {
    Ok {
        offenders: Vec<String>,
        per_dep: BTreeMap<String, u32>,
    },
    InvalidConfig,
}

pub fn audit(recipes: &[(&str, Vec<&str>)], deprecated: &[&str]) -> OutdatedVerdict {
    if recipes.is_empty() {
        return OutdatedVerdict::InvalidConfig;
    }
    let dep_set: BTreeSet<&str> = deprecated.iter().copied().collect();
    let mut offenders: Vec<String> = Vec::new();
    let mut per_dep: BTreeMap<String, u32> = BTreeMap::new();
    for (name, deps) in recipes {
        let mut has_dep = false;
        for d in deps {
            if dep_set.contains(d) {
                *per_dep.entry((*d).to_string()).or_insert(0) += 1;
                has_dep = true;
            }
        }
        if has_dep {
            offenders.push((*name).to_string());
        }
    }
    offenders.sort();
    offenders.dedup();
    OutdatedVerdict::Ok { offenders, per_dep }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_outdated_dep")?;

    let recipes = vec![
        ("r1", vec!["serde", "old_lib"]),
        ("r2", vec!["serde"]),
        ("r3", vec!["old_lib", "deprecated_x"]),
    ];
    let deprecated = ["old_lib", "deprecated_x"];
    println!("audit: {:?}", audit(&recipes, &deprecated));
    println!("invalid: {:?}", audit(&[], &deprecated));
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
    fn no_dep_no_offenders() {
        let recipes = vec![("r", vec!["serde"])];
        let v = audit(&recipes, &["old"]);
        if let OutdatedVerdict::Ok { offenders, .. } = v {
            assert!(offenders.is_empty());
        }
    }

    #[test]
    fn deprecated_dep_flagged() {
        let recipes = vec![("r", vec!["old"])];
        let v = audit(&recipes, &["old"]);
        if let OutdatedVerdict::Ok { offenders, .. } = v {
            assert_eq!(offenders, vec!["r".to_string()]);
        }
    }

    #[test]
    fn empty_recipes_rejected() {
        assert_eq!(audit(&[], &["old"]), OutdatedVerdict::InvalidConfig);
    }

    #[test]
    fn per_dep_count_correct() {
        let recipes = vec![("r1", vec!["old"]), ("r2", vec!["old"])];
        let v = audit(&recipes, &["old"]);
        if let OutdatedVerdict::Ok { per_dep, .. } = v {
            assert_eq!(per_dep.get("old"), Some(&2));
        }
    }

    #[test]
    fn offenders_sorted() {
        let recipes = vec![("zeta", vec!["old"]), ("alpha", vec!["old"])];
        let v = audit(&recipes, &["old"]);
        if let OutdatedVerdict::Ok { offenders, .. } = v {
            assert_eq!(offenders, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let recipes = vec![("r", vec!["x"])];
        let r1 = audit(&recipes, &["x"]);
        let r2 = audit(&recipes, &["x"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn case_sensitive_dep_match() {
        let recipes = vec![("r", vec!["Old"])];
        let v = audit(&recipes, &["old"]);
        if let OutdatedVerdict::Ok { offenders, .. } = v {
            assert!(offenders.is_empty());
        }
    }

    #[test]
    fn duplicate_dep_in_recipe_double_counted() {
        let recipes = vec![("r", vec!["old", "old"])];
        let v = audit(&recipes, &["old"]);
        if let OutdatedVerdict::Ok { per_dep, .. } = v {
            assert_eq!(per_dep.get("old"), Some(&2));
        }
    }

    #[test]
    fn duplicate_offender_dedup() {
        let recipes = vec![("r", vec!["old", "old"])];
        let v = audit(&recipes, &["old"]);
        if let OutdatedVerdict::Ok { offenders, .. } = v {
            assert_eq!(offenders, vec!["r".to_string()]);
        }
    }

    #[test]
    fn many_recipes_handled() {
        let recipes: Vec<(&str, Vec<&str>)> = (0..20).map(|_| ("r", vec!["old"])).collect();
        let v = audit(&recipes, &["old"]);
        if let OutdatedVerdict::Ok { offenders, .. } = v {
            // After dedup all "r" → single entry.
            assert_eq!(offenders.len(), 1);
        }
    }

    #[test]
    fn empty_deprecated_no_offenders() {
        let recipes = vec![("r", vec!["serde"])];
        let v = audit(&recipes, &[]);
        if let OutdatedVerdict::Ok { offenders, .. } = v {
            assert!(offenders.is_empty());
        }
    }
}
