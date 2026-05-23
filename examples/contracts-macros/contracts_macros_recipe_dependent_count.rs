//! # Contracts-Macros Recipe Dependent Count
//!
//! Count how many other recipes depend on each recipe (reverse-dep
//! graph). Returns sorted-by-dependents (descending) ranking and
//! orphans (zero dependents).
//!
//! Demonstrates the **CMM.194** recipe for PMAT-222 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: cargo reverse-dep tree; npm `dependents` graph queries.
//!
//! Run with: cargo run --example contracts_macros_recipe_dependent_count
//!
//! Added by PMAT-222 (catalog 1621→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum DependentVerdict {
    Ok {
        ranked: Vec<(String, u32)>,
        orphans: Vec<String>,
    },
    InvalidConfig,
}

/// `recipes`: list of all recipe IDs.
/// `deps`: list of (dependent, dependency) edges.
pub fn count(recipes: &[&str], deps: &[(&str, &str)]) -> DependentVerdict {
    if recipes.is_empty() {
        return DependentVerdict::InvalidConfig;
    }
    let mut counts: BTreeMap<String, u32> = recipes.iter().map(|r| ((*r).to_string(), 0)).collect();
    for (_, dep) in deps {
        if let Some(c) = counts.get_mut(*dep) {
            *c += 1;
        }
    }
    let orphans: Vec<String> = counts
        .iter()
        .filter(|(_, c)| **c == 0)
        .map(|(k, _)| k.clone())
        .collect();
    let mut ranked: Vec<(String, u32)> = counts.into_iter().collect();
    // Sort by count desc, name asc.
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    DependentVerdict::Ok { ranked, orphans }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_dependent_count")?;

    let recipes = ["a", "b", "c"];
    let deps = [("a", "b"), ("c", "b")];
    println!("count: {:?}", count(&recipes, &deps));
    println!("invalid: {:?}", count(&[], &[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn dependents_counted() {
        let recipes = ["a", "b"];
        let deps = [("a", "b")];
        let v = count(&recipes, &deps);
        if let DependentVerdict::Ok { ranked, .. } = v {
            // "b" has 1 dependent ("a"); "a" has 0.
            assert_eq!(ranked[0], ("b".to_string(), 1));
        }
    }

    #[test]
    fn orphan_detected() {
        let recipes = ["a", "b", "orphan"];
        let deps = [("a", "b")];
        let v = count(&recipes, &deps);
        if let DependentVerdict::Ok { orphans, .. } = v {
            assert!(orphans.contains(&"orphan".to_string()));
        }
    }

    #[test]
    fn empty_recipes_rejected() {
        assert_eq!(count(&[], &[]), DependentVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let recipes = ["a"];
        let r1 = count(&recipes, &[]);
        let r2 = count(&recipes, &[]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn ranked_sorted_descending() {
        let recipes = ["a", "b", "c"];
        let deps = [("a", "b"), ("c", "b"), ("a", "c")];
        let v = count(&recipes, &deps);
        if let DependentVerdict::Ok { ranked, .. } = v {
            // "b" has 2 deps, "c" has 1, "a" has 0.
            assert_eq!(ranked[0].1, 2);
            assert_eq!(ranked[1].1, 1);
            assert_eq!(ranked[2].1, 0);
        }
    }

    #[test]
    fn unknown_dep_ignored() {
        let recipes = ["a"];
        let deps = [("a", "missing")];
        let v = count(&recipes, &deps);
        if let DependentVerdict::Ok { orphans, .. } = v {
            // "a" has 0 dependents (unknown "missing" silently ignored).
            assert!(orphans.contains(&"a".to_string()));
        }
    }

    #[test]
    fn no_deps_all_orphans() {
        let recipes = ["a", "b", "c"];
        let v = count(&recipes, &[]);
        if let DependentVerdict::Ok { orphans, .. } = v {
            assert_eq!(orphans.len(), 3);
        }
    }

    #[test]
    fn many_recipes_handled() {
        let recipes: Vec<&str> = (0..30).map(|_| "r").collect();
        let v = count(&recipes, &[]);
        // BTreeMap dedupes by name, so just one orphan.
        if let DependentVerdict::Ok { orphans, .. } = v {
            assert_eq!(orphans.len(), 1);
        }
    }

    #[test]
    fn unicode_recipe_supported() {
        let recipes = ["café", "résumé"];
        let deps = [("café", "résumé")];
        let v = count(&recipes, &deps);
        if let DependentVerdict::Ok { ranked, .. } = v {
            assert_eq!(ranked[0], ("résumé".to_string(), 1));
        }
    }

    #[test]
    fn alphabetical_tie_break() {
        let recipes = ["zeta", "alpha"];
        let v = count(&recipes, &[]);
        if let DependentVerdict::Ok { ranked, .. } = v {
            // Both 0 deps; alphabetical order: alpha first.
            assert_eq!(ranked[0].0, "alpha");
            assert_eq!(ranked[1].0, "zeta");
        }
    }

    #[test]
    fn self_dep_handled() {
        let recipes = ["a"];
        let deps = [("a", "a")];
        let v = count(&recipes, &deps);
        if let DependentVerdict::Ok { ranked, .. } = v {
            assert_eq!(ranked[0], ("a".to_string(), 1));
        }
    }

    #[test]
    fn ranked_count_matches_recipes() {
        let recipes = ["a", "b", "c"];
        let v = count(&recipes, &[]);
        if let DependentVerdict::Ok { ranked, .. } = v {
            assert_eq!(ranked.len(), 3);
        }
    }
}
