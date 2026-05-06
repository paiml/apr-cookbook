//! # Contracts-Macros Recipe Test Count Audit
//!
//! Verify each recipe declares the minimum required test count.
//! Returns under-tested recipes plus per-recipe count.
//!
//! Demonstrates the **CMM.84** recipe for PMAT-185 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Beck, TDD by Example (2002); test-count discipline.
//!
//! Run with: cargo run --example contracts_macros_recipe_test_count_audit
//!
//! Added by PMAT-185 (catalog 1288→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum CountAuditVerdict {
    Ok {
        per_recipe: BTreeMap<String, u32>,
        under_tested: Vec<String>,
        median: u32,
    },
    InvalidConfig,
}

pub fn audit(recipes: &[(&str, u32)], min_tests: u32) -> CountAuditVerdict {
    if recipes.is_empty() || min_tests == 0 {
        return CountAuditVerdict::InvalidConfig;
    }
    let mut per_recipe: BTreeMap<String, u32> = BTreeMap::new();
    let mut counts: Vec<u32> = Vec::with_capacity(recipes.len());
    for (name, count) in recipes {
        per_recipe.insert((*name).to_string(), *count);
        counts.push(*count);
    }
    counts.sort_unstable();
    let median = counts[counts.len() / 2];
    let mut under_tested: Vec<String> = per_recipe
        .iter()
        .filter(|(_, &c)| c < min_tests)
        .map(|(name, _)| name.clone())
        .collect();
    under_tested.sort();
    CountAuditVerdict::Ok {
        per_recipe,
        under_tested,
        median,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_test_count_audit")?;

    let recipes = [("r1", 12), ("r2", 5), ("r3", 15)];
    println!("audit: {:?}", audit(&recipes, 10));
    println!("invalid: {:?}", audit(&[], 10));
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
    fn well_tested_no_offenders() {
        let recipes = [("r1", 12), ("r2", 11)];
        let v = audit(&recipes, 10);
        if let CountAuditVerdict::Ok { under_tested, .. } = v {
            assert!(under_tested.is_empty());
        }
    }

    #[test]
    fn under_tested_flagged() {
        let recipes = [("r1", 5), ("r2", 12)];
        let v = audit(&recipes, 10);
        if let CountAuditVerdict::Ok { under_tested, .. } = v {
            assert_eq!(under_tested, vec!["r1".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[], 10), CountAuditVerdict::InvalidConfig);
    }

    #[test]
    fn zero_min_rejected() {
        let recipes = [("r1", 5)];
        assert_eq!(audit(&recipes, 0), CountAuditVerdict::InvalidConfig);
    }

    #[test]
    fn boundary_min_passes() {
        let recipes = [("r1", 10)];
        let v = audit(&recipes, 10);
        if let CountAuditVerdict::Ok { under_tested, .. } = v {
            assert!(under_tested.is_empty());
        }
    }

    #[test]
    fn one_below_min_flagged() {
        let recipes = [("r1", 9)];
        let v = audit(&recipes, 10);
        if let CountAuditVerdict::Ok { under_tested, .. } = v {
            assert_eq!(under_tested, vec!["r1".to_string()]);
        }
    }

    #[test]
    fn median_correct() {
        let recipes = [("a", 5), ("b", 10), ("c", 15)];
        let v = audit(&recipes, 1);
        if let CountAuditVerdict::Ok { median, .. } = v {
            assert_eq!(median, 10);
        }
    }

    #[test]
    fn deterministic() {
        let recipes = [("r1", 10)];
        let r1 = audit(&recipes, 5);
        let r2 = audit(&recipes, 5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn under_tested_sorted() {
        let recipes = [("zeta", 1), ("alpha", 1)];
        let v = audit(&recipes, 10);
        if let CountAuditVerdict::Ok { under_tested, .. } = v {
            assert_eq!(under_tested, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn per_recipe_correct() {
        let recipes = [("r1", 12)];
        let v = audit(&recipes, 10);
        if let CountAuditVerdict::Ok { per_recipe, .. } = v {
            assert_eq!(per_recipe.get("r1"), Some(&12));
        }
    }

    #[test]
    fn many_recipes_handled() {
        let recipes: Vec<(&str, u32)> = (0..20).map(|_| ("r", 12)).collect();
        let v = audit(&recipes, 10);
        if let CountAuditVerdict::Ok { under_tested, .. } = v {
            assert!(under_tested.is_empty());
        }
    }
}
