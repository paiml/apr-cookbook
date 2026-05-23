//! # Contracts-Macros Recipe Runtime Budget
//!
//! Verify each recipe declares a max-runtime budget within bounds
//! `[min_ms, max_ms]`. Returns offenders (no budget / out-of-range).
//!
//! Demonstrates the **CMM.93** recipe for PMAT-188 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SLO/SLA budget conventions; Google SRE workbook ch.4.
//!
//! Run with: cargo run --example contracts_macros_recipe_runtime_budget
//!
//! Added by PMAT-188 (catalog 1315→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone)]
pub enum BudgetIssue {
    NoBudget,
    BelowMin,
    AboveMax,
}

#[derive(Debug, PartialEq)]
pub enum BudgetVerdict {
    Ok {
        per_recipe: Vec<(String, Option<BudgetIssue>)>,
        ok_count: u32,
    },
    InvalidConfig,
}

pub fn audit(recipes: &[(&str, Option<u32>)], min_ms: u32, max_ms: u32) -> BudgetVerdict {
    if recipes.is_empty() || min_ms >= max_ms {
        return BudgetVerdict::InvalidConfig;
    }
    let mut per_recipe: Vec<(String, Option<BudgetIssue>)> = Vec::with_capacity(recipes.len());
    let mut ok_count = 0u32;
    for (name, budget) in recipes {
        let issue = match budget {
            None => Some(BudgetIssue::NoBudget),
            Some(b) if *b < min_ms => Some(BudgetIssue::BelowMin),
            Some(b) if *b > max_ms => Some(BudgetIssue::AboveMax),
            Some(_) => None,
        };
        if issue.is_none() {
            ok_count += 1;
        }
        per_recipe.push(((*name).to_string(), issue));
    }
    BudgetVerdict::Ok {
        per_recipe,
        ok_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_runtime_budget")?;

    let recipes = [
        ("ok", Some(50)),
        ("missing", None),
        ("too_fast", Some(1)),
        ("too_slow", Some(100_000)),
    ];
    println!("audit: {:?}", audit(&recipes, 10, 1000));
    println!("invalid: {:?}", audit(&[], 10, 1000));
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
    fn within_range_ok() {
        let recipes = [("r", Some(50))];
        let v = audit(&recipes, 10, 100);
        if let BudgetVerdict::Ok { ok_count, .. } = v {
            assert_eq!(ok_count, 1);
        }
    }

    #[test]
    fn no_budget_flagged() {
        let recipes = [("r", None)];
        let v = audit(&recipes, 10, 100);
        if let BudgetVerdict::Ok { per_recipe, .. } = v {
            assert_eq!(per_recipe[0].1, Some(BudgetIssue::NoBudget));
        }
    }

    #[test]
    fn below_min_flagged() {
        let recipes = [("r", Some(5))];
        let v = audit(&recipes, 10, 100);
        if let BudgetVerdict::Ok { per_recipe, .. } = v {
            assert_eq!(per_recipe[0].1, Some(BudgetIssue::BelowMin));
        }
    }

    #[test]
    fn above_max_flagged() {
        let recipes = [("r", Some(200))];
        let v = audit(&recipes, 10, 100);
        if let BudgetVerdict::Ok { per_recipe, .. } = v {
            assert_eq!(per_recipe[0].1, Some(BudgetIssue::AboveMax));
        }
    }

    #[test]
    fn empty_recipes_rejected() {
        assert_eq!(audit(&[], 10, 100), BudgetVerdict::InvalidConfig);
    }

    #[test]
    fn min_geq_max_rejected() {
        let recipes = [("r", Some(50))];
        assert_eq!(audit(&recipes, 100, 50), BudgetVerdict::InvalidConfig);
    }

    #[test]
    fn boundary_min_ok() {
        let recipes = [("r", Some(10))];
        let v = audit(&recipes, 10, 100);
        if let BudgetVerdict::Ok { per_recipe, .. } = v {
            assert!(per_recipe[0].1.is_none());
        }
    }

    #[test]
    fn boundary_max_ok() {
        let recipes = [("r", Some(100))];
        let v = audit(&recipes, 10, 100);
        if let BudgetVerdict::Ok { per_recipe, .. } = v {
            assert!(per_recipe[0].1.is_none());
        }
    }

    #[test]
    fn deterministic() {
        let recipes = [("r", Some(50))];
        let r1 = audit(&recipes, 10, 100);
        let r2 = audit(&recipes, 10, 100);
        assert_eq!(r1, r2);
    }

    #[test]
    fn ok_count_correct() {
        let recipes = [("a", Some(50)), ("b", None), ("c", Some(80))];
        let v = audit(&recipes, 10, 100);
        if let BudgetVerdict::Ok { ok_count, .. } = v {
            assert_eq!(ok_count, 2);
        }
    }

    #[test]
    fn order_preserved() {
        let recipes = [("first", Some(50)), ("second", None)];
        let v = audit(&recipes, 10, 100);
        if let BudgetVerdict::Ok { per_recipe, .. } = v {
            assert_eq!(per_recipe[0].0, "first");
            assert_eq!(per_recipe[1].0, "second");
        }
    }
}
