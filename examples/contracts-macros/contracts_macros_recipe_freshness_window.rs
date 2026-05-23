//! # Contracts-Macros Recipe Freshness Window
//!
//! Given recipes' last-updated day-offsets and a current day, flag
//! any whose age exceeds a freshness window. Reports stale list +
//! mean age.
//!
//! Demonstrates the **CMM.72** recipe for PMAT-181 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: TTL-based cache invalidation; SLA freshness windows.
//!
//! Run with: cargo run --example contracts_macros_recipe_freshness_window
//!
//! Added by PMAT-181 (catalog 1252→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FreshnessVerdict {
    Ok { stale: Vec<String>, mean_age: f64 },
    InvalidConfig,
}

pub fn audit(recipes: &[(&str, u32)], current_day: u32, window_days: u32) -> FreshnessVerdict {
    if recipes.is_empty() || window_days == 0 {
        return FreshnessVerdict::InvalidConfig;
    }
    let mut stale: Vec<String> = Vec::new();
    let mut total_age: u64 = 0;
    for (name, last_day) in recipes {
        if current_day < *last_day {
            return FreshnessVerdict::InvalidConfig;
        }
        let age = current_day - *last_day;
        total_age += u64::from(age);
        if age > window_days {
            stale.push((*name).to_string());
        }
    }
    stale.sort();
    let mean_age = total_age as f64 / recipes.len() as f64;
    FreshnessVerdict::Ok { stale, mean_age }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_freshness_window")?;

    let recipes = [("r1", 100), ("r2", 50), ("r3", 200)];
    println!("audit: {:?}", audit(&recipes, 250, 90));
    println!("invalid: {:?}", audit(&[], 250, 90));
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
    fn fresh_recipe_not_stale() {
        let recipes = [("r1", 90)];
        let v = audit(&recipes, 100, 30);
        if let FreshnessVerdict::Ok { stale, .. } = v {
            assert!(stale.is_empty());
        }
    }

    #[test]
    fn old_recipe_flagged_stale() {
        let recipes = [("r1", 0)];
        let v = audit(&recipes, 200, 30);
        if let FreshnessVerdict::Ok { stale, .. } = v {
            assert_eq!(stale, vec!["r1".to_string()]);
        }
    }

    #[test]
    fn empty_recipes_rejected() {
        assert_eq!(audit(&[], 100, 30), FreshnessVerdict::InvalidConfig);
    }

    #[test]
    fn zero_window_rejected() {
        let recipes = [("r1", 50)];
        assert_eq!(audit(&recipes, 100, 0), FreshnessVerdict::InvalidConfig);
    }

    #[test]
    fn future_last_update_rejected() {
        let recipes = [("r1", 200)];
        assert_eq!(audit(&recipes, 100, 30), FreshnessVerdict::InvalidConfig);
    }

    #[test]
    fn stale_list_sorted() {
        let recipes = [("zeta", 0), ("alpha", 0)];
        let v = audit(&recipes, 200, 30);
        if let FreshnessVerdict::Ok { stale, .. } = v {
            assert_eq!(stale, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn mean_age_correct() {
        let recipes = [("r1", 90), ("r2", 80)];
        let v = audit(&recipes, 100, 30);
        if let FreshnessVerdict::Ok { mean_age, .. } = v {
            assert!((mean_age - 15.0).abs() < 1e-9);
        }
    }

    #[test]
    fn boundary_age_equal_window_not_stale() {
        let recipes = [("r1", 70)];
        let v = audit(&recipes, 100, 30);
        if let FreshnessVerdict::Ok { stale, .. } = v {
            assert!(stale.is_empty());
        }
    }

    #[test]
    fn boundary_one_over_window_stale() {
        let recipes = [("r1", 69)];
        let v = audit(&recipes, 100, 30);
        if let FreshnessVerdict::Ok { stale, .. } = v {
            assert_eq!(stale, vec!["r1".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let recipes = [("r1", 50)];
        let r1 = audit(&recipes, 100, 30);
        let r2 = audit(&recipes, 100, 30);
        assert_eq!(r1, r2);
    }

    #[test]
    fn mixed_fresh_and_stale() {
        let recipes = [("fresh", 95), ("stale", 0)];
        let v = audit(&recipes, 100, 30);
        if let FreshnessVerdict::Ok { stale, .. } = v {
            assert_eq!(stale, vec!["stale".to_string()]);
        }
    }
}
