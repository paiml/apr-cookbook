//! # Contracts-Macros Recipe Runtime Min
//!
//! Validate recipes' measured runtime ms is below a min-budget
//! threshold (faster is better). Returns sorted IDs that exceeded
//! the budget.
//!
//! Demonstrates the **CMM.203** recipe for PMAT-225 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: cargo bench warm-up time bounds; criterion.rs
//!  performance regression detection.
//!
//! Run with: cargo run --example contracts_macros_recipe_runtime_min
//!
//! Added by PMAT-225 (catalog 1648→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RuntimeVerdict {
    Ok {
        slow_ids: Vec<String>,
        fast_count: u32,
    },
    InvalidConfig,
}

pub fn check(items: &[(&str, u32)], budget_ms: u32) -> RuntimeVerdict {
    if items.is_empty() || budget_ms == 0 {
        return RuntimeVerdict::InvalidConfig;
    }
    let mut slow: Vec<String> = items
        .iter()
        .filter(|(_, ms)| *ms > budget_ms)
        .map(|(id, _)| (*id).to_string())
        .collect();
    slow.sort();
    let fast = (items.len() as u32) - (slow.len() as u32);
    RuntimeVerdict::Ok {
        slow_ids: slow,
        fast_count: fast,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_runtime_min")?;

    let items = [("r1", 50), ("r2", 200), ("r3", 100)];
    println!("budget-100: {:?}", check(&items, 100));
    println!("invalid: {:?}", check(&[], 100));
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
    fn within_budget_no_offender() {
        let v = check(&[("r", 50)], 100);
        if let RuntimeVerdict::Ok { slow_ids, .. } = v {
            assert!(slow_ids.is_empty());
        }
    }

    #[test]
    fn over_budget_flagged() {
        let v = check(&[("r", 200)], 100);
        if let RuntimeVerdict::Ok { slow_ids, .. } = v {
            assert_eq!(slow_ids, vec!["r".to_string()]);
        }
    }

    #[test]
    fn at_budget_no_offender() {
        let v = check(&[("r", 100)], 100);
        if let RuntimeVerdict::Ok { slow_ids, .. } = v {
            assert!(slow_ids.is_empty());
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(&[], 100), RuntimeVerdict::InvalidConfig);
    }

    #[test]
    fn zero_budget_rejected() {
        assert_eq!(check(&[("r", 50)], 0), RuntimeVerdict::InvalidConfig);
    }

    #[test]
    fn fast_count_correct() {
        let v = check(&[("a", 50), ("b", 200), ("c", 100)], 100);
        if let RuntimeVerdict::Ok { fast_count, .. } = v {
            assert_eq!(fast_count, 2);
        }
    }

    #[test]
    fn slow_sorted() {
        let v = check(&[("zeta", 200), ("alpha", 200)], 100);
        if let RuntimeVerdict::Ok { slow_ids, .. } = v {
            assert_eq!(slow_ids, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check(&[("r", 50)], 100);
        let r2 = check(&[("r", 50)], 100);
        assert_eq!(r1, r2);
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<(&str, u32)> = (0..30).map(|_| ("r", 200)).collect();
        let v = check(&items, 100);
        if let RuntimeVerdict::Ok { slow_ids, .. } = v {
            assert_eq!(slow_ids.len(), 30);
        }
    }

    #[test]
    fn no_offenders_returns_empty() {
        let v = check(&[("a", 5), ("b", 10)], 100);
        if let RuntimeVerdict::Ok { slow_ids, .. } = v {
            assert!(slow_ids.is_empty());
        }
    }

    #[test]
    fn unicode_id_supported() {
        let v = check(&[("café", 200)], 100);
        if let RuntimeVerdict::Ok { slow_ids, .. } = v {
            assert_eq!(slow_ids, vec!["café".to_string()]);
        }
    }

    #[test]
    fn high_budget_handled() {
        let v = check(&[("r", 1_000_000)], u32::MAX);
        if let RuntimeVerdict::Ok { slow_ids, .. } = v {
            assert!(slow_ids.is_empty());
        }
    }
}
