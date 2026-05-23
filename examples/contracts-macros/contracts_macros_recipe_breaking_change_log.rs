//! # Contracts-Macros Recipe Breaking Change Log
//!
//! Verify recipes flagged as breaking changes have a corresponding
//! `BREAKING:` entry in the changelog. Returns missing-entry list.
//!
//! Demonstrates the **CMM.114** recipe for PMAT-195 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: keep-a-changelog.com BREAKING markers; semver §8 major
//!  bump rationale.
//!
//! Run with: cargo run --example contracts_macros_recipe_breaking_change_log
//!
//! Added by PMAT-195 (catalog 1378→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum BreakingVerdict {
    Ok {
        missing_entries: Vec<String>,
        coverage_rate: f64,
    },
    InvalidConfig,
}

pub fn audit(breaking_recipes: &[&str], breaking_log_entries: &[&str]) -> BreakingVerdict {
    if breaking_recipes.is_empty() {
        return BreakingVerdict::InvalidConfig;
    }
    let log_set: BTreeSet<&str> = breaking_log_entries.iter().copied().collect();
    let mut missing: Vec<String> = breaking_recipes
        .iter()
        .filter(|r| !log_set.contains(*r))
        .map(|r| (*r).to_string())
        .collect();
    missing.sort();
    missing.dedup();
    let breaking_set: BTreeSet<&str> = breaking_recipes.iter().copied().collect();
    let covered = breaking_set.len() - missing.len();
    let coverage_rate = covered as f64 / breaking_set.len() as f64;
    BreakingVerdict::Ok {
        missing_entries: missing,
        coverage_rate,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_breaking_change_log")?;

    let breaking = ["recipe_a", "recipe_b"];
    let log = ["recipe_a"];
    println!("audit: {:?}", audit(&breaking, &log));
    println!("invalid: {:?}", audit(&[], &[]));
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
    fn full_coverage() {
        let v = audit(&["a", "b"], &["a", "b"]);
        if let BreakingVerdict::Ok {
            missing_entries,
            coverage_rate,
        } = v
        {
            assert!(missing_entries.is_empty());
            assert_eq!(coverage_rate, 1.0);
        }
    }

    #[test]
    fn no_log_entries_all_missing() {
        let v = audit(&["a", "b"], &[]);
        if let BreakingVerdict::Ok {
            missing_entries,
            coverage_rate,
        } = v
        {
            assert_eq!(missing_entries.len(), 2);
            assert_eq!(coverage_rate, 0.0);
        }
    }

    #[test]
    fn empty_breaking_rejected() {
        assert_eq!(audit(&[], &["a"]), BreakingVerdict::InvalidConfig);
    }

    #[test]
    fn partial_coverage() {
        let v = audit(&["a", "b", "c"], &["a"]);
        if let BreakingVerdict::Ok {
            missing_entries,
            coverage_rate,
        } = v
        {
            assert_eq!(missing_entries.len(), 2);
            assert!((coverage_rate - 1.0 / 3.0).abs() < 1e-9);
        }
    }

    #[test]
    fn missing_sorted() {
        let v = audit(&["zeta", "alpha"], &[]);
        if let BreakingVerdict::Ok {
            missing_entries, ..
        } = v
        {
            assert_eq!(
                missing_entries,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn extra_log_entries_ignored() {
        let v = audit(&["a"], &["a", "b", "c"]);
        if let BreakingVerdict::Ok {
            missing_entries, ..
        } = v
        {
            assert!(missing_entries.is_empty());
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&["a"], &["a"]);
        let r2 = audit(&["a"], &["a"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn case_sensitive() {
        let v = audit(&["Recipe_A"], &["recipe_a"]);
        if let BreakingVerdict::Ok {
            missing_entries, ..
        } = v
        {
            assert_eq!(missing_entries, vec!["Recipe_A".to_string()]);
        }
    }

    #[test]
    fn rate_in_unit_range() {
        let v = audit(&["a"], &["a"]);
        if let BreakingVerdict::Ok { coverage_rate, .. } = v {
            assert!((0.0..=1.0).contains(&coverage_rate));
        }
    }

    #[test]
    fn duplicate_breaking_dedup() {
        let v = audit(&["a", "a", "b"], &["a"]);
        if let BreakingVerdict::Ok {
            missing_entries, ..
        } = v
        {
            assert_eq!(missing_entries, vec!["b".to_string()]);
        }
    }

    #[test]
    fn many_recipes_handled() {
        let recipes: Vec<&str> = vec!["x"; 50];
        let log = ["x"];
        let v = audit(&recipes, &log);
        if let BreakingVerdict::Ok {
            missing_entries, ..
        } = v
        {
            assert!(missing_entries.is_empty());
        }
    }
}
