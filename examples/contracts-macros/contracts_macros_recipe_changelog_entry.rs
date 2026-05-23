//! # Contracts-Macros Recipe Changelog Entry
//!
//! Verify each modified recipe has a corresponding changelog entry.
//! Returns missing-entry list and entry-coverage rate.
//!
//! Demonstrates the **CMM.105** recipe for PMAT-192 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: keep-a-changelog.com convention; semver §10
//!  CHANGELOG hygiene.
//!
//! Run with: cargo run --example contracts_macros_recipe_changelog_entry
//!
//! Added by PMAT-192 (catalog 1351→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum ChangelogVerdict {
    Ok {
        missing: Vec<String>,
        coverage_rate: f64,
    },
    InvalidConfig,
}

pub fn audit(modified: &[&str], changelog_entries: &[&str]) -> ChangelogVerdict {
    if modified.is_empty() {
        return ChangelogVerdict::InvalidConfig;
    }
    let entries_set: BTreeSet<&str> = changelog_entries.iter().copied().collect();
    let mut missing: Vec<String> = modified
        .iter()
        .filter(|m| !entries_set.contains(*m))
        .map(|m| (*m).to_string())
        .collect();
    missing.sort();
    missing.dedup();
    let modified_set: BTreeSet<&str> = modified.iter().copied().collect();
    let covered = modified_set.len() - missing.len();
    let coverage_rate = covered as f64 / modified_set.len() as f64;
    ChangelogVerdict::Ok {
        missing,
        coverage_rate,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_changelog_entry")?;

    let modified = ["recipe_a", "recipe_b", "recipe_c"];
    let entries = ["recipe_a", "recipe_c"];
    println!("audit: {:?}", audit(&modified, &entries));
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
        if let ChangelogVerdict::Ok {
            missing,
            coverage_rate,
        } = v
        {
            assert!(missing.is_empty());
            assert!((coverage_rate - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn no_coverage() {
        let v = audit(&["a", "b"], &[]);
        if let ChangelogVerdict::Ok {
            missing,
            coverage_rate,
        } = v
        {
            assert_eq!(missing.len(), 2);
            assert_eq!(coverage_rate, 0.0);
        }
    }

    #[test]
    fn partial_coverage() {
        let v = audit(&["a", "b", "c"], &["a"]);
        if let ChangelogVerdict::Ok {
            missing,
            coverage_rate,
        } = v
        {
            assert_eq!(missing.len(), 2);
            assert!((coverage_rate - 1.0 / 3.0).abs() < 1e-9);
        }
    }

    #[test]
    fn empty_modified_rejected() {
        assert_eq!(audit(&[], &["a"]), ChangelogVerdict::InvalidConfig);
    }

    #[test]
    fn extra_entries_ignored() {
        let v = audit(&["a"], &["a", "b", "c"]);
        if let ChangelogVerdict::Ok { missing, .. } = v {
            assert!(missing.is_empty());
        }
    }

    #[test]
    fn missing_sorted() {
        let v = audit(&["zeta", "alpha"], &[]);
        if let ChangelogVerdict::Ok { missing, .. } = v {
            assert_eq!(missing, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn coverage_in_unit_range() {
        let v = audit(&["a"], &["a"]);
        if let ChangelogVerdict::Ok { coverage_rate, .. } = v {
            assert!((0.0..=1.0).contains(&coverage_rate));
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
        if let ChangelogVerdict::Ok { missing, .. } = v {
            assert_eq!(missing, vec!["Recipe_A".to_string()]);
        }
    }

    #[test]
    fn duplicate_modified_dedup() {
        let v = audit(&["a", "a", "b"], &["a"]);
        if let ChangelogVerdict::Ok { missing, .. } = v {
            assert_eq!(missing, vec!["b".to_string()]);
        }
    }

    #[test]
    fn many_recipes_handled() {
        let recipes: Vec<&str> = vec!["r"; 50];
        let entries = ["r"];
        let v = audit(&recipes, &entries);
        if let ChangelogVerdict::Ok { missing, .. } = v {
            assert!(missing.is_empty());
        }
    }
}
