//! # Contracts-Macros Obligation Diff
//!
//! Diff two snapshots of contract obligation IDs. Returns which were
//! added, removed, or unchanged. Used to track how a contract evolves
//! across releases.
//!
//! Demonstrates the **CMM.12** recipe for PMAT-161 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: semver-style API diff (cargo-public-api).
//!
//! Run with: cargo run --example contracts_macros_obligation_diff
//!
//! Added by PMAT-161 (catalog 1072→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum DiffVerdict {
    Ok {
        added: Vec<String>,
        removed: Vec<String>,
        unchanged: u32,
    },
    BothEmpty,
}

pub fn diff(prev: &[&str], current: &[&str]) -> DiffVerdict {
    if prev.is_empty() && current.is_empty() {
        return DiffVerdict::BothEmpty;
    }
    let prev_set: BTreeSet<&str> = prev.iter().copied().collect();
    let cur_set: BTreeSet<&str> = current.iter().copied().collect();
    let added: Vec<String> = cur_set
        .difference(&prev_set)
        .map(|s| (*s).to_string())
        .collect();
    let removed: Vec<String> = prev_set
        .difference(&cur_set)
        .map(|s| (*s).to_string())
        .collect();
    let unchanged = prev_set.intersection(&cur_set).count() as u32;
    DiffVerdict::Ok {
        added,
        removed,
        unchanged,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_diff")?;

    println!("added: {:?}", diff(&["a", "b"], &["a", "b", "c"]));
    println!("removed: {:?}", diff(&["a", "b", "c"], &["a", "b"]));
    println!("swap: {:?}", diff(&["a", "b", "c"], &["a", "x", "y"]));
    println!("identical: {:?}", diff(&["a", "b"], &["a", "b"]));
    println!("both empty: {:?}", diff(&[], &[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn differ_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn additions_detected() {
        let v = diff(&["a", "b"], &["a", "b", "c"]);
        if let DiffVerdict::Ok { added, .. } = v {
            assert_eq!(added, vec!["c".to_string()]);
        }
    }

    #[test]
    fn removals_detected() {
        let v = diff(&["a", "b", "c"], &["a", "b"]);
        if let DiffVerdict::Ok { removed, .. } = v {
            assert_eq!(removed, vec!["c".to_string()]);
        }
    }

    #[test]
    fn unchanged_count_correct() {
        let v = diff(&["a", "b", "c"], &["a", "b", "x"]);
        if let DiffVerdict::Ok { unchanged, .. } = v {
            assert_eq!(unchanged, 2);
        }
    }

    #[test]
    fn identical_no_changes() {
        let v = diff(&["a", "b"], &["a", "b"]);
        if let DiffVerdict::Ok {
            added,
            removed,
            unchanged,
        } = v
        {
            assert!(added.is_empty());
            assert!(removed.is_empty());
            assert_eq!(unchanged, 2);
        }
    }

    #[test]
    fn both_empty_special() {
        assert_eq!(diff(&[], &[]), DiffVerdict::BothEmpty);
    }

    #[test]
    fn empty_prev_all_added() {
        let v = diff(&[], &["a", "b"]);
        if let DiffVerdict::Ok {
            added,
            removed,
            unchanged,
        } = v
        {
            assert_eq!(added.len(), 2);
            assert!(removed.is_empty());
            assert_eq!(unchanged, 0);
        }
    }

    #[test]
    fn empty_current_all_removed() {
        let v = diff(&["a", "b"], &[]);
        if let DiffVerdict::Ok {
            added,
            removed,
            unchanged,
        } = v
        {
            assert!(added.is_empty());
            assert_eq!(removed.len(), 2);
            assert_eq!(unchanged, 0);
        }
    }

    #[test]
    fn full_swap_no_unchanged() {
        let v = diff(&["a", "b"], &["x", "y"]);
        if let DiffVerdict::Ok { unchanged, .. } = v {
            assert_eq!(unchanged, 0);
        }
    }

    #[test]
    fn duplicates_in_input_dedup() {
        let v = diff(&["a", "a", "b"], &["a", "b", "b"]);
        if let DiffVerdict::Ok {
            added,
            removed,
            unchanged,
        } = v
        {
            assert!(added.is_empty());
            assert!(removed.is_empty());
            assert_eq!(unchanged, 2);
        }
    }

    #[test]
    fn deterministic() {
        let a = diff(&["a", "b"], &["a", "c"]);
        let b = diff(&["a", "b"], &["a", "c"]);
        assert_eq!(a, b);
    }
}
