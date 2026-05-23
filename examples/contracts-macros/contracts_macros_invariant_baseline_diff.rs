//! # Contracts-Macros Invariant Baseline Diff
//!
//! Compare current invariant set vs a baseline; flag any added,
//! removed, or modified items. Returns sorted lists per category.
//!
//! Demonstrates the **CMM.174** recipe for PMAT-215 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: cargo lockfile diff conventions; semver-checks API
//!  surface diff.
//!
//! Run with: cargo run --example contracts_macros_invariant_baseline_diff
//!
//! Added by PMAT-215 (catalog 1558→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum DiffVerdict {
    Ok {
        added: Vec<String>,
        removed: Vec<String>,
        modified: Vec<String>,
    },
    InvalidConfig,
}

/// Items: (id, hash). Compare baseline vs current.
pub fn diff(baseline: &[(&str, u64)], current: &[(&str, u64)]) -> DiffVerdict {
    if baseline.is_empty() && current.is_empty() {
        return DiffVerdict::InvalidConfig;
    }
    let base_map: BTreeMap<&str, u64> = baseline.iter().copied().collect();
    let cur_map: BTreeMap<&str, u64> = current.iter().copied().collect();
    let mut added: Vec<String> = Vec::new();
    let mut removed: Vec<String> = Vec::new();
    let mut modified: Vec<String> = Vec::new();
    for (id, h) in &cur_map {
        match base_map.get(id) {
            None => added.push((*id).to_string()),
            Some(bh) if bh != h => modified.push((*id).to_string()),
            _ => {}
        }
    }
    for id in base_map.keys() {
        if !cur_map.contains_key(id) {
            removed.push((*id).to_string());
        }
    }
    added.sort();
    removed.sort();
    modified.sort();
    DiffVerdict::Ok {
        added,
        removed,
        modified,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_invariant_baseline_diff")?;

    let baseline = [("inv_a", 100u64), ("inv_b", 200)];
    let current = [("inv_a", 100u64), ("inv_b", 250), ("inv_c", 300)];
    println!("diff: {:?}", diff(&baseline, &current));
    println!("invalid: {:?}", diff(&[], &[]));
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
    fn no_change_no_diff() {
        let b = [("a", 1u64), ("b", 2)];
        let c = [("a", 1u64), ("b", 2)];
        let v = diff(&b, &c);
        if let DiffVerdict::Ok {
            added,
            removed,
            modified,
        } = v
        {
            assert!(added.is_empty());
            assert!(removed.is_empty());
            assert!(modified.is_empty());
        }
    }

    #[test]
    fn added_detected() {
        let b = [("a", 1u64)];
        let c = [("a", 1u64), ("b", 2)];
        let v = diff(&b, &c);
        if let DiffVerdict::Ok { added, .. } = v {
            assert_eq!(added, vec!["b".to_string()]);
        }
    }

    #[test]
    fn removed_detected() {
        let b = [("a", 1u64), ("b", 2)];
        let c = [("a", 1u64)];
        let v = diff(&b, &c);
        if let DiffVerdict::Ok { removed, .. } = v {
            assert_eq!(removed, vec!["b".to_string()]);
        }
    }

    #[test]
    fn modified_detected() {
        let b = [("a", 1u64)];
        let c = [("a", 99u64)];
        let v = diff(&b, &c);
        if let DiffVerdict::Ok { modified, .. } = v {
            assert_eq!(modified, vec!["a".to_string()]);
        }
    }

    #[test]
    fn empty_both_rejected() {
        assert_eq!(diff(&[], &[]), DiffVerdict::InvalidConfig);
    }

    #[test]
    fn empty_baseline_all_added() {
        let c = [("a", 1u64), ("b", 2)];
        let v = diff(&[], &c);
        if let DiffVerdict::Ok { added, .. } = v {
            assert_eq!(added.len(), 2);
        }
    }

    #[test]
    fn empty_current_all_removed() {
        let b = [("a", 1u64), ("b", 2)];
        let v = diff(&b, &[]);
        if let DiffVerdict::Ok { removed, .. } = v {
            assert_eq!(removed.len(), 2);
        }
    }

    #[test]
    fn deterministic() {
        let b = [("a", 1u64)];
        let c = [("a", 1u64)];
        let r1 = diff(&b, &c);
        let r2 = diff(&b, &c);
        assert_eq!(r1, r2);
    }

    #[test]
    fn lists_sorted() {
        let b = [("zeta", 1u64), ("alpha", 2)];
        let c: [(&str, u64); 0] = [];
        let v = diff(&b, &c);
        if let DiffVerdict::Ok { removed, .. } = v {
            assert_eq!(removed, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn many_changes_handled() {
        let b: Vec<(&str, u64)> = (0..30).map(|_| ("x", 1u64)).collect();
        let c: Vec<(&str, u64)> = (0..30).map(|_| ("y", 1u64)).collect();
        let v = diff(&b, &c);
        if let DiffVerdict::Ok { added, removed, .. } = v {
            // BTreeMap dedupes by key — both reduce to single entry.
            assert_eq!(added.len(), 1);
            assert_eq!(removed.len(), 1);
        }
    }

    #[test]
    fn unicode_id_supported() {
        let b = [("café", 1u64)];
        let c = [("café", 99u64)];
        let v = diff(&b, &c);
        if let DiffVerdict::Ok { modified, .. } = v {
            assert_eq!(modified, vec!["café".to_string()]);
        }
    }

    #[test]
    fn mixed_changes_handled() {
        let b = [("a", 1u64), ("b", 2)];
        let c = [("a", 99u64), ("c", 3)];
        let v = diff(&b, &c);
        if let DiffVerdict::Ok {
            added,
            removed,
            modified,
        } = v
        {
            assert_eq!(added, vec!["c".to_string()]);
            assert_eq!(removed, vec!["b".to_string()]);
            assert_eq!(modified, vec!["a".to_string()]);
        }
    }
}
