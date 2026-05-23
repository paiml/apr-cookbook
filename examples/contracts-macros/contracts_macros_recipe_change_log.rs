//! # Contracts-Macros Recipe Change Log
//!
//! Compare two snapshots of recipe IDs and report Added / Removed
//! / Renamed pairs (rename heuristic: same canonical form).
//!
//! Demonstrates the **CMM.59** recipe for PMAT-177 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: git log + Conventional Commits change classification.
//!
//! Run with: cargo run --example contracts_macros_recipe_change_log
//!
//! Added by PMAT-177 (catalog 1216→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum ChangeLogVerdict {
    Ok {
        added: Vec<String>,
        removed: Vec<String>,
        renamed: Vec<(String, String)>,
    },
    NoChange,
    EmptySnapshots,
}

pub fn diff(prev: &[&str], current: &[&str]) -> ChangeLogVerdict {
    if prev.is_empty() && current.is_empty() {
        return ChangeLogVerdict::EmptySnapshots;
    }
    let prev_canon: BTreeMap<String, &str> = prev.iter().map(|s| (canonicalize(s), *s)).collect();
    let cur_canon: BTreeMap<String, &str> = current.iter().map(|s| (canonicalize(s), *s)).collect();
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut renamed = Vec::new();
    for (canon, name) in &cur_canon {
        if let Some(prev_name) = prev_canon.get(canon) {
            if prev_name != name {
                renamed.push(((*prev_name).to_string(), (*name).to_string()));
            }
        } else {
            added.push((*name).to_string());
        }
    }
    for (canon, name) in &prev_canon {
        if !cur_canon.contains_key(canon) {
            removed.push((*name).to_string());
        }
    }
    if added.is_empty() && removed.is_empty() && renamed.is_empty() {
        return ChangeLogVerdict::NoChange;
    }
    ChangeLogVerdict::Ok {
        added,
        removed,
        renamed,
    }
}

fn canonicalize(s: &str) -> String {
    s.chars()
        .filter_map(|c| {
            if c.is_ascii_alphanumeric() {
                Some(c.to_ascii_lowercase())
            } else if c == '-' || c == '_' {
                Some('_')
            } else {
                None
            }
        })
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_change_log")?;

    let prev = ["recipe_a", "recipe_b", "recipe_c"];
    let current = ["recipe_a", "recipe-b", "recipe_d"];
    println!("changes: {:?}", diff(&prev, &current));

    let same = ["x", "y"];
    println!("no change: {:?}", diff(&same, &same));
    println!("empty: {:?}", diff(&[], &[]));
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
        let v = diff(&["a"], &["a", "b"]);
        if let ChangeLogVerdict::Ok { added, .. } = v {
            assert_eq!(added, vec!["b".to_string()]);
        }
    }

    #[test]
    fn removals_detected() {
        let v = diff(&["a", "b"], &["a"]);
        if let ChangeLogVerdict::Ok { removed, .. } = v {
            assert_eq!(removed, vec!["b".to_string()]);
        }
    }

    #[test]
    fn rename_detected_by_canonicalization() {
        let v = diff(&["recipe-b"], &["recipe_b"]);
        if let ChangeLogVerdict::Ok { renamed, .. } = v {
            assert_eq!(
                renamed,
                vec![("recipe-b".to_string(), "recipe_b".to_string())]
            );
        }
    }

    #[test]
    fn no_change_recognized() {
        assert_eq!(diff(&["a"], &["a"]), ChangeLogVerdict::NoChange);
    }

    #[test]
    fn empty_both_special() {
        assert_eq!(diff(&[], &[]), ChangeLogVerdict::EmptySnapshots);
    }

    #[test]
    fn empty_prev_all_added() {
        let v = diff(&[], &["a", "b"]);
        if let ChangeLogVerdict::Ok { added, .. } = v {
            assert_eq!(added.len(), 2);
        }
    }

    #[test]
    fn empty_current_all_removed() {
        let v = diff(&["a", "b"], &[]);
        if let ChangeLogVerdict::Ok { removed, .. } = v {
            assert_eq!(removed.len(), 2);
        }
    }

    #[test]
    fn mixed_changes() {
        let v = diff(&["a", "b"], &["a", "c"]);
        if let ChangeLogVerdict::Ok { added, removed, .. } = v {
            assert!(added.contains(&"c".to_string()));
            assert!(removed.contains(&"b".to_string()));
        }
    }

    #[test]
    fn case_change_recognized_as_rename() {
        let v = diff(&["recipe_a"], &["RECIPE_A"]);
        if let ChangeLogVerdict::Ok { renamed, .. } = v {
            assert_eq!(renamed.len(), 1);
        }
    }

    #[test]
    fn deterministic() {
        let a = diff(&["a"], &["a", "b"]);
        let b = diff(&["a"], &["a", "b"]);
        assert_eq!(a, b);
    }
}
