//! # Contracts-Macros YAML Sequence Dedupe
//!
//! Detect duplicate scalar entries within YAML sequences. Returns
//! sorted duplicate values and the unique-entry count.
//!
//! Demonstrates the **CMM.199** recipe for PMAT-224 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: yamllint `unique` rule; jq `unique` filter semantics.
//!
//! Run with: cargo run --example contracts_macros_yaml_seq_dedupe
//!
//! Added by PMAT-224 (catalog 1639→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum DedupeVerdict {
    Ok {
        duplicates: Vec<String>,
        unique_count: u32,
    },
    InvalidConfig,
}

pub fn check(entries: &[&str]) -> DedupeVerdict {
    if entries.is_empty() {
        return DedupeVerdict::InvalidConfig;
    }
    let mut counts: BTreeMap<String, u32> = BTreeMap::new();
    for e in entries {
        *counts.entry((*e).to_string()).or_insert(0) += 1;
    }
    let mut duplicates: Vec<String> = counts
        .iter()
        .filter(|(_, c)| **c > 1)
        .map(|(k, _)| k.clone())
        .collect();
    duplicates.sort();
    DedupeVerdict::Ok {
        duplicates,
        unique_count: counts.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_seq_dedupe")?;

    println!("clean: {:?}", check(&["a", "b", "c"]));
    println!("dups: {:?}", check(&["a", "b", "a", "c", "b"]));
    println!("invalid: {:?}", check(&[]));
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
    fn no_duplicates_clean() {
        let v = check(&["a", "b", "c"]);
        if let DedupeVerdict::Ok { duplicates, .. } = v {
            assert!(duplicates.is_empty());
        }
    }

    #[test]
    fn duplicate_detected() {
        let v = check(&["a", "a"]);
        if let DedupeVerdict::Ok { duplicates, .. } = v {
            assert_eq!(duplicates, vec!["a".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(&[]), DedupeVerdict::InvalidConfig);
    }

    #[test]
    fn unique_count_correct() {
        let v = check(&["a", "b", "a", "c"]);
        if let DedupeVerdict::Ok { unique_count, .. } = v {
            assert_eq!(unique_count, 3);
        }
    }

    #[test]
    fn duplicates_sorted() {
        let v = check(&["zeta", "alpha", "zeta", "alpha"]);
        if let DedupeVerdict::Ok { duplicates, .. } = v {
            assert_eq!(duplicates, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check(&["a", "a"]);
        let r2 = check(&["a", "a"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn triple_occurrence_one_dup() {
        let v = check(&["a", "a", "a"]);
        if let DedupeVerdict::Ok { duplicates, .. } = v {
            assert_eq!(duplicates.len(), 1);
        }
    }

    #[test]
    fn case_sensitive() {
        let v = check(&["A", "a"]);
        if let DedupeVerdict::Ok { duplicates, .. } = v {
            assert!(duplicates.is_empty());
        }
    }

    #[test]
    fn unicode_entry_supported() {
        let v = check(&["café", "café"]);
        if let DedupeVerdict::Ok { duplicates, .. } = v {
            assert_eq!(duplicates, vec!["café".to_string()]);
        }
    }

    #[test]
    fn many_entries_handled() {
        let entries: Vec<&str> = (0..30).map(|_| "x").collect();
        let v = check(&entries);
        if let DedupeVerdict::Ok {
            unique_count,
            duplicates,
        } = v
        {
            assert_eq!(unique_count, 1);
            assert_eq!(duplicates, vec!["x".to_string()]);
        }
    }

    #[test]
    fn all_unique_no_duplicates() {
        let v = check(&["a", "b", "c", "d"]);
        if let DedupeVerdict::Ok { duplicates, .. } = v {
            assert!(duplicates.is_empty());
        }
    }
}
