//! # Contracts-Macros YAML Alias Duplicate Check
//!
//! Detect YAML alias names declared more than once across all anchors.
//! Returns the sorted list of duplicate names and the duplicate count.
//!
//! Demonstrates the **CMM.137** recipe for PMAT-203 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 spec §6.9 (aliases & anchors); libyaml dup-detect
//!  semantics.
//!
//! Run with: cargo run --example contracts_macros_yaml_alias_dup_check
//!
//! Added by PMAT-203 (catalog 1450→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum AliasDupVerdict {
    Ok {
        duplicates: Vec<String>,
        unique_count: u32,
    },
    InvalidConfig,
}

pub fn check(aliases: &[&str]) -> AliasDupVerdict {
    if aliases.is_empty() {
        return AliasDupVerdict::InvalidConfig;
    }
    let mut counts: BTreeMap<String, u32> = BTreeMap::new();
    for a in aliases {
        *counts.entry((*a).to_string()).or_insert(0) += 1;
    }
    let mut dups: Vec<String> = counts
        .iter()
        .filter(|(_, c)| **c > 1)
        .map(|(k, _)| k.clone())
        .collect();
    dups.sort();
    let unique = counts.len() as u32;
    AliasDupVerdict::Ok {
        duplicates: dups,
        unique_count: unique,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_alias_dup_check")?;

    let aliases = ["common", "shared", "common", "extra"];
    println!("check: {:?}", check(&aliases));
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
    fn no_duplicates_empty_result() {
        let v = check(&["a", "b", "c"]);
        if let AliasDupVerdict::Ok { duplicates, .. } = v {
            assert!(duplicates.is_empty());
        }
    }

    #[test]
    fn single_dup_detected() {
        let v = check(&["a", "a"]);
        if let AliasDupVerdict::Ok { duplicates, .. } = v {
            assert_eq!(duplicates, vec!["a".to_string()]);
        }
    }

    #[test]
    fn unique_count_correct() {
        let v = check(&["a", "b", "a"]);
        if let AliasDupVerdict::Ok { unique_count, .. } = v {
            assert_eq!(unique_count, 2);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(&[]), AliasDupVerdict::InvalidConfig);
    }

    #[test]
    fn multi_dup_detected() {
        let v = check(&["a", "b", "a", "b", "c"]);
        if let AliasDupVerdict::Ok { duplicates, .. } = v {
            assert_eq!(duplicates, vec!["a".to_string(), "b".to_string()]);
        }
    }

    #[test]
    fn duplicates_sorted() {
        let v = check(&["zeta", "alpha", "zeta", "alpha"]);
        if let AliasDupVerdict::Ok { duplicates, .. } = v {
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
    fn triple_occurrence_dup() {
        let v = check(&["a", "a", "a"]);
        if let AliasDupVerdict::Ok { duplicates, .. } = v {
            assert_eq!(duplicates, vec!["a".to_string()]);
        }
    }

    #[test]
    fn case_sensitive() {
        let v = check(&["A", "a"]);
        if let AliasDupVerdict::Ok { duplicates, .. } = v {
            assert!(duplicates.is_empty());
        }
    }

    #[test]
    fn many_aliases_handled() {
        let aliases: Vec<&str> = (0..30).map(|_| "shared").collect();
        let v = check(&aliases);
        if let AliasDupVerdict::Ok { duplicates, .. } = v {
            assert_eq!(duplicates, vec!["shared".to_string()]);
        }
    }

    #[test]
    fn unicode_alias_supported() {
        let v = check(&["café", "café"]);
        if let AliasDupVerdict::Ok { duplicates, .. } = v {
            assert_eq!(duplicates, vec!["café".to_string()]);
        }
    }
}
