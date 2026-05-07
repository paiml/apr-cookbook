//! # Contracts-Macros Obligation Split Grouping
//!
//! Split a flat obligation list into per-topic groups by prefix.
//! Returns groups + count per group.
//!
//! Demonstrates the **CMM.130** recipe for PMAT-201 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: namespace partitioning (Codd 1970, RM); module
//!  decomposition (Parnas 1972).
//!
//! Run with: cargo run --example contracts_macros_obligation_split_grouping
//!
//! Added by PMAT-201 (catalog 1432→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum SplitVerdict {
    Ok {
        groups: BTreeMap<String, Vec<String>>,
        group_count: u32,
    },
    InvalidConfig,
}

pub fn split(obligations: &[&str], delimiter: char) -> SplitVerdict {
    if obligations.is_empty() {
        return SplitVerdict::InvalidConfig;
    }
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for ob in obligations {
        let prefix = ob.split(delimiter).next().unwrap_or(ob).to_string();
        groups.entry(prefix).or_default().push((*ob).to_string());
    }
    let group_count = groups.len() as u32;
    SplitVerdict::Ok {
        groups,
        group_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_split_grouping")?;

    let obligations = ["auth.login", "auth.logout", "data.read", "data.write"];
    println!("by dot: {:?}", split(&obligations, '.'));
    println!("invalid: {:?}", split(&[], '.'));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn dot_split_groups() {
        let v = split(&["a.x", "a.y", "b.z"], '.');
        if let SplitVerdict::Ok { groups, .. } = v {
            assert_eq!(groups.get("a").unwrap().len(), 2);
            assert_eq!(groups.get("b").unwrap().len(), 1);
        }
    }

    #[test]
    fn no_delimiter_single_group() {
        let v = split(&["foo"], '.');
        if let SplitVerdict::Ok { group_count, .. } = v {
            assert_eq!(group_count, 1);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(split(&[], '.'), SplitVerdict::InvalidConfig);
    }

    #[test]
    fn group_count_correct() {
        let v = split(&["a.x", "b.y", "c.z"], '.');
        if let SplitVerdict::Ok { group_count, .. } = v {
            assert_eq!(group_count, 3);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = split(&["a.x"], '.');
        let r2 = split(&["a.x"], '.');
        assert_eq!(r1, r2);
    }

    #[test]
    fn case_sensitive_grouping() {
        let v = split(&["A.x", "a.y"], '.');
        if let SplitVerdict::Ok { groups, .. } = v {
            assert!(groups.contains_key("A"));
            assert!(groups.contains_key("a"));
        }
    }

    #[test]
    fn slash_delimiter_works() {
        let v = split(&["dir/file"], '/');
        if let SplitVerdict::Ok { groups, .. } = v {
            assert!(groups.contains_key("dir"));
        }
    }

    #[test]
    fn duplicate_in_group() {
        let v = split(&["a.x", "a.x"], '.');
        if let SplitVerdict::Ok { groups, .. } = v {
            assert_eq!(groups.get("a").unwrap().len(), 2);
        }
    }

    #[test]
    fn many_obligations_handled() {
        let obs: Vec<&str> = vec!["a.x"; 50];
        let v = split(&obs, '.');
        if let SplitVerdict::Ok { groups, .. } = v {
            assert_eq!(groups.get("a").unwrap().len(), 50);
        }
    }

    #[test]
    fn unicode_prefix_supported() {
        let v = split(&["café.x", "résumé.y"], '.');
        if let SplitVerdict::Ok { group_count, .. } = v {
            assert_eq!(group_count, 2);
        }
    }

    #[test]
    fn empty_prefix_handled() {
        let v = split(&[".key"], '.');
        if let SplitVerdict::Ok { groups, .. } = v {
            assert!(groups.contains_key(""));
        }
    }
}
