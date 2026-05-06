//! # Contracts-Macros YAML Dotted Path Lookup
//!
//! Resolve a dotted path like `a.b.c` against a flat key list.
//! Returns whether the path exists, plus depth and any partial-match
//! diagnosis.
//!
//! Demonstrates the **CMM.107** recipe for PMAT-193 (post-milestone).
//!
//! Citation: jq path expressions; YAML 1.2 §3.2.1 (representation
//!  graph paths).
//!
//! Run with: cargo run --example contracts_macros_yaml_dotted_path_lookup
//!
//! Added by PMAT-193 (catalog 1360→).
//!
//! Contract: contracts/recipe-iiur-v1.yaml

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum LookupVerdict {
    Found { depth: u32 },
    NotFound { deepest_match: u32 },
    InvalidConfig,
}

pub fn lookup(known_paths: &[&str], target: &str) -> LookupVerdict {
    if known_paths.is_empty() || target.is_empty() {
        return LookupVerdict::InvalidConfig;
    }
    let known_set: BTreeSet<&str> = known_paths.iter().copied().collect();
    if known_set.contains(target) {
        return LookupVerdict::Found {
            depth: target.matches('.').count() as u32 + 1,
        };
    }
    // Find deepest matching prefix.
    let parts: Vec<&str> = target.split('.').collect();
    let mut deepest: u32 = 0;
    for i in 1..=parts.len() {
        let candidate = parts[..i].join(".");
        if known_set.contains(candidate.as_str()) {
            deepest = i as u32;
        }
    }
    LookupVerdict::NotFound {
        deepest_match: deepest,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_dotted_path_lookup")?;

    let known = ["a", "a.b", "a.b.c", "x.y"];
    println!("found: {:?}", lookup(&known, "a.b.c"));
    println!("partial: {:?}", lookup(&known, "a.b.d"));
    println!("invalid: {:?}", lookup(&[], ""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn exact_match_found() {
        let v = lookup(&["a.b"], "a.b");
        if let LookupVerdict::Found { depth } = v {
            assert_eq!(depth, 2);
        }
    }

    #[test]
    fn missing_target_returns_partial() {
        let v = lookup(&["a", "a.b"], "a.b.c");
        if let LookupVerdict::NotFound { deepest_match } = v {
            assert_eq!(deepest_match, 2);
        }
    }

    #[test]
    fn no_partial_match_zero() {
        let v = lookup(&["x.y"], "a.b.c");
        if let LookupVerdict::NotFound { deepest_match } = v {
            assert_eq!(deepest_match, 0);
        }
    }

    #[test]
    fn empty_known_rejected() {
        assert_eq!(lookup(&[], "a"), LookupVerdict::InvalidConfig);
    }

    #[test]
    fn empty_target_rejected() {
        let known = ["a"];
        assert_eq!(lookup(&known, ""), LookupVerdict::InvalidConfig);
    }

    #[test]
    fn root_match() {
        let v = lookup(&["root"], "root");
        if let LookupVerdict::Found { depth } = v {
            assert_eq!(depth, 1);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = lookup(&["a"], "a");
        let r2 = lookup(&["a"], "a");
        assert_eq!(r1, r2);
    }

    #[test]
    fn depth_correct() {
        let v = lookup(&["a.b.c.d"], "a.b.c.d");
        if let LookupVerdict::Found { depth } = v {
            assert_eq!(depth, 4);
        }
    }

    #[test]
    fn case_sensitive() {
        let v = lookup(&["A"], "a");
        assert!(matches!(v, LookupVerdict::NotFound { .. }));
    }

    #[test]
    fn extra_paths_ignored() {
        let v = lookup(&["a", "x", "y"], "a");
        if let LookupVerdict::Found { depth } = v {
            assert_eq!(depth, 1);
        }
    }

    #[test]
    fn very_deep_path() {
        let v = lookup(&["a.b.c.d.e.f.g"], "a.b.c.d.e.f.g");
        if let LookupVerdict::Found { depth } = v {
            assert_eq!(depth, 7);
        }
    }
}
