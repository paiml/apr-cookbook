//! # Contracts-Macros Recipe Orphan Detector
//!
//! Cross-reference `(declared_recipes, referenced_recipes)`. Find:
//! - **orphan recipes**: declared but never referenced
//! - **dangling refs**: referenced but never declared
//!
//! Useful for keeping recipe catalog and contracts in sync.
//!
//! Demonstrates the **CMM.66** recipe for PMAT-179 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: dead-code analysis (Aho et al., Compilers §10.1).
//!
//! Run with: cargo run --example contracts_macros_recipe_orphan_detector
//!
//! Added by PMAT-179 (catalog 1234→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum OrphanVerdict {
    Ok {
        orphans: Vec<String>,
        dangling: Vec<String>,
    },
    InvalidConfig,
}

pub fn detect(declared: &[&str], referenced: &[&str]) -> OrphanVerdict {
    if declared.is_empty() && referenced.is_empty() {
        return OrphanVerdict::InvalidConfig;
    }
    let dec: BTreeSet<&str> = declared.iter().copied().collect();
    let refs: BTreeSet<&str> = referenced.iter().copied().collect();
    let orphans: Vec<String> = dec.difference(&refs).map(|s| (*s).to_string()).collect();
    let dangling: Vec<String> = refs.difference(&dec).map(|s| (*s).to_string()).collect();
    OrphanVerdict::Ok { orphans, dangling }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_orphan_detector")?;

    let declared = ["r_alpha", "r_beta", "r_gamma"];
    let referenced = ["r_beta", "r_gamma", "r_missing"];
    println!("audit: {:?}", detect(&declared, &referenced));
    println!("invalid: {:?}", detect(&[], &[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn perfect_match_no_orphans() {
        let v = detect(&["a", "b"], &["a", "b"]);
        if let OrphanVerdict::Ok { orphans, dangling } = v {
            assert!(orphans.is_empty());
            assert!(dangling.is_empty());
        }
    }

    #[test]
    fn unreferenced_is_orphan() {
        let v = detect(&["a", "b", "c"], &["a"]);
        if let OrphanVerdict::Ok { orphans, .. } = v {
            assert_eq!(orphans, vec!["b".to_string(), "c".to_string()]);
        }
    }

    #[test]
    fn undeclared_ref_is_dangling() {
        let v = detect(&["a"], &["a", "b", "c"]);
        if let OrphanVerdict::Ok { dangling, .. } = v {
            assert_eq!(dangling, vec!["b".to_string(), "c".to_string()]);
        }
    }

    #[test]
    fn empty_inputs_rejected() {
        assert_eq!(detect(&[], &[]), OrphanVerdict::InvalidConfig);
    }

    #[test]
    fn empty_referenced_all_orphans() {
        let v = detect(&["a", "b"], &[]);
        if let OrphanVerdict::Ok { orphans, dangling } = v {
            assert_eq!(orphans, vec!["a".to_string(), "b".to_string()]);
            assert!(dangling.is_empty());
        }
    }

    #[test]
    fn empty_declared_all_dangling() {
        let v = detect(&[], &["a", "b"]);
        if let OrphanVerdict::Ok { orphans, dangling } = v {
            assert!(orphans.is_empty());
            assert_eq!(dangling, vec!["a".to_string(), "b".to_string()]);
        }
    }

    #[test]
    fn duplicates_collapse() {
        let v = detect(&["a", "a", "b"], &["a", "a"]);
        if let OrphanVerdict::Ok { orphans, .. } = v {
            assert_eq!(orphans, vec!["b".to_string()]);
        }
    }

    #[test]
    fn results_alphabetically_sorted() {
        let v = detect(&["zeta", "alpha"], &[]);
        if let OrphanVerdict::Ok { orphans, .. } = v {
            assert_eq!(orphans, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let a = detect(&["x", "y"], &["x"]);
        let b = detect(&["x", "y"], &["x"]);
        assert_eq!(a, b);
    }

    #[test]
    fn both_orphans_and_dangling() {
        let v = detect(&["a", "b"], &["b", "c"]);
        if let OrphanVerdict::Ok { orphans, dangling } = v {
            assert_eq!(orphans, vec!["a".to_string()]);
            assert_eq!(dangling, vec!["c".to_string()]);
        }
    }

    #[test]
    fn idempotent() {
        let v1 = detect(&["a", "b"], &["b"]);
        let v2 = detect(&["a", "b"], &["b"]);
        assert_eq!(v1, v2);
    }
}
