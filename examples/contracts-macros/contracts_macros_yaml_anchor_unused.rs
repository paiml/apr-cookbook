//! # Contracts-Macros YAML Anchor Unused
//!
//! Find YAML anchors defined but never referenced via alias `*name`.
//! Returns dead anchors plus reference count per anchor.
//!
//! Demonstrates the **CMM.116** recipe for PMAT-196 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: dead-code analysis (Aho et al., Compilers); YAML 1.2
//!  spec §6.9 (anchor/alias).
//!
//! Run with: cargo run --example contracts_macros_yaml_anchor_unused
//!
//! Added by PMAT-196 (catalog 1387→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum UnusedVerdict {
    Ok {
        unused_anchors: Vec<String>,
        ref_counts: BTreeMap<String, u32>,
    },
    InvalidConfig,
}

pub fn audit(defined: &[&str], referenced: &[&str]) -> UnusedVerdict {
    if defined.is_empty() {
        return UnusedVerdict::InvalidConfig;
    }
    let mut ref_counts: BTreeMap<String, u32> = BTreeMap::new();
    for anchor in defined {
        ref_counts.insert((*anchor).to_string(), 0);
    }
    for r in referenced {
        if let Some(c) = ref_counts.get_mut(*r) {
            *c += 1;
        }
    }
    let mut unused_anchors: Vec<String> = ref_counts
        .iter()
        .filter(|(_, &c)| c == 0)
        .map(|(name, _)| name.clone())
        .collect();
    unused_anchors.sort();
    UnusedVerdict::Ok {
        unused_anchors,
        ref_counts,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_anchor_unused")?;

    let defined = ["base", "config", "dead"];
    let referenced = ["base", "config", "config", "external"];
    println!("audit: {:?}", audit(&defined, &referenced));
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
    fn used_anchor_not_unused() {
        let v = audit(&["a"], &["a"]);
        if let UnusedVerdict::Ok { unused_anchors, .. } = v {
            assert!(unused_anchors.is_empty());
        }
    }

    #[test]
    fn unused_anchor_flagged() {
        let v = audit(&["a", "b"], &["a"]);
        if let UnusedVerdict::Ok { unused_anchors, .. } = v {
            assert_eq!(unused_anchors, vec!["b".to_string()]);
        }
    }

    #[test]
    fn empty_defined_rejected() {
        assert_eq!(audit(&[], &["a"]), UnusedVerdict::InvalidConfig);
    }

    #[test]
    fn external_refs_ignored() {
        let v = audit(&["a"], &["external"]);
        if let UnusedVerdict::Ok { unused_anchors, .. } = v {
            assert_eq!(unused_anchors, vec!["a".to_string()]);
        }
    }

    #[test]
    fn ref_counts_correct() {
        let v = audit(&["a"], &["a", "a", "a"]);
        if let UnusedVerdict::Ok { ref_counts, .. } = v {
            assert_eq!(ref_counts.get("a"), Some(&3));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&["a"], &["a"]);
        let r2 = audit(&["a"], &["a"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn unused_sorted() {
        let v = audit(&["zeta", "alpha"], &[]);
        if let UnusedVerdict::Ok { unused_anchors, .. } = v {
            assert_eq!(
                unused_anchors,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn all_unused_when_no_refs() {
        let v = audit(&["a", "b", "c"], &[]);
        if let UnusedVerdict::Ok { unused_anchors, .. } = v {
            assert_eq!(unused_anchors.len(), 3);
        }
    }

    #[test]
    fn case_sensitive() {
        let v = audit(&["A"], &["a"]);
        if let UnusedVerdict::Ok { unused_anchors, .. } = v {
            assert_eq!(unused_anchors, vec!["A".to_string()]);
        }
    }

    #[test]
    fn duplicate_defined_dedup() {
        let v = audit(&["a", "a"], &["a"]);
        if let UnusedVerdict::Ok { ref_counts, .. } = v {
            assert_eq!(ref_counts.len(), 1);
        }
    }

    #[test]
    fn many_anchors_handled() {
        let defined: Vec<&str> = vec!["a"; 30];
        let v = audit(&defined, &[]);
        if let UnusedVerdict::Ok { unused_anchors, .. } = v {
            assert_eq!(unused_anchors.len(), 1);
        }
    }
}
