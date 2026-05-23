//! # Contracts-Macros Recipe Namespace Audit
//!
//! Verify recipes share consistent namespace prefixes. Returns sorted
//! recipes that lack the expected prefix and the most-common
//! namespace observed.
//!
//! Demonstrates the **CMM.197** recipe for PMAT-223 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: cargo crate-name namespace conventions; npm scoped-
//!  package `@scope/name` rules.
//!
//! Run with: cargo run --example contracts_macros_recipe_namespace_audit
//!
//! Added by PMAT-223 (catalog 1630→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum NamespaceVerdict {
    Ok {
        offending_ids: Vec<String>,
        dominant_namespace: String,
    },
    InvalidConfig,
}

pub fn audit(recipes: &[&str]) -> NamespaceVerdict {
    if recipes.is_empty() {
        return NamespaceVerdict::InvalidConfig;
    }
    let mut ns_counts: BTreeMap<String, u32> = BTreeMap::new();
    for r in recipes {
        if let Some(idx) = r.find('_') {
            let ns = &r[..idx];
            *ns_counts.entry(ns.to_string()).or_insert(0) += 1;
        } else {
            *ns_counts.entry("(none)".to_string()).or_insert(0) += 1;
        }
    }
    let dominant = ns_counts
        .iter()
        .max_by_key(|(_, c)| *c)
        .map(|(k, _)| k.clone())
        .unwrap_or_default();
    let mut offenders: Vec<String> = recipes
        .iter()
        .filter(|r| {
            let prefix = r.split('_').next().unwrap_or("(none)");
            prefix != dominant
        })
        .map(|r| (*r).to_string())
        .collect();
    offenders.sort();
    NamespaceVerdict::Ok {
        offending_ids: offenders,
        dominant_namespace: dominant,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_namespace_audit")?;

    let recipes = ["mc_walk", "mc_pi", "tui_render", "mc_kelly"];
    println!("audit: {:?}", audit(&recipes));
    println!("invalid: {:?}", audit(&[]));
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
    fn all_same_ns_no_offender() {
        let v = audit(&["mc_a", "mc_b"]);
        if let NamespaceVerdict::Ok { offending_ids, .. } = v {
            assert!(offending_ids.is_empty());
        }
    }

    #[test]
    fn outlier_flagged() {
        let v = audit(&["mc_a", "mc_b", "tui_x"]);
        if let NamespaceVerdict::Ok { offending_ids, .. } = v {
            assert_eq!(offending_ids, vec!["tui_x".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), NamespaceVerdict::InvalidConfig);
    }

    #[test]
    fn dominant_correct() {
        let v = audit(&["mc_a", "mc_b", "mc_c", "tui_x"]);
        if let NamespaceVerdict::Ok {
            dominant_namespace, ..
        } = v
        {
            assert_eq!(dominant_namespace, "mc");
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&["mc_a"]);
        let r2 = audit(&["mc_a"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn no_underscore_classified_none() {
        let v = audit(&["plain"]);
        if let NamespaceVerdict::Ok {
            dominant_namespace, ..
        } = v
        {
            assert_eq!(dominant_namespace, "(none)");
        }
    }

    #[test]
    fn offenders_sorted() {
        // 3 distinct namespaces (mc/zeta/alpha) all tied at 1.
        // BTreeMap.max_by_key picks the last on ties → "zeta" wins.
        // Offenders are mc_a + alpha_c → sorted ascending.
        let v = audit(&["mc_a", "zeta_b", "alpha_c"]);
        if let NamespaceVerdict::Ok { offending_ids, .. } = v {
            for w in offending_ids.windows(2) {
                assert!(w[0] < w[1]);
            }
        }
    }

    #[test]
    fn many_recipes_handled() {
        let recipes: Vec<&str> = (0..30).map(|_| "mc_x").collect();
        let v = audit(&recipes);
        if let NamespaceVerdict::Ok { offending_ids, .. } = v {
            assert!(offending_ids.is_empty());
        }
    }

    #[test]
    fn unicode_ns_supported() {
        let v = audit(&["café_a", "café_b"]);
        if let NamespaceVerdict::Ok {
            dominant_namespace, ..
        } = v
        {
            assert_eq!(dominant_namespace, "café");
        }
    }

    #[test]
    fn case_sensitive_ns() {
        let v = audit(&["Mc_a", "mc_a"]);
        if let NamespaceVerdict::Ok { offending_ids, .. } = v {
            assert_eq!(offending_ids.len(), 1);
        }
    }

    #[test]
    fn single_recipe_dominant_self() {
        let v = audit(&["only_one"]);
        if let NamespaceVerdict::Ok {
            dominant_namespace, ..
        } = v
        {
            assert_eq!(dominant_namespace, "only");
        }
    }
}
