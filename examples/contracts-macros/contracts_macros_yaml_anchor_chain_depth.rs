//! # Contracts-Macros YAML Anchor Chain Depth
//!
//! Limit how deeply anchor aliases can transitively reference each
//! other. Returns sorted offending anchor names and the maximum chain
//! depth observed.
//!
//! Demonstrates the **CMM.151** recipe for PMAT-208 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 §6.9 alias semantics; CVE-2017-18342 yaml-bomb
//!  exhaustion vector.
//!
//! Run with: cargo run --example contracts_macros_yaml_anchor_chain_depth
//!
//! Added by PMAT-208 (catalog 1495→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum AnchorChainVerdict {
    Ok {
        offending_anchors: Vec<String>,
        max_depth_observed: u32,
    },
    InvalidConfig,
}

pub fn check(refs: &[(&str, &str)], max_allowed: u32) -> AnchorChainVerdict {
    if refs.is_empty() || max_allowed == 0 {
        return AnchorChainVerdict::InvalidConfig;
    }
    let mut adj: BTreeMap<String, String> = BTreeMap::new();
    for (anchor, target) in refs {
        adj.insert((*anchor).to_string(), (*target).to_string());
    }
    let mut offenders: Vec<String> = Vec::new();
    let mut max_depth = 0u32;
    for anchor in adj.keys() {
        let depth = chain_depth(anchor, &adj, 0, max_allowed + 1);
        if depth > max_depth {
            max_depth = depth;
        }
        if depth > max_allowed {
            offenders.push(anchor.clone());
        }
    }
    offenders.sort();
    AnchorChainVerdict::Ok {
        offending_anchors: offenders,
        max_depth_observed: max_depth,
    }
}

fn chain_depth(node: &str, adj: &BTreeMap<String, String>, d: u32, cap: u32) -> u32 {
    if d >= cap {
        return d;
    }
    match adj.get(node) {
        Some(next) => chain_depth(next, adj, d + 1, cap),
        None => d,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_anchor_chain_depth")?;

    let refs = [("a", "b"), ("b", "c")];
    println!("max-3: {:?}", check(&refs, 3));
    println!("max-1: {:?}", check(&refs, 1));
    println!("invalid: {:?}", check(&[], 3));
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
    fn shallow_chain_no_offender() {
        let v = check(&[("a", "b")], 3);
        if let AnchorChainVerdict::Ok {
            offending_anchors, ..
        } = v
        {
            assert!(offending_anchors.is_empty());
        }
    }

    #[test]
    fn deep_chain_offender() {
        let v = check(&[("a", "b"), ("b", "c"), ("c", "d")], 1);
        if let AnchorChainVerdict::Ok {
            offending_anchors, ..
        } = v
        {
            assert!(offending_anchors.contains(&"a".to_string()));
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(&[], 3), AnchorChainVerdict::InvalidConfig);
    }

    #[test]
    fn zero_max_rejected() {
        assert_eq!(check(&[("a", "b")], 0), AnchorChainVerdict::InvalidConfig);
    }

    #[test]
    fn max_depth_reported() {
        let v = check(&[("a", "b"), ("b", "c")], 5);
        if let AnchorChainVerdict::Ok {
            max_depth_observed, ..
        } = v
        {
            assert_eq!(max_depth_observed, 2);
        }
    }

    #[test]
    fn at_max_in_band() {
        let v = check(&[("a", "b"), ("b", "c")], 2);
        if let AnchorChainVerdict::Ok {
            offending_anchors, ..
        } = v
        {
            assert!(offending_anchors.is_empty());
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check(&[("a", "b")], 3);
        let r2 = check(&[("a", "b")], 3);
        assert_eq!(r1, r2);
    }

    #[test]
    fn offenders_sorted() {
        let v = check(
            &[("zeta", "x"), ("alpha", "y"), ("x", "long"), ("y", "long2")],
            1,
        );
        if let AnchorChainVerdict::Ok {
            offending_anchors, ..
        } = v
        {
            for w in offending_anchors.windows(2) {
                assert!(w[0] < w[1]);
            }
        }
    }

    #[test]
    fn cycle_capped() {
        // Cycle: a→b→a. Walk should cap at max_allowed+1.
        let v = check(&[("a", "b"), ("b", "a")], 5);
        if let AnchorChainVerdict::Ok {
            max_depth_observed, ..
        } = v
        {
            assert!(max_depth_observed <= 6);
        }
    }

    #[test]
    fn many_anchors_handled() {
        let refs: Vec<(&str, &str)> = (0..20).map(|_| ("k", "v")).collect();
        let v = check(&refs, 3);
        assert!(matches!(v, AnchorChainVerdict::Ok { .. }));
    }

    #[test]
    fn unicode_anchor_supported() {
        let v = check(&[("café", "résumé")], 3);
        if let AnchorChainVerdict::Ok {
            offending_anchors, ..
        } = v
        {
            assert!(offending_anchors.is_empty());
        }
    }

    #[test]
    fn isolated_anchor_depth_one() {
        let v = check(&[("a", "b")], 5);
        if let AnchorChainVerdict::Ok {
            max_depth_observed, ..
        } = v
        {
            assert_eq!(max_depth_observed, 1);
        }
    }
}
