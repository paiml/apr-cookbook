//! # Contracts-Macros Obligation Chain Audit
//!
//! Verify each obligation has at most one parent (no diamond
//! inheritance). Returns offending nodes (multi-parent) plus orphan
//! list (no parent and not declared as root).
//!
//! Demonstrates the **CMM.73** recipe for PMAT-182 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: tree property in dependency-tracking; Tarjan, Data
//!  Structures and Network Algorithms (1983) §3.
//!
//! Run with: cargo run --example contracts_macros_obligation_chain_audit
//!
//! Added by PMAT-182 (catalog 1261→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum ChainVerdict {
    Ok {
        multi_parent: Vec<String>,
        orphans: Vec<String>,
    },
    InvalidConfig,
}

pub fn audit(edges: &[(&str, &str)], roots: &[&str]) -> ChainVerdict {
    if edges.is_empty() && roots.is_empty() {
        return ChainVerdict::InvalidConfig;
    }
    let mut parent_count: BTreeMap<String, u32> = BTreeMap::new();
    let mut all_nodes: BTreeSet<String> = BTreeSet::new();
    for (parent, child) in edges {
        *parent_count.entry((*child).to_string()).or_insert(0) += 1;
        all_nodes.insert((*parent).to_string());
        all_nodes.insert((*child).to_string());
    }
    for r in roots {
        all_nodes.insert((*r).to_string());
    }
    let multi_parent: Vec<String> = parent_count
        .iter()
        .filter(|(_, &c)| c > 1)
        .map(|(k, _)| k.clone())
        .collect();
    let root_set: BTreeSet<&str> = roots.iter().copied().collect();
    let orphans: Vec<String> = all_nodes
        .iter()
        .filter(|n| {
            parent_count.get(n.as_str()).copied().unwrap_or(0) == 0
                && !root_set.contains(n.as_str())
        })
        .cloned()
        .collect();
    ChainVerdict::Ok {
        multi_parent,
        orphans,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_chain_audit")?;

    let edges = [("root", "a"), ("root", "b"), ("a", "leaf")];
    println!("clean tree: {:?}", audit(&edges, &["root"]));
    let dia = [("root", "a"), ("root", "b"), ("a", "leaf"), ("b", "leaf")];
    println!("diamond: {:?}", audit(&dia, &["root"]));
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
    fn clean_tree_no_violations() {
        let edges = [("root", "a"), ("a", "leaf")];
        let v = audit(&edges, &["root"]);
        if let ChainVerdict::Ok {
            multi_parent,
            orphans,
        } = v
        {
            assert!(multi_parent.is_empty());
            assert!(orphans.is_empty());
        }
    }

    #[test]
    fn diamond_creates_multi_parent() {
        let edges = [("a", "leaf"), ("b", "leaf")];
        let v = audit(&edges, &["a", "b"]);
        if let ChainVerdict::Ok { multi_parent, .. } = v {
            assert_eq!(multi_parent, vec!["leaf".to_string()]);
        }
    }

    #[test]
    fn unrooted_node_is_orphan() {
        let edges = [("a", "b")];
        let v = audit(&edges, &[]);
        if let ChainVerdict::Ok { orphans, .. } = v {
            assert!(orphans.contains(&"a".to_string()));
        }
    }

    #[test]
    fn declared_root_not_orphan() {
        let edges = [("root", "leaf")];
        let v = audit(&edges, &["root"]);
        if let ChainVerdict::Ok { orphans, .. } = v {
            assert!(!orphans.contains(&"root".to_string()));
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[], &[]), ChainVerdict::InvalidConfig);
    }

    #[test]
    fn multi_parent_sorted() {
        let edges = [("a", "zeta"), ("b", "zeta"), ("c", "alpha"), ("d", "alpha")];
        let v = audit(&edges, &["a", "b", "c", "d"]);
        if let ChainVerdict::Ok { multi_parent, .. } = v {
            assert_eq!(multi_parent, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn orphans_sorted() {
        let edges = [("zeta", "x"), ("alpha", "y")];
        let v = audit(&edges, &[]);
        if let ChainVerdict::Ok { orphans, .. } = v {
            // alpha and zeta are unrooted; x and y have parents.
            assert_eq!(orphans, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let edges = [("a", "b")];
        let r1 = audit(&edges, &["a"]);
        let r2 = audit(&edges, &["a"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn three_parents_one_node() {
        let edges = [("a", "x"), ("b", "x"), ("c", "x")];
        let v = audit(&edges, &["a", "b", "c"]);
        if let ChainVerdict::Ok { multi_parent, .. } = v {
            assert_eq!(multi_parent, vec!["x".to_string()]);
        }
    }

    #[test]
    fn isolated_root_no_orphan_no_multi() {
        let v = audit(&[], &["root"]);
        if let ChainVerdict::Ok {
            multi_parent,
            orphans,
        } = v
        {
            assert!(multi_parent.is_empty());
            assert!(orphans.is_empty());
        }
    }

    #[test]
    fn long_chain_clean() {
        let edges = [("a", "b"), ("b", "c"), ("c", "d")];
        let v = audit(&edges, &["a"]);
        if let ChainVerdict::Ok {
            multi_parent,
            orphans,
        } = v
        {
            assert!(multi_parent.is_empty());
            assert!(orphans.is_empty());
        }
    }
}
