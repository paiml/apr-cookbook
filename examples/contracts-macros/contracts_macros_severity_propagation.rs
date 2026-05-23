//! # Contracts-Macros Severity Propagation
//!
//! Given a dependency graph of obligations + each leaf's own severity,
//! propagate `max(severity)` up to each parent. Returns final
//! severity per node and the list of root-level criticals.
//!
//! Demonstrates the **CMM.70** recipe for PMAT-181 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: fault tree analysis (Vesely, NUREG-0492, 1981).
//!
//! Run with: cargo run --example contracts_macros_severity_propagation
//!
//! Added by PMAT-181 (catalog 1252→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq, Clone, Copy, PartialOrd, Ord, Eq)]
pub enum Severity {
    Info,
    Warn,
    Error,
    Critical,
}

#[derive(Debug, PartialEq)]
pub enum PropagationVerdict {
    Ok {
        per_node: BTreeMap<String, Severity>,
        critical_roots: Vec<String>,
    },
    InvalidConfig,
}

pub fn propagate(
    leaf_severities: &[(&str, Severity)],
    edges: &[(&str, &str)],
) -> PropagationVerdict {
    if leaf_severities.is_empty() {
        return PropagationVerdict::InvalidConfig;
    }
    let mut sev: BTreeMap<String, Severity> = BTreeMap::new();
    for (node, s) in leaf_severities {
        sev.insert((*node).to_string(), *s);
    }
    let mut children: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut all_nodes: Vec<String> = sev.keys().cloned().collect();
    for (parent, child) in edges {
        children
            .entry((*parent).to_string())
            .or_default()
            .push((*child).to_string());
        for n in [parent, child] {
            if !all_nodes.iter().any(|x| x == *n) {
                all_nodes.push((*n).to_string());
            }
        }
    }
    let mut per_node: BTreeMap<String, Severity> = BTreeMap::new();
    for node in &all_nodes {
        let v = compute(node, &children, &sev);
        per_node.insert(node.clone(), v);
    }
    let mut roots: Vec<String> = all_nodes.clone();
    for (_, child_list) in &children {
        for c in child_list {
            roots.retain(|r| r != c);
        }
    }
    let mut critical_roots: Vec<String> = roots
        .iter()
        .filter(|r| per_node.get(*r) == Some(&Severity::Critical))
        .cloned()
        .collect();
    critical_roots.sort();
    PropagationVerdict::Ok {
        per_node,
        critical_roots,
    }
}

fn compute(
    node: &str,
    children: &BTreeMap<String, Vec<String>>,
    leaf_sev: &BTreeMap<String, Severity>,
) -> Severity {
    if let Some(child_list) = children.get(node) {
        let mut max = leaf_sev.get(node).copied().unwrap_or(Severity::Info);
        for c in child_list {
            let cs = compute(c, children, leaf_sev);
            if cs > max {
                max = cs;
            }
        }
        max
    } else {
        leaf_sev.get(node).copied().unwrap_or(Severity::Info)
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_severity_propagation")?;

    let leaves = [
        ("leaf_a", Severity::Warn),
        ("leaf_b", Severity::Critical),
        ("leaf_c", Severity::Info),
    ];
    let edges = [
        ("root_x", "leaf_a"),
        ("root_x", "leaf_b"),
        ("root_y", "leaf_c"),
    ];
    println!("audit: {:?}", propagate(&leaves, &edges));
    println!("invalid: {:?}", propagate(&[], &[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propagator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn parent_inherits_max_child() {
        let leaves = [("a", Severity::Warn), ("b", Severity::Critical)];
        let edges = [("root", "a"), ("root", "b")];
        let v = propagate(&leaves, &edges);
        if let PropagationVerdict::Ok { per_node, .. } = v {
            assert_eq!(per_node.get("root"), Some(&Severity::Critical));
        }
    }

    #[test]
    fn leaf_keeps_own_severity() {
        let leaves = [("leaf", Severity::Warn)];
        let v = propagate(&leaves, &[]);
        if let PropagationVerdict::Ok { per_node, .. } = v {
            assert_eq!(per_node.get("leaf"), Some(&Severity::Warn));
        }
    }

    #[test]
    fn critical_roots_collected() {
        let leaves = [("a", Severity::Critical)];
        let edges = [("root", "a")];
        let v = propagate(&leaves, &edges);
        if let PropagationVerdict::Ok { critical_roots, .. } = v {
            assert_eq!(critical_roots, vec!["root".to_string()]);
        }
    }

    #[test]
    fn non_critical_root_not_listed() {
        let leaves = [("a", Severity::Warn)];
        let edges = [("root", "a")];
        let v = propagate(&leaves, &edges);
        if let PropagationVerdict::Ok { critical_roots, .. } = v {
            assert!(critical_roots.is_empty());
        }
    }

    #[test]
    fn empty_leaves_rejected() {
        assert_eq!(propagate(&[], &[]), PropagationVerdict::InvalidConfig);
    }

    #[test]
    fn multi_level_propagation() {
        // grandparent → parent → leaf(critical).
        let leaves = [("leaf", Severity::Critical)];
        let edges = [("parent", "leaf"), ("grand", "parent")];
        let v = propagate(&leaves, &edges);
        if let PropagationVerdict::Ok { per_node, .. } = v {
            assert_eq!(per_node.get("grand"), Some(&Severity::Critical));
        }
    }

    #[test]
    fn deterministic() {
        let leaves = [("a", Severity::Error)];
        let edges = [("root", "a")];
        let r1 = propagate(&leaves, &edges);
        let r2 = propagate(&leaves, &edges);
        assert_eq!(r1, r2);
    }

    #[test]
    fn ordering_severity_levels() {
        assert!(Severity::Critical > Severity::Error);
        assert!(Severity::Error > Severity::Warn);
        assert!(Severity::Warn > Severity::Info);
    }

    #[test]
    fn root_with_multiple_children_takes_max() {
        let leaves = [
            ("a", Severity::Info),
            ("b", Severity::Error),
            ("c", Severity::Warn),
        ];
        let edges = [("root", "a"), ("root", "b"), ("root", "c")];
        let v = propagate(&leaves, &edges);
        if let PropagationVerdict::Ok { per_node, .. } = v {
            assert_eq!(per_node.get("root"), Some(&Severity::Error));
        }
    }

    #[test]
    fn isolated_leaf_in_per_node() {
        let leaves = [("orphan", Severity::Warn)];
        let v = propagate(&leaves, &[]);
        if let PropagationVerdict::Ok { per_node, .. } = v {
            assert!(per_node.contains_key("orphan"));
        }
    }

    #[test]
    fn unknown_node_default_info() {
        let leaves = [("real", Severity::Info)];
        let edges = [("placeholder", "real")];
        let v = propagate(&leaves, &edges);
        if let PropagationVerdict::Ok { per_node, .. } = v {
            assert_eq!(per_node.get("placeholder"), Some(&Severity::Info));
        }
    }
}
