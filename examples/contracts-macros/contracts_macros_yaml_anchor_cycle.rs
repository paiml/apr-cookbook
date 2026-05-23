//! # Contracts-Macros YAML Anchor Cycle Detector
//!
//! Detect cycles in YAML anchor/alias references like
//! `a: &a [*b]` + `b: &b [*a]`. Returns NoCycle, CycleDetected with
//! the cycle path, or InvalidConfig.
//!
//! Demonstrates the **CMM.68** recipe for PMAT-180 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 spec §6.9 (alias node); Tarjan SCC algorithm.
//!
//! Run with: cargo run --example contracts_macros_yaml_anchor_cycle
//!
//! Added by PMAT-180 (catalog 1243→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum CycleVerdict {
    NoCycle,
    CycleDetected { path: Vec<String> },
    InvalidConfig,
}

pub fn detect(edges: &[(&str, &str)]) -> CycleVerdict {
    if edges.is_empty() {
        return CycleVerdict::InvalidConfig;
    }
    let mut graph: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (from, to) in edges {
        graph
            .entry((*from).to_string())
            .or_default()
            .push((*to).to_string());
    }
    let nodes: Vec<String> = graph.keys().cloned().collect();
    for start in &nodes {
        let mut visited: BTreeSet<String> = BTreeSet::new();
        let mut path: Vec<String> = Vec::new();
        if dfs(&graph, start, &mut visited, &mut path) {
            return CycleVerdict::CycleDetected { path };
        }
    }
    CycleVerdict::NoCycle
}

fn dfs(
    graph: &BTreeMap<String, Vec<String>>,
    node: &str,
    visited: &mut BTreeSet<String>,
    path: &mut Vec<String>,
) -> bool {
    if path.iter().any(|p| p == node) {
        path.push(node.to_string());
        return true;
    }
    if visited.contains(node) {
        return false;
    }
    visited.insert(node.to_string());
    path.push(node.to_string());
    if let Some(neighbors) = graph.get(node) {
        for n in neighbors {
            if dfs(graph, n, visited, path) {
                return true;
            }
        }
    }
    path.pop();
    false
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_anchor_cycle")?;

    let acyclic = [("a", "b"), ("b", "c")];
    println!("acyclic: {:?}", detect(&acyclic));
    let cyclic = [("a", "b"), ("b", "a")];
    println!("cyclic: {:?}", detect(&cyclic));
    println!("invalid: {:?}", detect(&[]));
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
    fn linear_chain_no_cycle() {
        let edges = [("a", "b"), ("b", "c"), ("c", "d")];
        assert_eq!(detect(&edges), CycleVerdict::NoCycle);
    }

    #[test]
    fn two_cycle_detected() {
        let edges = [("a", "b"), ("b", "a")];
        let v = detect(&edges);
        assert!(matches!(v, CycleVerdict::CycleDetected { .. }));
    }

    #[test]
    fn three_cycle_detected() {
        let edges = [("a", "b"), ("b", "c"), ("c", "a")];
        let v = detect(&edges);
        assert!(matches!(v, CycleVerdict::CycleDetected { .. }));
    }

    #[test]
    fn self_loop_detected() {
        let edges = [("a", "a")];
        let v = detect(&edges);
        assert!(matches!(v, CycleVerdict::CycleDetected { .. }));
    }

    #[test]
    fn empty_edges_rejected() {
        assert_eq!(detect(&[]), CycleVerdict::InvalidConfig);
    }

    #[test]
    fn dag_with_diamond_no_cycle() {
        // a → b, a → c, b → d, c → d.
        let edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")];
        assert_eq!(detect(&edges), CycleVerdict::NoCycle);
    }

    #[test]
    fn cycle_path_contains_offender() {
        let edges = [("x", "y"), ("y", "x")];
        let v = detect(&edges);
        if let CycleVerdict::CycleDetected { path } = v {
            assert!(path.iter().any(|p| p == "x") && path.iter().any(|p| p == "y"));
        }
    }

    #[test]
    fn deterministic() {
        let edges = [("a", "b"), ("b", "c")];
        let r1 = detect(&edges);
        let r2 = detect(&edges);
        assert_eq!(r1, r2);
    }

    #[test]
    fn isolated_disjoint_components_no_cycle() {
        let edges = [("a", "b"), ("c", "d")];
        assert_eq!(detect(&edges), CycleVerdict::NoCycle);
    }

    #[test]
    fn cycle_in_branch_detected() {
        // a → b → c → b (cycle in subtree, but a is not in cycle).
        let edges = [("a", "b"), ("b", "c"), ("c", "b")];
        let v = detect(&edges);
        assert!(matches!(v, CycleVerdict::CycleDetected { .. }));
    }

    #[test]
    fn long_chain_no_cycle() {
        let edges: Vec<(&str, &str)> =
            vec![("a", "b"), ("b", "c"), ("c", "d"), ("d", "e"), ("e", "f")];
        assert_eq!(detect(&edges), CycleVerdict::NoCycle);
    }
}
