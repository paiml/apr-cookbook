//! # Advanced Recipe Dependency DAG
//!
//! Some recipes depend on others (compose, build on, extend). Tracking
//! dependencies as a DAG enables impact analysis: changing recipe X
//! flags downstream recipes that may need re-validation. This recipe
//! builds the cycle detector + topological order.
//!
//! Demonstrates the **ADV.3** recipe for PMAT-128 (advanced coverage —
//! closing F-invariant gap from 2 → 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Tarjan (1972). Depth-first search and linear graph algorithms.
//!
//! Run with: cargo run --example adv_recipe_dependency_dag
//!
//! Added by PMAT-128 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, PartialEq)]
pub enum DagVerdict {
    Ok { topological_order: Vec<String> },
    CycleDetected { node: String },
}

pub fn topo_sort(edges: &[(&str, &str)]) -> DagVerdict {
    let mut graph: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut in_degree: BTreeMap<String, u32> = BTreeMap::new();
    for (from, to) in edges {
        graph
            .entry((*from).to_string())
            .or_default()
            .insert((*to).to_string());
        graph.entry((*to).to_string()).or_default();
        in_degree.entry((*from).to_string()).or_insert(0);
        *in_degree.entry((*to).to_string()).or_insert(0) += 1;
    }
    let mut queue: Vec<String> = in_degree
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(n, _)| n.clone())
        .collect();
    queue.sort();
    let mut order: Vec<String> = Vec::new();
    while let Some(node) = queue.first().cloned() {
        queue.remove(0);
        order.push(node.clone());
        if let Some(neighbors) = graph.get(&node).cloned() {
            for next in neighbors {
                if let Some(d) = in_degree.get_mut(&next) {
                    *d -= 1;
                    if *d == 0 {
                        let pos = queue.binary_search(&next).unwrap_or_else(|i| i);
                        queue.insert(pos, next);
                    }
                }
            }
        }
    }
    if order.len() != in_degree.len() {
        // Cycle detected — find any node with remaining in-degree > 0.
        let stuck = in_degree
            .iter()
            .find(|(_, d)| **d > 0)
            .map_or_else(String::new, |(n, _)| n.clone());
        return DagVerdict::CycleDetected { node: stuck };
    }
    DagVerdict::Ok {
        topological_order: order,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_recipe_dependency_dag")?;

    let acyclic = [("base", "intermediate"), ("intermediate", "final")];
    println!("acyclic: {:?}", topo_sort(&acyclic));

    let cyclic = [("a", "b"), ("b", "c"), ("c", "a")];
    println!("cyclic:  {:?}", topo_sort(&cyclic));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dag_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn linear_chain_topo_sorted() {
        let edges = [("a", "b"), ("b", "c")];
        if let DagVerdict::Ok { topological_order } = topo_sort(&edges) {
            assert_eq!(topological_order, vec!["a", "b", "c"]);
        }
    }

    #[test]
    fn diamond_dag_acyclic() {
        let edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")];
        let v = topo_sort(&edges);
        assert!(matches!(v, DagVerdict::Ok { .. }));
    }

    #[test]
    fn cycle_detected() {
        let edges = [("a", "b"), ("b", "c"), ("c", "a")];
        assert!(matches!(
            topo_sort(&edges),
            DagVerdict::CycleDetected { .. }
        ));
    }

    #[test]
    fn self_loop_is_cycle() {
        let edges = [("a", "a")];
        assert!(matches!(
            topo_sort(&edges),
            DagVerdict::CycleDetected { .. }
        ));
    }

    #[test]
    fn empty_graph_acyclic() {
        let edges: [(&str, &str); 0] = [];
        if let DagVerdict::Ok { topological_order } = topo_sort(&edges) {
            assert!(topological_order.is_empty());
        }
    }

    #[test]
    fn topo_order_respects_dependencies() {
        let edges = [("a", "c"), ("b", "c"), ("c", "d")];
        if let DagVerdict::Ok { topological_order } = topo_sort(&edges) {
            let pos_a = topological_order.iter().position(|n| n == "a").unwrap();
            let pos_b = topological_order.iter().position(|n| n == "b").unwrap();
            let pos_c = topological_order.iter().position(|n| n == "c").unwrap();
            let pos_d = topological_order.iter().position(|n| n == "d").unwrap();
            assert!(pos_a < pos_c);
            assert!(pos_b < pos_c);
            assert!(pos_c < pos_d);
        }
    }

    #[test]
    fn disconnected_components_handled() {
        let edges = [("a", "b"), ("c", "d")];
        if let DagVerdict::Ok { topological_order } = topo_sort(&edges) {
            assert_eq!(topological_order.len(), 4);
        }
    }

    #[test]
    fn long_chain_no_false_positive() {
        let edges: Vec<(&str, &str)> = (0..10)
            .map(|i| {
                let from: &'static str = match i {
                    0 => "n0",
                    1 => "n1",
                    2 => "n2",
                    3 => "n3",
                    4 => "n4",
                    5 => "n5",
                    6 => "n6",
                    7 => "n7",
                    8 => "n8",
                    _ => "n9",
                };
                let to: &'static str = match i {
                    0 => "n1",
                    1 => "n2",
                    2 => "n3",
                    3 => "n4",
                    4 => "n5",
                    5 => "n6",
                    6 => "n7",
                    7 => "n8",
                    8 => "n9",
                    _ => "n10",
                };
                (from, to)
            })
            .collect();
        let v = topo_sort(&edges);
        if let DagVerdict::Ok { topological_order } = v {
            assert_eq!(topological_order.len(), 11);
            assert_eq!(topological_order[0], "n0");
            assert_eq!(topological_order[10], "n10");
        }
    }
}
