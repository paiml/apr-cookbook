//! # Contracts-Macros Dependency Graph Validator
//!
//! Verify a `metadata.depends_on:` graph: no cycles, every reference
//! resolves to a declared equation, no self-loops.
//!
//! Demonstrates the **CMM.17** recipe for PMAT-163 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: DAG topological sort (Kahn's algorithm).
//!
//! Run with: cargo run --example contracts_macros_dependency_graph
//!
//! Added by PMAT-163 (catalog 1090→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, PartialEq)]
pub enum GraphVerdict {
    Ok {
        topological_order: Vec<String>,
    },
    UnknownReference {
        equation: String,
        references: String,
    },
    SelfLoop {
        equation: String,
    },
    Cycle,
    EmptyGraph,
}

pub fn validate(equations: &[(&str, Vec<&str>)]) -> GraphVerdict {
    if equations.is_empty() {
        return GraphVerdict::EmptyGraph;
    }
    let names: BTreeSet<&str> = equations.iter().map(|(n, _)| *n).collect();
    let mut in_degree: BTreeMap<&str, usize> = names.iter().map(|n| (*n, 0)).collect();
    let mut graph: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (name, deps) in equations {
        for d in deps {
            if d == name {
                return GraphVerdict::SelfLoop {
                    equation: (*name).to_string(),
                };
            }
            if !names.contains(d) {
                return GraphVerdict::UnknownReference {
                    equation: (*name).to_string(),
                    references: (*d).to_string(),
                };
            }
            *in_degree.entry(name).or_insert(0) += 1;
            graph.entry(d).or_default().push(name);
        }
    }
    let mut topo: Vec<String> = Vec::with_capacity(equations.len());
    let mut queue: Vec<&str> = in_degree
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(n, _)| *n)
        .collect();
    while let Some(n) = queue.pop() {
        topo.push(n.to_string());
        if let Some(succs) = graph.get(n) {
            for s in succs {
                let d = in_degree.entry(s).or_insert(0);
                *d = d.saturating_sub(1);
                if *d == 0 {
                    queue.push(s);
                }
            }
        }
    }
    if topo.len() != equations.len() {
        return GraphVerdict::Cycle;
    }
    GraphVerdict::Ok {
        topological_order: topo,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_dependency_graph")?;

    let dag = vec![("a", vec![]), ("b", vec!["a"]), ("c", vec!["a", "b"])];
    println!("dag: {:?}", validate(&dag));

    let cycle = vec![("a", vec!["b"]), ("b", vec!["a"])];
    println!("cycle: {:?}", validate(&cycle));

    let unknown = vec![("a", vec!["missing"])];
    println!("unknown: {:?}", validate(&unknown));

    let self_loop = vec![("a", vec!["a"])];
    println!("self loop: {:?}", validate(&self_loop));

    println!("empty: {:?}", validate(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn dag_returns_topological() {
        let dag = vec![("a", vec![]), ("b", vec!["a"]), ("c", vec!["a", "b"])];
        if let GraphVerdict::Ok { topological_order } = validate(&dag) {
            assert_eq!(topological_order.len(), 3);
            // 'a' must come before 'b' and 'c'.
            let pos_a = topological_order.iter().position(|x| x == "a").unwrap();
            let pos_b = topological_order.iter().position(|x| x == "b").unwrap();
            let pos_c = topological_order.iter().position(|x| x == "c").unwrap();
            assert!(pos_a < pos_b);
            assert!(pos_a < pos_c);
            assert!(pos_b < pos_c);
        }
    }

    #[test]
    fn cycle_detected() {
        let cycle = vec![("a", vec!["b"]), ("b", vec!["a"])];
        assert_eq!(validate(&cycle), GraphVerdict::Cycle);
    }

    #[test]
    fn unknown_ref_rejected() {
        let unknown = vec![("a", vec!["missing"])];
        assert!(matches!(
            validate(&unknown),
            GraphVerdict::UnknownReference { .. }
        ));
    }

    #[test]
    fn self_loop_detected() {
        let sl = vec![("a", vec!["a"])];
        assert!(matches!(validate(&sl), GraphVerdict::SelfLoop { .. }));
    }

    #[test]
    fn empty_graph_special() {
        assert_eq!(validate(&[]), GraphVerdict::EmptyGraph);
    }

    #[test]
    fn isolated_nodes_ok() {
        let iso = vec![("a", vec![]), ("b", vec![]), ("c", vec![])];
        if let GraphVerdict::Ok { topological_order } = validate(&iso) {
            assert_eq!(topological_order.len(), 3);
        }
    }

    #[test]
    fn long_chain_works() {
        let chain: Vec<(&str, Vec<&str>)> = vec![
            ("a", vec![]),
            ("b", vec!["a"]),
            ("c", vec!["b"]),
            ("d", vec!["c"]),
        ];
        assert!(matches!(validate(&chain), GraphVerdict::Ok { .. }));
    }

    #[test]
    fn diamond_works() {
        let diamond = vec![
            ("a", vec![]),
            ("b", vec!["a"]),
            ("c", vec!["a"]),
            ("d", vec!["b", "c"]),
        ];
        assert!(matches!(validate(&diamond), GraphVerdict::Ok { .. }));
    }

    #[test]
    fn three_node_cycle() {
        let c = vec![("a", vec!["b"]), ("b", vec!["c"]), ("c", vec!["a"])];
        assert_eq!(validate(&c), GraphVerdict::Cycle);
    }

    #[test]
    fn deterministic() {
        let dag = vec![("a", vec![]), ("b", vec!["a"])];
        let a = validate(&dag);
        let b = validate(&dag);
        assert_eq!(a, b);
    }
}
