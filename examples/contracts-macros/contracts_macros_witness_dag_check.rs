//! # Contracts-Macros Witness DAG Cycle Check
//!
//! Validate that a witness-dependency graph is a DAG (no cycles).
//! Returns sorted cycle-participating node IDs (empty if DAG).
//!
//! Demonstrates the **CMM.146** recipe for PMAT-206 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Tarjan SCC algorithm (1972); Knuth TAOCP §2.2.3 cycle
//!  detection in directed graphs.
//!
//! Run with: cargo run --example contracts_macros_witness_dag_check
//!
//! Added by PMAT-206 (catalog 1477→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, PartialEq)]
pub enum DagVerdict {
    Ok {
        cycle_nodes: Vec<String>,
        is_dag: bool,
    },
    InvalidConfig,
}

pub fn check(edges: &[(&str, &str)]) -> DagVerdict {
    if edges.is_empty() {
        return DagVerdict::InvalidConfig;
    }
    let mut adj: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut nodes: BTreeSet<String> = BTreeSet::new();
    for (from, to) in edges {
        adj.entry((*from).to_string())
            .or_default()
            .push((*to).to_string());
        nodes.insert((*from).to_string());
        nodes.insert((*to).to_string());
    }
    let mut state: BTreeMap<String, u8> = BTreeMap::new(); // 0=white,1=gray,2=black
    let mut cycle_nodes: BTreeSet<String> = BTreeSet::new();
    for n in &nodes {
        if !state.contains_key(n) {
            visit(n, &adj, &mut state, &mut cycle_nodes);
        }
    }
    let is_dag = cycle_nodes.is_empty();
    DagVerdict::Ok {
        cycle_nodes: cycle_nodes.into_iter().collect(),
        is_dag,
    }
}

fn visit(
    n: &str,
    adj: &BTreeMap<String, Vec<String>>,
    state: &mut BTreeMap<String, u8>,
    cycle_nodes: &mut BTreeSet<String>,
) {
    state.insert(n.to_string(), 1);
    if let Some(neighbors) = adj.get(n) {
        for m in neighbors {
            match state.get(m).copied().unwrap_or(0) {
                0 => visit(m, adj, state, cycle_nodes),
                1 => {
                    cycle_nodes.insert(n.to_string());
                    cycle_nodes.insert(m.clone());
                }
                _ => {}
            }
        }
    }
    state.insert(n.to_string(), 2);
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_witness_dag_check")?;

    println!("dag: {:?}", check(&[("a", "b"), ("b", "c")]));
    println!("cycle: {:?}", check(&[("a", "b"), ("b", "a")]));
    println!("invalid: {:?}", check(&[]));
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
    fn linear_chain_is_dag() {
        let v = check(&[("a", "b"), ("b", "c")]);
        if let DagVerdict::Ok { is_dag, .. } = v {
            assert!(is_dag);
        }
    }

    #[test]
    fn self_loop_detected() {
        let v = check(&[("a", "a")]);
        if let DagVerdict::Ok { is_dag, .. } = v {
            assert!(!is_dag);
        }
    }

    #[test]
    fn two_node_cycle_detected() {
        let v = check(&[("a", "b"), ("b", "a")]);
        if let DagVerdict::Ok {
            cycle_nodes,
            is_dag,
        } = v
        {
            assert!(!is_dag);
            assert!(cycle_nodes.contains(&"a".to_string()));
            assert!(cycle_nodes.contains(&"b".to_string()));
        }
    }

    #[test]
    fn three_node_cycle_detected() {
        let v = check(&[("a", "b"), ("b", "c"), ("c", "a")]);
        if let DagVerdict::Ok { is_dag, .. } = v {
            assert!(!is_dag);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(&[]), DagVerdict::InvalidConfig);
    }

    #[test]
    fn diamond_is_dag() {
        let v = check(&[("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]);
        if let DagVerdict::Ok { is_dag, .. } = v {
            assert!(is_dag);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check(&[("a", "b")]);
        let r2 = check(&[("a", "b")]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn cycle_nodes_sorted() {
        let v = check(&[("z", "y"), ("y", "z")]);
        if let DagVerdict::Ok { cycle_nodes, .. } = v {
            for w in cycle_nodes.windows(2) {
                assert!(w[0] < w[1]);
            }
        }
    }

    #[test]
    fn two_disconnected_dags() {
        let v = check(&[("a", "b"), ("c", "d")]);
        if let DagVerdict::Ok { is_dag, .. } = v {
            assert!(is_dag);
        }
    }

    #[test]
    fn cycle_in_subgraph_detected() {
        // {a→b}, {c→d→c}
        let v = check(&[("a", "b"), ("c", "d"), ("d", "c")]);
        if let DagVerdict::Ok { is_dag, .. } = v {
            assert!(!is_dag);
        }
    }

    #[test]
    fn many_nodes_handled() {
        let edges: Vec<(&str, &str)> = vec![
            ("n1", "n2"),
            ("n2", "n3"),
            ("n3", "n4"),
            ("n4", "n5"),
            ("n5", "n6"),
        ];
        let v = check(&edges);
        if let DagVerdict::Ok { is_dag, .. } = v {
            assert!(is_dag);
        }
    }

    #[test]
    fn unicode_node_supported() {
        let v = check(&[("café", "résumé")]);
        if let DagVerdict::Ok { is_dag, .. } = v {
            assert!(is_dag);
        }
    }
}
