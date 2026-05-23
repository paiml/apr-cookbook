//! # apr pipeline plan — DAG Topological Sort
//!
//! `apr pipeline plan` produces an execution order for the resource DAG.
//! This recipe builds the topological-sort algorithm as a pure function
//! and asserts the contract: nodes ordered such that every edge (a → b)
//! has a before b in the output, cycles return an error, disconnected
//! components are concatenated.
//!
//! Demonstrates the **PIPELINE.13** recipe for PMAT-107 (apr pipeline coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PIPELINE-001 + Kahn (1962) topological sort
//!
//! Run with: cargo run --example cli_pipeline_dag_topological_sort
//!
//! Added by PMAT-107 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SortVerdict {
    Ok(Vec<String>),
    CycleDetected { remaining: Vec<String> },
}

pub fn topo_sort(edges: &[(String, String)]) -> SortVerdict {
    // Build in-degree map and adjacency.
    let mut in_degree: BTreeMap<String, u32> = BTreeMap::new();
    let mut adjacency: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (from, to) in edges {
        in_degree.entry(from.clone()).or_insert(0);
        *in_degree.entry(to.clone()).or_insert(0) += 1;
        adjacency
            .entry(from.clone())
            .or_default()
            .insert(to.clone());
    }

    // Kahn's: start with all in-degree-0 nodes.
    let mut queue: VecDeque<String> = in_degree
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(k, _)| k.clone())
        .collect();
    let mut sorted = Vec::new();

    while let Some(n) = queue.pop_front() {
        sorted.push(n.clone());
        if let Some(neighbors) = adjacency.get(&n) {
            for nb in neighbors {
                if let Some(d) = in_degree.get_mut(nb) {
                    *d -= 1;
                    if *d == 0 {
                        queue.push_back(nb.clone());
                    }
                }
            }
        }
    }

    if sorted.len() == in_degree.len() {
        SortVerdict::Ok(sorted)
    } else {
        let remaining: Vec<String> = in_degree
            .iter()
            .filter(|(k, _)| !sorted.contains(k))
            .map(|(k, _)| k.clone())
            .collect();
        SortVerdict::CycleDetected { remaining }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_pipeline_dag_topological_sort")?;

    let edges = vec![
        ("download".into(), "preprocess".into()),
        ("preprocess".into(), "train".into()),
        ("train".into(), "evaluate".into()),
        ("download".into(), "tokenize".into()),
        ("tokenize".into(), "train".into()),
    ];
    println!("dag: {edges:#?}");
    println!("sort: {:?}", topo_sort(&edges));

    let cycle = vec![
        ("a".into(), "b".into()),
        ("b".into(), "c".into()),
        ("c".into(), "a".into()),
    ];
    println!("\ncycle: {cycle:?}");
    println!("sort: {:?}", topo_sort(&cycle));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sort_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_edges_yield_empty_sort() {
        assert_eq!(topo_sort(&[]), SortVerdict::Ok(vec![]));
    }

    #[test]
    fn linear_chain_sorted() {
        let edges = vec![("a".into(), "b".into()), ("b".into(), "c".into())];
        if let SortVerdict::Ok(s) = topo_sort(&edges) {
            assert_eq!(s, vec!["a".to_string(), "b".into(), "c".into()]);
        } else {
            panic!("expected Ok");
        }
    }

    #[test]
    fn diamond_dag_sorts_with_root_first() {
        // a → b, a → c, b → d, c → d
        let edges = vec![
            ("a".into(), "b".into()),
            ("a".into(), "c".into()),
            ("b".into(), "d".into()),
            ("c".into(), "d".into()),
        ];
        if let SortVerdict::Ok(s) = topo_sort(&edges) {
            assert_eq!(s.first().map(String::as_str), Some("a"));
            assert_eq!(s.last().map(String::as_str), Some("d"));
        }
    }

    #[test]
    fn cycle_detected() {
        let edges = vec![
            ("a".into(), "b".into()),
            ("b".into(), "c".into()),
            ("c".into(), "a".into()),
        ];
        let v = topo_sort(&edges);
        assert!(matches!(v, SortVerdict::CycleDetected { .. }));
    }

    #[test]
    fn cycle_remaining_lists_unsortable_nodes() {
        let edges = vec![("a".into(), "b".into()), ("b".into(), "a".into())];
        if let SortVerdict::CycleDetected { remaining } = topo_sort(&edges) {
            assert_eq!(remaining.len(), 2);
        }
    }

    #[test]
    fn disconnected_components_both_included() {
        // Two unrelated chains: a→b and c→d
        let edges = vec![("a".into(), "b".into()), ("c".into(), "d".into())];
        if let SortVerdict::Ok(s) = topo_sort(&edges) {
            assert_eq!(s.len(), 4);
        }
    }
}
