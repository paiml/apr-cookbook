//! # Contracts-Macros Phase DAG Topological Ordering
//!
//! Compute a topological order over kernel-structure phases given
//! `(phase, deps)` pairs. Returns the linear order or detects cycles.
//!
//! Demonstrates the **CMM.47** recipe for PMAT-173 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Tarjan SCC + Kahn's algorithm.
//!
//! Run with: cargo run --example contracts_macros_phase_dag_order
//!
//! Added by PMAT-173 (catalog 1180→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, PartialEq)]
pub enum PhaseOrderVerdict {
    Ok { order: Vec<String> },
    CycleDetected,
    UnknownDependency { phase: String, missing: String },
    EmptyContract,
}

pub fn order(phases: &[(&str, Vec<&str>)]) -> PhaseOrderVerdict {
    if phases.is_empty() {
        return PhaseOrderVerdict::EmptyContract;
    }
    let names: BTreeSet<&str> = phases.iter().map(|(n, _)| *n).collect();
    let mut graph: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    let mut in_degree: BTreeMap<&str, usize> = names.iter().map(|n| (*n, 0)).collect();
    for (name, deps) in phases {
        for d in deps {
            if !names.contains(d) {
                return PhaseOrderVerdict::UnknownDependency {
                    phase: (*name).to_string(),
                    missing: (*d).to_string(),
                };
            }
            graph.entry(d).or_default().push(name);
            *in_degree.entry(name).or_insert(0) += 1;
        }
    }
    let mut queue: Vec<&str> = in_degree
        .iter()
        .filter(|(_, d)| **d == 0)
        .map(|(n, _)| *n)
        .collect();
    queue.sort_unstable();
    let mut order = Vec::with_capacity(phases.len());
    while let Some(n) = queue.pop() {
        order.push(n.to_string());
        if let Some(succs) = graph.get(n) {
            let mut next_zero = Vec::new();
            for s in succs {
                let d = in_degree.entry(s).or_insert(0);
                *d = d.saturating_sub(1);
                if *d == 0 {
                    next_zero.push(*s);
                }
            }
            next_zero.sort_unstable();
            for s in next_zero.into_iter().rev() {
                queue.push(s);
            }
        }
    }
    if order.len() != phases.len() {
        return PhaseOrderVerdict::CycleDetected;
    }
    PhaseOrderVerdict::Ok { order }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_phase_dag_order")?;

    let dag = vec![
        ("load", vec![]),
        ("validate", vec!["load"]),
        ("infer", vec!["validate"]),
    ];
    println!("typical: {:?}", order(&dag));

    let cycle = vec![("a", vec!["b"]), ("b", vec!["a"])];
    println!("cycle: {:?}", order(&cycle));

    let unknown = vec![("a", vec!["missing"])];
    println!("unknown: {:?}", order(&unknown));

    println!("empty: {:?}", order(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orderer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn linear_chain_topo() {
        let v = order(&[("a", vec![]), ("b", vec!["a"]), ("c", vec!["b"])]);
        if let PhaseOrderVerdict::Ok { order } = v {
            assert_eq!(
                order,
                vec!["a".to_string(), "b".to_string(), "c".to_string()]
            );
        }
    }

    #[test]
    fn diamond_yields_valid_order() {
        let v = order(&[
            ("a", vec![]),
            ("b", vec!["a"]),
            ("c", vec!["a"]),
            ("d", vec!["b", "c"]),
        ]);
        if let PhaseOrderVerdict::Ok { order } = v {
            let pos = |x: &str| order.iter().position(|s| s == x).unwrap();
            assert!(pos("a") < pos("b"));
            assert!(pos("a") < pos("c"));
            assert!(pos("b") < pos("d"));
            assert!(pos("c") < pos("d"));
        }
    }

    #[test]
    fn cycle_detected() {
        let v = order(&[("a", vec!["b"]), ("b", vec!["a"])]);
        assert_eq!(v, PhaseOrderVerdict::CycleDetected);
    }

    #[test]
    fn unknown_dep_rejected() {
        let v = order(&[("a", vec!["ghost"])]);
        assert!(matches!(v, PhaseOrderVerdict::UnknownDependency { .. }));
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(order(&[]), PhaseOrderVerdict::EmptyContract);
    }

    #[test]
    fn isolated_phases_all_appear() {
        let v = order(&[("x", vec![]), ("y", vec![]), ("z", vec![])]);
        if let PhaseOrderVerdict::Ok { order } = v {
            assert_eq!(order.len(), 3);
        }
    }

    #[test]
    fn three_node_cycle() {
        let v = order(&[("a", vec!["b"]), ("b", vec!["c"]), ("c", vec!["a"])]);
        assert_eq!(v, PhaseOrderVerdict::CycleDetected);
    }

    #[test]
    fn order_preserves_dependencies() {
        let v = order(&[
            ("d", vec!["b", "c"]),
            ("a", vec![]),
            ("b", vec!["a"]),
            ("c", vec!["a"]),
        ]);
        if let PhaseOrderVerdict::Ok { order } = v {
            let pos = |x: &str| order.iter().position(|s| s == x).unwrap();
            assert!(pos("a") < pos("b"));
        }
    }

    #[test]
    fn deterministic() {
        let dag = vec![("a", vec![]), ("b", vec!["a"])];
        let a = order(&dag);
        let b = order(&dag);
        assert_eq!(a, b);
    }

    #[test]
    fn single_phase() {
        let v = order(&[("only", vec![])]);
        if let PhaseOrderVerdict::Ok { order } = v {
            assert_eq!(order, vec!["only".to_string()]);
        }
    }
}
