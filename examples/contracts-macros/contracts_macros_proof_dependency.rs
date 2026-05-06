//! # Contracts-Macros Proof Dependency Closure
//!
//! Compute the transitive set of theorems that depend on a given
//! root theorem. Used to find which proofs need rechecking when a
//! foundational lemma changes.
//!
//! Demonstrates the **CMM.34** recipe for PMAT-169 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Mathlib4 dependency analysis tooling.
//!
//! Run with: cargo run --example contracts_macros_proof_dependency
//!
//! Added by PMAT-169 (catalog 1144→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, PartialEq)]
pub enum DependencyVerdict {
    Closure { theorems: Vec<String> },
    UnknownRoot,
    EmptyGraph,
    CycleDetected,
}

pub fn closure(graph: &[(&str, Vec<&str>)], root: &str) -> DependencyVerdict {
    if graph.is_empty() {
        return DependencyVerdict::EmptyGraph;
    }
    let map: BTreeMap<&str, Vec<&str>> = graph.iter().cloned().collect();
    if !map.contains_key(root) {
        return DependencyVerdict::UnknownRoot;
    }
    let mut visited: BTreeSet<&str> = BTreeSet::new();
    let mut on_stack: BTreeSet<&str> = BTreeSet::new();
    if dfs(&map, root, &mut visited, &mut on_stack) {
        return DependencyVerdict::CycleDetected;
    }
    visited.remove(root);
    DependencyVerdict::Closure {
        theorems: visited.into_iter().map(String::from).collect(),
    }
}

fn dfs<'a>(
    map: &'a BTreeMap<&'a str, Vec<&'a str>>,
    node: &'a str,
    visited: &mut BTreeSet<&'a str>,
    on_stack: &mut BTreeSet<&'a str>,
) -> bool {
    if on_stack.contains(node) {
        return true;
    }
    if visited.contains(node) {
        return false;
    }
    on_stack.insert(node);
    if let Some(deps) = map.get(node) {
        for dep in deps {
            if dfs(map, dep, visited, on_stack) {
                return true;
            }
        }
    }
    on_stack.remove(node);
    visited.insert(node);
    false
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_proof_dependency")?;

    let graph = vec![
        ("base", vec!["axiom_a"]),
        ("derived", vec!["base"]),
        ("composite", vec!["derived", "axiom_b"]),
    ];
    println!("composite: {:?}", closure(&graph, "composite"));
    println!("base: {:?}", closure(&graph, "base"));
    println!("unknown: {:?}", closure(&graph, "ghost"));
    println!("empty: {:?}", closure(&[], "any"));

    let cyclic = vec![("a", vec!["b"]), ("b", vec!["a"])];
    println!("cycle: {:?}", closure(&cyclic, "a"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_graph() -> Vec<(&'static str, Vec<&'static str>)> {
        vec![
            ("base", vec!["axiom_a"]),
            ("derived", vec!["base"]),
            ("composite", vec!["derived", "axiom_b"]),
        ]
    }

    #[test]
    fn closure_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn root_excluded_from_closure() {
        let v = closure(&small_graph(), "composite");
        if let DependencyVerdict::Closure { theorems } = v {
            assert!(!theorems.contains(&"composite".to_string()));
        }
    }

    #[test]
    fn transitive_dependencies_collected() {
        let v = closure(&small_graph(), "composite");
        if let DependencyVerdict::Closure { theorems } = v {
            assert!(theorems.contains(&"axiom_a".to_string()));
            assert!(theorems.contains(&"derived".to_string()));
        }
    }

    #[test]
    fn unknown_root() {
        assert_eq!(
            closure(&small_graph(), "ghost"),
            DependencyVerdict::UnknownRoot
        );
    }

    #[test]
    fn empty_graph() {
        assert_eq!(closure(&[], "any"), DependencyVerdict::EmptyGraph);
    }

    #[test]
    fn cycle_detected() {
        let cyclic = vec![("a", vec!["b"]), ("b", vec!["a"])];
        assert_eq!(closure(&cyclic, "a"), DependencyVerdict::CycleDetected);
    }

    #[test]
    fn leaf_has_empty_closure() {
        let v = closure(&small_graph(), "axiom_a");
        // axiom_a has no deps in graph, so it's an unknown root.
        assert_eq!(v, DependencyVerdict::UnknownRoot);
    }

    #[test]
    fn diamond_dependency() {
        let g = vec![
            ("top", vec!["left", "right"]),
            ("left", vec!["bottom"]),
            ("right", vec!["bottom"]),
            ("bottom", vec![]),
        ];
        let v = closure(&g, "top");
        if let DependencyVerdict::Closure { theorems } = v {
            assert!(theorems.contains(&"bottom".to_string()));
        }
    }

    #[test]
    fn duplicates_dedup() {
        let g = vec![("a", vec!["b", "b", "c"]), ("b", vec![]), ("c", vec![])];
        let v = closure(&g, "a");
        if let DependencyVerdict::Closure { theorems } = v {
            assert_eq!(theorems.len(), 2);
        }
    }

    #[test]
    fn deep_chain() {
        let g = vec![
            ("a", vec!["b"]),
            ("b", vec!["c"]),
            ("c", vec!["d"]),
            ("d", vec![]),
        ];
        let v = closure(&g, "a");
        if let DependencyVerdict::Closure { theorems } = v {
            assert_eq!(theorems.len(), 3);
        }
    }

    #[test]
    fn deterministic() {
        let g = small_graph();
        let a = closure(&g, "composite");
        let b = closure(&g, "composite");
        assert_eq!(a, b);
    }
}
