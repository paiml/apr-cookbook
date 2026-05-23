//! # apr registry lineage — Cycle Detector
//!
//! Model lineage forms a DAG: child models point to parents (base, LoRA
//! adapter base, distillation teacher). Cycles indicate registry
//! corruption (e.g., re-publishing a model as a child of itself). This
//! recipe builds a cycle detector via Tarjan-style DFS coloring.
//!
//! Demonstrates the **REG.5** recipe for PMAT-114 (apr registry coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender REG-001 + Tarjan 1972 (DFS cycles)
//!
//! Run with: cargo run --example cli_registry_lineage_cycle_detector
//!
//! Added by PMAT-114 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashMap;

#[derive(Debug, PartialEq)]
pub enum LineageVerdict {
    Acyclic,
    CycleAt { node: String },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Color {
    White,
    Gray,
    Black,
}

pub fn detect_cycle(edges: &[(&str, &str)]) -> LineageVerdict {
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    for (child, parent) in edges {
        adj.entry(*child).or_default().push(*parent);
        adj.entry(*parent).or_default();
    }
    let mut color: HashMap<&str, Color> = adj.keys().map(|k| (*k, Color::White)).collect();
    for &node in adj.keys() {
        if color[node] == Color::White {
            if let Some(c) = dfs(node, &adj, &mut color) {
                return LineageVerdict::CycleAt { node: c.into() };
            }
        }
    }
    LineageVerdict::Acyclic
}

fn dfs<'a>(
    node: &'a str,
    adj: &HashMap<&'a str, Vec<&'a str>>,
    color: &mut HashMap<&'a str, Color>,
) -> Option<&'a str> {
    color.insert(node, Color::Gray);
    if let Some(neighbors) = adj.get(node) {
        for next in neighbors {
            match color.get(next).copied().unwrap_or(Color::White) {
                Color::Gray => return Some(next),
                Color::White => {
                    if let Some(c) = dfs(next, adj, color) {
                        return Some(c);
                    }
                }
                Color::Black => {}
            }
        }
    }
    color.insert(node, Color::Black);
    None
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_registry_lineage_cycle_detector")?;

    let acyclic = [("llama-3-8b-lora", "llama-3-8b"), ("llama-3-8b", "llama-3")];
    println!("acyclic → {:?}", detect_cycle(&acyclic));

    let cyclic = [("a", "b"), ("b", "c"), ("c", "a")];
    println!("cyclic → {:?}", detect_cycle(&cyclic));
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
    fn linear_chain_acyclic() {
        let edges = [("c", "b"), ("b", "a")];
        assert_eq!(detect_cycle(&edges), LineageVerdict::Acyclic);
    }

    #[test]
    fn diamond_dag_acyclic() {
        // d → b → a, d → c → a (multiple paths, but no cycle).
        let edges = [("d", "b"), ("d", "c"), ("b", "a"), ("c", "a")];
        assert_eq!(detect_cycle(&edges), LineageVerdict::Acyclic);
    }

    #[test]
    fn three_cycle_detected() {
        let edges = [("a", "b"), ("b", "c"), ("c", "a")];
        assert!(matches!(
            detect_cycle(&edges),
            LineageVerdict::CycleAt { .. }
        ));
    }

    #[test]
    fn self_loop_detected() {
        let edges = [("a", "a")];
        assert!(matches!(
            detect_cycle(&edges),
            LineageVerdict::CycleAt { .. }
        ));
    }

    #[test]
    fn empty_graph_acyclic() {
        let edges: [(&str, &str); 0] = [];
        assert_eq!(detect_cycle(&edges), LineageVerdict::Acyclic);
    }

    #[test]
    fn disconnected_components_acyclic() {
        // Two independent chains, no cycle.
        let edges = [("a", "b"), ("c", "d")];
        assert_eq!(detect_cycle(&edges), LineageVerdict::Acyclic);
    }

    #[test]
    fn disconnected_one_cycle_detected() {
        // First chain clean, second has a cycle.
        let edges = [("a", "b"), ("c", "d"), ("d", "c")];
        assert!(matches!(
            detect_cycle(&edges),
            LineageVerdict::CycleAt { .. }
        ));
    }

    #[test]
    fn long_chain_no_false_positive() {
        let edges: Vec<(&str, &str)> = vec![
            ("h", "g"),
            ("g", "f"),
            ("f", "e"),
            ("e", "d"),
            ("d", "c"),
            ("c", "b"),
            ("b", "a"),
        ];
        assert_eq!(detect_cycle(&edges), LineageVerdict::Acyclic);
    }
}
