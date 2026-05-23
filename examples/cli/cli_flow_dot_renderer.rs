//! # apr flow — Graphviz DOT Renderer
//!
//! `apr flow <FILE>` renders the data-flow graph to Graphviz DOT format
//! by default. This recipe builds the renderer as a pure function and
//! asserts the contract: every node has a unique id, every edge connects
//! two declared nodes, and the digraph header is present.
//!
//! Demonstrates the **FLOW.5** recipe for PMAT-100 (apr flow coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender FLOW-002 + Graphviz DOT spec
//!
//! Run with: cargo run --example cli_flow_dot_renderer
//!
//! Added by PMAT-100 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlowNode {
    pub id: String,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlowEdge {
    pub from: String,
    pub to: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlowGraph {
    pub nodes: Vec<FlowNode>,
    pub edges: Vec<FlowEdge>,
}

pub fn render_dot(g: &FlowGraph) -> String {
    use std::fmt::Write as _;
    let mut out = String::from("digraph apr_flow {\n");
    out.push_str("  rankdir=LR;\n");
    for n in &g.nodes {
        let _ = writeln!(out, "  \"{}\" [label=\"{}\"];", n.id, n.label);
    }
    for e in &g.edges {
        let _ = writeln!(out, "  \"{}\" -> \"{}\";", e.from, e.to);
    }
    out.push_str("}\n");
    out
}

pub fn validate_graph(g: &FlowGraph) -> Vec<String> {
    let mut issues = Vec::new();
    let mut ids = std::collections::HashSet::new();
    for n in &g.nodes {
        if !ids.insert(n.id.clone()) {
            issues.push(format!("duplicate node id: {}", n.id));
        }
    }
    for e in &g.edges {
        if !ids.contains(&e.from) {
            issues.push(format!("edge from missing node: {}", e.from));
        }
        if !ids.contains(&e.to) {
            issues.push(format!("edge to missing node: {}", e.to));
        }
    }
    issues
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_flow_dot_renderer")?;

    let g = FlowGraph {
        nodes: vec![
            FlowNode {
                id: "embed".into(),
                label: "embed_tokens".into(),
            },
            FlowNode {
                id: "attn0".into(),
                label: "layers.0.attn".into(),
            },
            FlowNode {
                id: "mlp0".into(),
                label: "layers.0.mlp".into(),
            },
            FlowNode {
                id: "head".into(),
                label: "lm_head".into(),
            },
        ],
        edges: vec![
            FlowEdge {
                from: "embed".into(),
                to: "attn0".into(),
            },
            FlowEdge {
                from: "attn0".into(),
                to: "mlp0".into(),
            },
            FlowEdge {
                from: "mlp0".into(),
                to: "head".into(),
            },
        ],
    };

    println!("{}", render_dot(&g));
    println!("validation: {:?}", validate_graph(&g));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> FlowGraph {
        FlowGraph {
            nodes: vec![
                FlowNode {
                    id: "a".into(),
                    label: "Alpha".into(),
                },
                FlowNode {
                    id: "b".into(),
                    label: "Beta".into(),
                },
            ],
            edges: vec![FlowEdge {
                from: "a".into(),
                to: "b".into(),
            }],
        }
    }

    #[test]
    fn renderer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn output_has_digraph_header() {
        let dot = render_dot(&sample_graph());
        assert!(dot.starts_with("digraph apr_flow"));
        assert!(dot.trim_end().ends_with('}'));
    }

    #[test]
    fn every_node_appears_in_dot() {
        let dot = render_dot(&sample_graph());
        assert!(dot.contains("\"a\""));
        assert!(dot.contains("\"b\""));
        assert!(dot.contains("Alpha"));
        assert!(dot.contains("Beta"));
    }

    #[test]
    fn edge_uses_directed_arrow() {
        let dot = render_dot(&sample_graph());
        assert!(dot.contains("\"a\" -> \"b\";"));
    }

    #[test]
    fn empty_graph_renders_valid_dot() {
        let g = FlowGraph {
            nodes: vec![],
            edges: vec![],
        };
        let dot = render_dot(&g);
        assert!(dot.starts_with("digraph"));
        assert!(dot.trim_end().ends_with('}'));
    }

    #[test]
    fn duplicate_node_ids_flagged() {
        let g = FlowGraph {
            nodes: vec![
                FlowNode {
                    id: "a".into(),
                    label: "A".into(),
                },
                FlowNode {
                    id: "a".into(),
                    label: "A2".into(),
                },
            ],
            edges: vec![],
        };
        let issues = validate_graph(&g);
        assert!(issues.iter().any(|s| s.contains("duplicate")));
    }

    #[test]
    fn edge_to_unknown_node_flagged() {
        let g = FlowGraph {
            nodes: vec![FlowNode {
                id: "a".into(),
                label: "A".into(),
            }],
            edges: vec![FlowEdge {
                from: "a".into(),
                to: "ghost".into(),
            }],
        };
        let issues = validate_graph(&g);
        assert!(issues.iter().any(|s| s.contains("ghost")));
    }

    #[test]
    fn rankdir_is_left_to_right() {
        // Convention for transformer-flow viz — readable as a horizontal pipeline.
        let dot = render_dot(&sample_graph());
        assert!(dot.contains("rankdir=LR"));
    }
}
