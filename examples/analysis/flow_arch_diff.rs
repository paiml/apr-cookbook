//! # Recipe: Flow — Cross-Architecture Diff (Encoder vs Encoder-Decoder)
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr flow --compare encoder.apr encoder_decoder.apr --edges`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example flow_arch_diff` exits 0
//! 2. [x] `cargo test --example flow_arch_diff` passes
//! 3. [x] Deterministic output (pure graph algorithms)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr flow --compare` in-process (no shell-out)
//! 10. [x] Unit tests cover nodes_added, edges_added, isomorphism
//!
//! ## Learning Objective
//! Builds the computation flow graph for two architectures (encoder-only vs
//! encoder-decoder) and reports added nodes / added edges — the canonical
//! "what structural changes shipped between architectures" view.
//!
//! ## Run Command
//! ```bash
//! cargo run --example flow_arch_diff
//! ```
//!
//! ## References
//! - Cytron, R. et al. (1991). *Efficiently Computing Static Single Assignment Form*. TOPLAS. DOI: 10.1145/115372.115320

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;
use std::collections::BTreeSet;

#[derive(Debug, Clone)]
struct FlowGraph {
    nodes: BTreeSet<String>,
    edges: BTreeSet<(String, String)>,
}

impl FlowGraph {
    fn new() -> Self {
        Self {
            nodes: BTreeSet::new(),
            edges: BTreeSet::new(),
        }
    }
    fn node(&mut self, n: &str) {
        self.nodes.insert(n.to_string());
    }
    fn edge(&mut self, from: &str, to: &str) {
        self.node(from);
        self.node(to);
        self.edges.insert((from.to_string(), to.to_string()));
    }
}

fn build_encoder_only() -> FlowGraph {
    let mut g = FlowGraph::new();
    g.edge("input", "embed");
    g.edge("embed", "enc_block_0");
    g.edge("enc_block_0", "enc_block_1");
    g.edge("enc_block_1", "pool");
    g.edge("pool", "head");
    g
}

fn build_encoder_decoder() -> FlowGraph {
    let mut g = FlowGraph::new();
    g.edge("input", "embed");
    g.edge("embed", "enc_block_0");
    g.edge("enc_block_0", "enc_block_1");
    g.edge("enc_block_1", "cross_attn");
    g.edge("dec_input", "dec_embed");
    g.edge("dec_embed", "dec_block_0");
    g.edge("dec_block_0", "cross_attn");
    g.edge("cross_attn", "dec_block_1");
    g.edge("dec_block_1", "head");
    g
}

#[derive(Debug, Clone)]
struct FlowDiff {
    nodes_added: Vec<String>,
    nodes_removed: Vec<String>,
    edges_added: Vec<(String, String)>,
    edges_removed: Vec<(String, String)>,
}

fn diff_flows(a: &FlowGraph, b: &FlowGraph) -> FlowDiff {
    let nodes_added: Vec<_> = b.nodes.difference(&a.nodes).cloned().collect();
    let nodes_removed: Vec<_> = a.nodes.difference(&b.nodes).cloned().collect();
    let edges_added: Vec<_> = b.edges.difference(&a.edges).cloned().collect();
    let edges_removed: Vec<_> = a.edges.difference(&b.edges).cloned().collect();
    FlowDiff {
        nodes_added,
        nodes_removed,
        edges_added,
        edges_removed,
    }
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("flow_arch_diff")?;
    println!("=== Recipe: {} ===", ctx.name());

    let a = build_encoder_only();
    let b = build_encoder_decoder();

    println!(
        "Encoder-only:       {} nodes, {} edges",
        a.nodes.len(),
        a.edges.len()
    );
    println!(
        "Encoder-decoder:    {} nodes, {} edges",
        b.nodes.len(),
        b.edges.len()
    );

    let d = diff_flows(&a, &b);
    println!("\n--- Flow diff ---");
    println!("Nodes added:   {:?}", d.nodes_added);
    println!("Nodes removed: {:?}", d.nodes_removed);
    println!("Edges added:   {:?}", d.edges_added);
    println!("Edges removed: {:?}", d.edges_removed);

    let report = json!({
        "recipe": ctx.name(),
        "encoder_only": {
            "n_nodes": a.nodes.len(),
            "n_edges": a.edges.len(),
        },
        "encoder_decoder": {
            "n_nodes": b.nodes.len(),
            "n_edges": b.edges.len(),
        },
        "diff": {
            "nodes_added": d.nodes_added,
            "nodes_removed": d.nodes_removed,
            "edges_added": d.edges_added.iter().map(|(f, t)| json!([f, t])).collect::<Vec<_>>(),
            "edges_removed": d.edges_removed.iter().map(|(f, t)| json!([f, t])).collect::<Vec<_>>(),
        },
    });
    let out = ctx.path("flow-arch-diff.json");
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out, bytes)?;

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_diff_on_equal_graphs() {
        let a = build_encoder_only();
        let b = build_encoder_only();
        let d = diff_flows(&a, &b);
        assert!(d.nodes_added.is_empty());
        assert!(d.nodes_removed.is_empty());
        assert!(d.edges_added.is_empty());
        assert!(d.edges_removed.is_empty());
    }

    #[test]
    fn nodes_added_includes_decoder_stack() {
        let a = build_encoder_only();
        let b = build_encoder_decoder();
        let d = diff_flows(&a, &b);
        assert!(d.nodes_added.contains(&"dec_block_0".to_string()));
        assert!(d.nodes_added.contains(&"cross_attn".to_string()));
    }

    #[test]
    fn edges_removed_includes_encoder_direct_to_head() {
        let a = build_encoder_only();
        let b = build_encoder_decoder();
        let d = diff_flows(&a, &b);
        assert!(d
            .edges_removed
            .iter()
            .any(|(f, t)| f == "enc_block_1" && t == "pool"));
    }

    #[test]
    fn diff_is_antisymmetric() {
        let a = build_encoder_only();
        let b = build_encoder_decoder();
        let ab = diff_flows(&a, &b);
        let ba = diff_flows(&b, &a);
        assert_eq!(ab.nodes_added, ba.nodes_removed);
        assert_eq!(ab.edges_added, ba.edges_removed);
    }
}
