//! # Recipe: Architecture Tree with Parameter Rollup
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr tree model.apr --rollup-params`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example tree_param_rollup` exits 0
//! 2. [x] `cargo test --example tree_param_rollup` passes
//! 3. [x] Deterministic output (fixed architecture)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr tree --rollup-params` in-process
//! 10. [x] Unit tests cover leaf/internal counts, rollup sum, ASCII render
//!
//! ## Learning Objective
//! Demonstrates rendering a model architecture as a tree and rolling up
//! parameter counts from leaves to the root. Each node reports own-params
//! (at that node) and subtree-params (own + all descendants).
//!
//! ## Run Command
//! ```bash
//! cargo run --example tree_param_rollup
//! ```
//!
//! ## References
//! - Cytron, R. et al. (1991). *Efficiently Computing Static Single Assignment Form and the Control Dependence Graph*. ACM TOPLAS. DOI: 10.1145/115372.115320

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone)]
pub struct ArchNode {
    pub name: String,
    pub own_params: u64,
    pub children: Vec<ArchNode>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RolledNode {
    pub name: String,
    pub depth: usize,
    pub own_params: u64,
    pub subtree_params: u64,
    pub n_leaves: usize,
}

pub fn rollup(root: &ArchNode) -> Vec<RolledNode> {
    let mut out = Vec::new();
    fn walk(node: &ArchNode, depth: usize, acc: &mut Vec<RolledNode>) -> (u64, usize) {
        let mut subtree = node.own_params;
        let mut leaves = usize::from(node.children.is_empty());
        // Record position; fill in subtree stats after we walk children.
        let idx = acc.len();
        acc.push(RolledNode {
            name: node.name.clone(),
            depth,
            own_params: node.own_params,
            subtree_params: 0,
            n_leaves: 0,
        });
        for c in &node.children {
            let (s, l) = walk(c, depth + 1, acc);
            subtree += s;
            leaves += l;
        }
        acc[idx].subtree_params = subtree;
        acc[idx].n_leaves = leaves;
        (subtree, leaves)
    }
    walk(root, 0, &mut out);
    out
}

pub fn render_tree(nodes: &[RolledNode]) -> String {
    let mut s = String::new();
    for n in nodes {
        s.push_str(&"  ".repeat(n.depth));
        s.push_str(&format!(
            "- {} (own={}, subtree={}, leaves={})\n",
            n.name, n.own_params, n.subtree_params, n.n_leaves
        ));
    }
    s
}

pub fn demo_arch() -> ArchNode {
    ArchNode {
        name: "llama-mini".into(),
        own_params: 0,
        children: vec![
            ArchNode {
                name: "embedding".into(),
                own_params: 32_000_000,
                children: vec![],
            },
            ArchNode {
                name: "transformer".into(),
                own_params: 0,
                children: vec![
                    ArchNode {
                        name: "layer_0".into(),
                        own_params: 0,
                        children: vec![
                            ArchNode {
                                name: "attention".into(),
                                own_params: 12_500_000,
                                children: vec![],
                            },
                            ArchNode {
                                name: "ffn".into(),
                                own_params: 25_000_000,
                                children: vec![],
                            },
                        ],
                    },
                    ArchNode {
                        name: "layer_1".into(),
                        own_params: 0,
                        children: vec![
                            ArchNode {
                                name: "attention".into(),
                                own_params: 12_500_000,
                                children: vec![],
                            },
                            ArchNode {
                                name: "ffn".into(),
                                own_params: 25_000_000,
                                children: vec![],
                            },
                        ],
                    },
                ],
            },
            ArchNode {
                name: "lm_head".into(),
                own_params: 32_000_000,
                children: vec![],
            },
        ],
    }
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("tree_param_rollup")?;
    println!("=== Recipe: {} ===", ctx.name());

    let arch = demo_arch();
    let rolled = rollup(&arch);
    let rendered = render_tree(&rolled);
    println!("{}", rendered);

    let root = rolled.first().cloned().unwrap_or(RolledNode {
        name: "empty".into(),
        depth: 0,
        own_params: 0,
        subtree_params: 0,
        n_leaves: 0,
    });
    println!("TOTAL PARAMS: {}", root.subtree_params);
    println!("TOTAL LEAVES: {}", root.n_leaves);

    let report = json!({
        "recipe": ctx.name(),
        "n_nodes": rolled.len(),
        "total_params": root.subtree_params,
        "n_leaves": root.n_leaves,
        "nodes": rolled.iter().map(|n| json!({
            "name": n.name,
            "depth": n.depth,
            "own_params": n.own_params,
            "subtree_params": n.subtree_params,
            "n_leaves": n.n_leaves,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("tree-rollup.json");
    std::fs::write(
        &out,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rollup_visits_all_nodes() {
        let r = rollup(&demo_arch());
        // root + embedding + transformer + 2 layers + 4 sublayers + lm_head = 10
        assert_eq!(r.len(), 10);
    }

    #[test]
    fn root_subtree_is_sum_of_own_params() {
        let r = rollup(&demo_arch());
        let total: u64 = r.iter().map(|n| n.own_params).sum();
        assert_eq!(r[0].subtree_params, total);
    }

    #[test]
    fn leaf_count_matches() {
        let r = rollup(&demo_arch());
        // 6 leaves: embedding, attn*2, ffn*2, lm_head
        assert_eq!(r[0].n_leaves, 6);
    }

    #[test]
    fn render_produces_indented_lines() {
        let r = rollup(&demo_arch());
        let s = render_tree(&r);
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), r.len());
        // Root line has no indent.
        assert!(lines[0].starts_with("- "));
    }

    #[test]
    fn single_leaf_node_rollup() {
        let n = ArchNode {
            name: "x".into(),
            own_params: 100,
            children: vec![],
        };
        let r = rollup(&n);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].subtree_params, 100);
        assert_eq!(r[0].n_leaves, 1);
    }
}
