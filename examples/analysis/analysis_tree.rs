//! # Architecture Visualization as ASCII Tree
//!
//! Renders model tensor hierarchy as an ASCII tree with parameter counts,
//! grouping tensors by their dotted name paths.
//!
//! ## CLI equivalent
//! ```bash
//! apr tree model.apr
//! ```
//!
//! ## What this demonstrates
//! - Hierarchical grouping of flat tensor names
//! - Recursive tree construction and rendering
//! - Parameter count aggregation at each level
//! - Box-drawing character output

use apr_cookbook::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TreeNode {
    name: String,
    children: Vec<TreeNode>,
    params: usize,
    dtype: String,
}

impl TreeNode {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            children: Vec::new(),
            params: 0,
            dtype: String::new(),
        }
    }

    #[allow(dead_code)]
    fn new_leaf(name: &str, params: usize, dtype: &str) -> Self {
        Self {
            name: name.to_string(),
            children: Vec::new(),
            params,
            dtype: dtype.to_string(),
        }
    }

    /// Total parameters including all descendants.
    fn total_params(&self) -> usize {
        if self.children.is_empty() {
            self.params
        } else {
            self.children.iter().map(TreeNode::total_params).sum()
        }
    }

    /// Find or create a child with the given name.
    fn get_or_create_child(&mut self, name: &str) -> &mut TreeNode {
        if let Some(pos) = self.children.iter().position(|c| c.name == name) {
            &mut self.children[pos]
        } else {
            self.children.push(TreeNode::new(name));
            self.children.last_mut().unwrap()
        }
    }
}

// ---------------------------------------------------------------------------
// Tree construction
// ---------------------------------------------------------------------------

/// Build a hierarchical tree from flat tensor names and shapes.
///
/// Names are split by "." to create the hierarchy.
/// E.g., "model.layers.0.attn.q_proj.weight" becomes:
///   model -> layers -> 0 -> attn -> q_proj -> weight (leaf)
fn build_tree(tensors: &[(String, Vec<usize>)]) -> TreeNode {
    let mut root = TreeNode::new("model");

    for (name, shape) in tensors {
        let params: usize = shape.iter().product();
        let parts: Vec<&str> = name.split('.').collect();

        let mut current = &mut root;
        for (i, &part) in parts.iter().enumerate() {
            if i == parts.len() - 1 {
                // Leaf node: set params
                let child = current.get_or_create_child(part);
                child.params = params;
                child.dtype = "f32".to_string();
            } else {
                current = current.get_or_create_child(part);
            }
        }
    }

    root
}

/// Format a parameter count in human-readable form.
fn format_params(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1e3)
    } else {
        format!("{}", n)
    }
}

/// Format a shape as a human-readable string.
fn format_shape(shape: &[usize]) -> String {
    let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
    dims.join("\u{00d7}") // multiplication sign
}

// ---------------------------------------------------------------------------
// Tree rendering
// ---------------------------------------------------------------------------

/// Render the tree as an ASCII string with box-drawing characters.
fn render_tree(node: &TreeNode, prefix: &str, is_last: bool, is_root: bool) -> String {
    let mut output = String::new();

    // Current node line
    let connector = if is_root {
        ""
    } else if is_last {
        "\u{2514}\u{2500}\u{2500} " // └──
    } else {
        "\u{251c}\u{2500}\u{2500} " // ├──
    };

    let total = node.total_params();
    let param_str = if total > 0 {
        format!(" ({})", format_params(total))
    } else {
        String::new()
    };

    output.push_str(&format!(
        "{}{}{}{}\n",
        prefix, connector, node.name, param_str
    ));

    // Children
    let child_prefix = if is_root {
        String::new()
    } else if is_last {
        format!("{}    ", prefix)
    } else {
        format!("{}\u{2502}   ", prefix) // │
    };

    for (i, child) in node.children.iter().enumerate() {
        let child_is_last = i == node.children.len() - 1;
        output.push_str(&render_tree(child, &child_prefix, child_is_last, false));
    }

    output
}

/// Top-level render function.
fn render(root: &TreeNode) -> String {
    render_tree(root, "", true, true)
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("analysis_tree")?;

    // ── Section 1: Define a transformer-like model ──────────────────────
    println!("=== Model Architecture Tree ===\n");

    let tensors: Vec<(String, Vec<usize>)> = vec![
        ("embed.weight".into(), vec![768, 50257]),
        ("layers.0.attn.q_proj.weight".into(), vec![768, 768]),
        ("layers.0.attn.k_proj.weight".into(), vec![768, 768]),
        ("layers.0.attn.v_proj.weight".into(), vec![768, 768]),
        ("layers.0.attn.o_proj.weight".into(), vec![768, 768]),
        ("layers.0.mlp.gate.weight".into(), vec![768, 3072]),
        ("layers.0.mlp.up.weight".into(), vec![768, 3072]),
        ("layers.0.mlp.down.weight".into(), vec![3072, 768]),
        ("layers.1.attn.q_proj.weight".into(), vec![768, 768]),
        ("layers.1.attn.k_proj.weight".into(), vec![768, 768]),
        ("layers.1.attn.v_proj.weight".into(), vec![768, 768]),
        ("layers.1.attn.o_proj.weight".into(), vec![768, 768]),
        ("layers.1.mlp.gate.weight".into(), vec![768, 3072]),
        ("layers.1.mlp.up.weight".into(), vec![768, 3072]),
        ("layers.1.mlp.down.weight".into(), vec![3072, 768]),
        ("norm.weight".into(), vec![768]),
        ("lm_head.weight".into(), vec![50257, 768]),
    ];

    // ── Section 2: Flat tensor list ─────────────────────────────────────
    println!("--- Flat Tensor List ---");
    let mut total_params = 0usize;
    for (name, shape) in &tensors {
        let p: usize = shape.iter().product();
        total_params += p;
        println!(
            "  {} : {} ({})",
            name,
            format_shape(shape),
            format_params(p)
        );
    }
    println!(
        "Total: {} params ({})\n",
        total_params,
        format_params(total_params)
    );

    // ── Section 3: Build hierarchical tree ──────────────────────────────
    println!("--- Hierarchical Tree ---");
    let root = build_tree(&tensors);
    let tree_str = render(&root);
    println!("{}", tree_str);

    // ── Section 4: Parameter count at each level ────────────────────────
    println!("--- Parameter Aggregation ---");
    for child in &root.children {
        println!(
            "  {}: {} params ({})",
            child.name,
            child.total_params(),
            format_params(child.total_params())
        );
    }

    // Fingerprint
    let mut hasher = DefaultHasher::new();
    total_params.hash(&mut hasher);
    root.children.len().hash(&mut hasher);
    println!("\nTree fingerprint: {:016x}", hasher.finish());

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tensors() -> Vec<(String, Vec<usize>)> {
        vec![
            ("embed.weight".into(), vec![768, 50257]),
            ("layers.0.attn.q_proj.weight".into(), vec![768, 768]),
            ("layers.0.mlp.gate.weight".into(), vec![768, 3072]),
            ("layers.1.attn.q_proj.weight".into(), vec![768, 768]),
            ("norm.weight".into(), vec![768]),
            ("lm_head.weight".into(), vec![50257, 768]),
        ]
    }

    #[test]
    fn test_tree_structure_correct() {
        let root = build_tree(&sample_tensors());
        assert_eq!(root.name, "model");
        let child_names: Vec<&str> = root.children.iter().map(|c| c.name.as_str()).collect();
        assert!(child_names.contains(&"embed"));
        assert!(child_names.contains(&"layers"));
        assert!(child_names.contains(&"norm"));
        assert!(child_names.contains(&"lm_head"));
    }

    #[test]
    fn test_param_aggregation() {
        let root = build_tree(&sample_tensors());
        let total = root.total_params();
        // embed: 768*50257 + layers.0.attn.q_proj: 768*768 + layers.0.mlp.gate: 768*3072
        // + layers.1.attn.q_proj: 768*768 + norm: 768 + lm_head: 50257*768
        let expected = 768 * 50257 + 768 * 768 + 768 * 3072 + 768 * 768 + 768 + 50257 * 768;
        assert_eq!(total, expected);
    }

    #[test]
    fn test_render_non_empty() {
        let root = build_tree(&sample_tensors());
        let rendered = render(&root);
        assert!(!rendered.is_empty());
        assert!(rendered.contains("model"));
        assert!(rendered.contains("embed"));
        assert!(rendered.contains("layers"));
    }

    #[test]
    fn test_single_tensor_tree() {
        let tensors = vec![("weight".into(), vec![10, 20])];
        let root = build_tree(&tensors);
        assert_eq!(root.children.len(), 1);
        assert_eq!(root.children[0].name, "weight");
        assert_eq!(root.total_params(), 200);
    }

    #[test]
    fn test_deeply_nested() {
        let tensors = vec![("a.b.c.d.e.f.weight".into(), vec![4, 4])];
        let root = build_tree(&tensors);
        // Traverse down: model -> a -> b -> c -> d -> e -> f -> weight
        let a = &root.children[0];
        assert_eq!(a.name, "a");
        let b = &a.children[0];
        assert_eq!(b.name, "b");
        let c = &b.children[0];
        assert_eq!(c.name, "c");
        assert_eq!(root.total_params(), 16);
    }

    #[test]
    fn test_empty_tensors() {
        let tensors: Vec<(String, Vec<usize>)> = vec![];
        let root = build_tree(&tensors);
        assert_eq!(root.children.len(), 0);
        assert_eq!(root.total_params(), 0);
    }

    #[test]
    fn test_format_params_human_readable() {
        assert_eq!(format_params(500), "500");
        assert_eq!(format_params(1500), "1.5K");
        assert_eq!(format_params(2_500_000), "2.5M");
        assert_eq!(format_params(1_500_000_000), "1.5B");
    }

    #[test]
    fn test_render_contains_box_drawing() {
        let tensors = vec![
            ("a.x".into(), vec![10]),
            ("a.y".into(), vec![20]),
            ("b.z".into(), vec![30]),
        ];
        let root = build_tree(&tensors);
        let rendered = render(&root);
        // Should contain box-drawing characters
        assert!(
            rendered.contains('\u{251c}') || rendered.contains('\u{2514}'),
            "rendered tree should contain box-drawing characters"
        );
    }

    #[test]
    fn test_siblings_share_parent() {
        let tensors = vec![
            ("layer.weight".into(), vec![10, 20]),
            ("layer.bias".into(), vec![10]),
        ];
        let root = build_tree(&tensors);
        assert_eq!(root.children.len(), 1);
        let layer = &root.children[0];
        assert_eq!(layer.name, "layer");
        assert_eq!(layer.children.len(), 2);
    }

    #[test]
    fn test_format_shape() {
        assert_eq!(format_shape(&[768, 768]), "768\u{00d7}768");
        assert_eq!(format_shape(&[3, 224, 224]), "3\u{00d7}224\u{00d7}224");
    }
}
