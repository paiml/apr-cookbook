//! # TUI Tree-View Collapse State
//!
//! Compute visible nodes given a parent-child tree and a set of
//! collapsed parents. Returns the flat visible-row list with depth.
//!
//! Demonstrates the **TUI.20** recipe for PMAT-166 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS NSOutlineView collapse semantics.
//!
//! Run with: cargo run --example tui_tree_view_collapse
//!
//! Added by PMAT-166 (catalog 1117→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeNode {
    pub id: String,
    pub parent: Option<String>,
    pub depth: u32,
}

#[derive(Debug, PartialEq)]
pub enum TreeVerdict {
    Ok { visible: Vec<(String, u32)> },
    EmptyTree,
}

pub fn compute_visible(nodes: &[TreeNode], collapsed: &[&str]) -> TreeVerdict {
    if nodes.is_empty() {
        return TreeVerdict::EmptyTree;
    }
    let collapsed_set: BTreeSet<&str> = collapsed.iter().copied().collect();
    let collapsed_ids: BTreeSet<String> = collapsed_set.iter().map(|s| (*s).to_string()).collect();
    let mut hidden: BTreeSet<String> = BTreeSet::new();
    for node in nodes {
        if let Some(parent) = &node.parent {
            if collapsed_ids.contains(parent) || hidden.contains(parent) {
                hidden.insert(node.id.clone());
            }
        }
    }
    let visible: Vec<(String, u32)> = nodes
        .iter()
        .filter(|n| !hidden.contains(&n.id))
        .map(|n| (n.id.clone(), n.depth))
        .collect();
    TreeVerdict::Ok { visible }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_tree_view_collapse")?;

    let nodes = vec![
        TreeNode {
            id: "root".to_string(),
            parent: None,
            depth: 0,
        },
        TreeNode {
            id: "a".to_string(),
            parent: Some("root".to_string()),
            depth: 1,
        },
        TreeNode {
            id: "a.x".to_string(),
            parent: Some("a".to_string()),
            depth: 2,
        },
        TreeNode {
            id: "b".to_string(),
            parent: Some("root".to_string()),
            depth: 1,
        },
    ];
    println!("none collapsed: {:?}", compute_visible(&nodes, &[]));
    println!("a collapsed: {:?}", compute_visible(&nodes, &["a"]));
    println!("root collapsed: {:?}", compute_visible(&nodes, &["root"]));
    println!("empty: {:?}", compute_visible(&[], &[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree() -> Vec<TreeNode> {
        vec![
            TreeNode {
                id: "root".to_string(),
                parent: None,
                depth: 0,
            },
            TreeNode {
                id: "a".to_string(),
                parent: Some("root".to_string()),
                depth: 1,
            },
            TreeNode {
                id: "a.x".to_string(),
                parent: Some("a".to_string()),
                depth: 2,
            },
            TreeNode {
                id: "b".to_string(),
                parent: Some("root".to_string()),
                depth: 1,
            },
        ]
    }

    #[test]
    fn computer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn nothing_collapsed_all_visible() {
        let v = compute_visible(&tree(), &[]);
        if let TreeVerdict::Ok { visible } = v {
            assert_eq!(visible.len(), 4);
        }
    }

    #[test]
    fn collapsed_node_hides_descendants() {
        let v = compute_visible(&tree(), &["a"]);
        if let TreeVerdict::Ok { visible } = v {
            // root, a, b visible; a.x hidden.
            assert_eq!(visible.len(), 3);
            assert!(visible.iter().all(|(id, _)| id != "a.x"));
        }
    }

    #[test]
    fn collapsed_root_hides_all_below() {
        let v = compute_visible(&tree(), &["root"]);
        if let TreeVerdict::Ok { visible } = v {
            // Only root visible.
            assert_eq!(visible.len(), 1);
            assert_eq!(visible[0].0, "root");
        }
    }

    #[test]
    fn empty_tree_rejected() {
        assert_eq!(compute_visible(&[], &[]), TreeVerdict::EmptyTree);
    }

    #[test]
    fn unknown_collapsed_id_ignored() {
        let v = compute_visible(&tree(), &["nonexistent"]);
        if let TreeVerdict::Ok { visible } = v {
            assert_eq!(visible.len(), 4);
        }
    }

    #[test]
    fn depth_preserved() {
        let v = compute_visible(&tree(), &[]);
        if let TreeVerdict::Ok { visible } = v {
            let depths: Vec<u32> = visible.iter().map(|(_, d)| *d).collect();
            assert_eq!(depths, vec![0, 1, 2, 1]);
        }
    }

    #[test]
    fn multiple_collapsed() {
        // Wider tree.
        let nodes = vec![
            TreeNode {
                id: "root".to_string(),
                parent: None,
                depth: 0,
            },
            TreeNode {
                id: "a".to_string(),
                parent: Some("root".to_string()),
                depth: 1,
            },
            TreeNode {
                id: "a.x".to_string(),
                parent: Some("a".to_string()),
                depth: 2,
            },
            TreeNode {
                id: "b".to_string(),
                parent: Some("root".to_string()),
                depth: 1,
            },
            TreeNode {
                id: "b.y".to_string(),
                parent: Some("b".to_string()),
                depth: 2,
            },
        ];
        let v = compute_visible(&nodes, &["a", "b"]);
        if let TreeVerdict::Ok { visible } = v {
            assert_eq!(visible.len(), 3);
        }
    }

    #[test]
    fn deeply_nested_collapse() {
        let nodes = vec![
            TreeNode {
                id: "a".to_string(),
                parent: None,
                depth: 0,
            },
            TreeNode {
                id: "b".to_string(),
                parent: Some("a".to_string()),
                depth: 1,
            },
            TreeNode {
                id: "c".to_string(),
                parent: Some("b".to_string()),
                depth: 2,
            },
            TreeNode {
                id: "d".to_string(),
                parent: Some("c".to_string()),
                depth: 3,
            },
        ];
        let v = compute_visible(&nodes, &["a"]);
        if let TreeVerdict::Ok { visible } = v {
            assert_eq!(visible.len(), 1);
        }
    }

    #[test]
    fn root_node_no_parent() {
        let nodes = vec![TreeNode {
            id: "only".to_string(),
            parent: None,
            depth: 0,
        }];
        let v = compute_visible(&nodes, &[]);
        if let TreeVerdict::Ok { visible } = v {
            assert_eq!(visible.len(), 1);
        }
    }

    #[test]
    fn deterministic() {
        let a = compute_visible(&tree(), &["a"]);
        let b = compute_visible(&tree(), &["a"]);
        assert_eq!(a, b);
    }
}
