//! # Contracts-Macros Recipe Dependency Height
//!
//! Compute max height of the recipe dependency tree (longest
//! parent → leaf chain). Useful for refactoring deeply-nested
//! recipes.
//!
//! Demonstrates the **CMM.111** recipe for PMAT-194 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: dependency tree depth analysis (CMU SEI Software
//!  Architecture Pattern catalog).
//!
//! Run with: cargo run --example contracts_macros_recipe_dependency_height
//!
//! Added by PMAT-194 (catalog 1369→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum HeightVerdict {
    Ok {
        max_height: u32,
        deepest_root: String,
    },
    InvalidConfig,
}

pub fn measure(edges: &[(&str, &str)]) -> HeightVerdict {
    if edges.is_empty() {
        return HeightVerdict::InvalidConfig;
    }
    let mut children: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut all_nodes: BTreeSet<String> = BTreeSet::new();
    let mut child_set: BTreeSet<String> = BTreeSet::new();
    for (parent, child) in edges {
        children
            .entry((*parent).to_string())
            .or_default()
            .push((*child).to_string());
        all_nodes.insert((*parent).to_string());
        all_nodes.insert((*child).to_string());
        child_set.insert((*child).to_string());
    }
    let roots: Vec<String> = all_nodes
        .iter()
        .filter(|n| !child_set.contains(*n))
        .cloned()
        .collect();
    let mut max_height = 0u32;
    let mut deepest_root = String::new();
    for root in &roots {
        let h = depth(root, &children, 100);
        if h > max_height {
            max_height = h;
            deepest_root = root.clone();
        }
    }
    if deepest_root.is_empty() && !roots.is_empty() {
        deepest_root = roots[0].clone();
    }
    HeightVerdict::Ok {
        max_height,
        deepest_root,
    }
}

fn depth(node: &str, children: &BTreeMap<String, Vec<String>>, cap: u32) -> u32 {
    if cap == 0 {
        return 0;
    }
    match children.get(node) {
        None => 0,
        Some(child_list) if child_list.is_empty() => 0,
        Some(child_list) => {
            let mut max_child = 0;
            for c in child_list {
                let d = 1 + depth(c, children, cap - 1);
                if d > max_child {
                    max_child = d;
                }
            }
            max_child
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_dependency_height")?;

    let edges = [("root", "a"), ("a", "b"), ("b", "leaf")];
    println!("audit: {:?}", measure(&edges));
    println!("invalid: {:?}", measure(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measurer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn linear_chain_height() {
        let edges = [("a", "b"), ("b", "c"), ("c", "d")];
        let v = measure(&edges);
        if let HeightVerdict::Ok { max_height, .. } = v {
            assert_eq!(max_height, 3);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(measure(&[]), HeightVerdict::InvalidConfig);
    }

    #[test]
    fn single_edge_height_one() {
        let edges = [("root", "leaf")];
        let v = measure(&edges);
        if let HeightVerdict::Ok { max_height, .. } = v {
            assert_eq!(max_height, 1);
        }
    }

    #[test]
    fn deepest_root_identified() {
        let edges = [("root", "a"), ("a", "b")];
        let v = measure(&edges);
        if let HeightVerdict::Ok { deepest_root, .. } = v {
            assert_eq!(deepest_root, "root");
        }
    }

    #[test]
    fn diamond_height_via_either_path() {
        let edges = [("root", "a"), ("root", "b"), ("a", "leaf"), ("b", "leaf")];
        let v = measure(&edges);
        if let HeightVerdict::Ok { max_height, .. } = v {
            assert_eq!(max_height, 2);
        }
    }

    #[test]
    fn deterministic() {
        let edges = [("a", "b")];
        let r1 = measure(&edges);
        let r2 = measure(&edges);
        assert_eq!(r1, r2);
    }

    #[test]
    fn picks_longer_branch_for_deepest() {
        let edges = [
            ("root", "shallow"),
            ("root", "deep1"),
            ("deep1", "deep2"),
            ("deep2", "deep3"),
        ];
        let v = measure(&edges);
        if let HeightVerdict::Ok { max_height, .. } = v {
            assert_eq!(max_height, 3);
        }
    }

    #[test]
    fn multiple_roots_pick_deepest() {
        let edges = [("r1", "a"), ("r2", "b"), ("b", "c"), ("c", "d")];
        let v = measure(&edges);
        if let HeightVerdict::Ok { deepest_root, .. } = v {
            assert_eq!(deepest_root, "r2");
        }
    }

    #[test]
    fn cycle_capped_by_recursion_limit() {
        let edges = [("a", "b"), ("b", "a")];
        let v = measure(&edges);
        if let HeightVerdict::Ok { max_height, .. } = v {
            assert!(max_height <= 100);
        }
    }

    #[test]
    fn root_with_two_leaves() {
        let edges = [("r", "l1"), ("r", "l2")];
        let v = measure(&edges);
        if let HeightVerdict::Ok { max_height, .. } = v {
            assert_eq!(max_height, 1);
        }
    }
}
