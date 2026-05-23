//! # Contracts-Macros Obligation Priority Inheritance
//!
//! Propagate parent obligation priority down to children when child's
//! own priority is lower. Reports final per-obligation priorities
//! and which were promoted.
//!
//! Demonstrates the **CMM.121** recipe for PMAT-198 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: priority inheritance protocol (Sha, Rajkumar, Lehoczky
//!  IEEE TC 1990); Eiffel covariant downcasting.
//!
//! Run with: cargo run --example contracts_macros_obligation_priority_inheritance
//!
//! Added by PMAT-198 (catalog 1405→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum InheritanceVerdict {
    Ok {
        final_priorities: BTreeMap<String, u32>,
        promoted: Vec<String>,
    },
    InvalidConfig,
}

pub fn propagate(base: &[(&str, u32)], edges: &[(&str, &str)]) -> InheritanceVerdict {
    if base.is_empty() {
        return InheritanceVerdict::InvalidConfig;
    }
    let mut priorities: BTreeMap<String, u32> = BTreeMap::new();
    for (id, p) in base {
        priorities.insert((*id).to_string(), *p);
    }
    let mut children: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (parent, child) in edges {
        children
            .entry((*parent).to_string())
            .or_default()
            .push((*child).to_string());
    }
    let mut promoted: Vec<String> = Vec::new();
    let mut changed = true;
    let mut iter = 0u32;
    while changed && iter < 100 {
        changed = false;
        iter += 1;
        let snapshot = priorities.clone();
        for (parent, child_list) in &children {
            let parent_p = snapshot.get(parent).copied().unwrap_or(0);
            for child in child_list {
                let child_p = snapshot.get(child).copied().unwrap_or(0);
                if parent_p > child_p {
                    priorities.insert(child.clone(), parent_p);
                    if !promoted.contains(child) {
                        promoted.push(child.clone());
                    }
                    changed = true;
                }
            }
        }
    }
    promoted.sort();
    InheritanceVerdict::Ok {
        final_priorities: priorities,
        promoted,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_priority_inheritance")?;

    let base = [("root", 100), ("child", 50), ("grand", 20)];
    let edges = [("root", "child"), ("child", "grand")];
    println!("propagate: {:?}", propagate(&base, &edges));
    println!("invalid: {:?}", propagate(&[], &[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propagator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn lower_child_promoted() {
        let base = [("root", 100), ("child", 10)];
        let edges = [("root", "child")];
        let v = propagate(&base, &edges);
        if let InheritanceVerdict::Ok { promoted, .. } = v {
            assert_eq!(promoted, vec!["child".to_string()]);
        }
    }

    #[test]
    fn higher_child_unchanged() {
        let base = [("root", 50), ("child", 100)];
        let edges = [("root", "child")];
        let v = propagate(&base, &edges);
        if let InheritanceVerdict::Ok { promoted, .. } = v {
            assert!(promoted.is_empty());
        }
    }

    #[test]
    fn empty_base_rejected() {
        assert_eq!(propagate(&[], &[]), InheritanceVerdict::InvalidConfig);
    }

    #[test]
    fn transitive_propagation() {
        let base = [("root", 100), ("child", 10), ("grand", 5)];
        let edges = [("root", "child"), ("child", "grand")];
        let v = propagate(&base, &edges);
        if let InheritanceVerdict::Ok {
            final_priorities, ..
        } = v
        {
            assert_eq!(final_priorities.get("grand"), Some(&100));
        }
    }

    #[test]
    fn final_priority_correct() {
        let base = [("root", 100), ("child", 10)];
        let edges = [("root", "child")];
        let v = propagate(&base, &edges);
        if let InheritanceVerdict::Ok {
            final_priorities, ..
        } = v
        {
            assert_eq!(final_priorities.get("child"), Some(&100));
        }
    }

    #[test]
    fn deterministic() {
        let base = [("root", 100), ("child", 10)];
        let edges = [("root", "child")];
        let r1 = propagate(&base, &edges);
        let r2 = propagate(&base, &edges);
        assert_eq!(r1, r2);
    }

    #[test]
    fn no_edges_no_promotion() {
        let base = [("a", 100), ("b", 10)];
        let v = propagate(&base, &[]);
        if let InheritanceVerdict::Ok { promoted, .. } = v {
            assert!(promoted.is_empty());
        }
    }

    #[test]
    fn promoted_sorted() {
        let base = [("root1", 100), ("root2", 100), ("zeta", 10), ("alpha", 10)];
        let edges = [("root1", "zeta"), ("root2", "alpha")];
        let v = propagate(&base, &edges);
        if let InheritanceVerdict::Ok { promoted, .. } = v {
            assert_eq!(promoted, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn cycle_capped_by_iter_limit() {
        let base = [("a", 100), ("b", 10)];
        let edges = [("a", "b"), ("b", "a")];
        let v = propagate(&base, &edges);
        if let InheritanceVerdict::Ok {
            final_priorities, ..
        } = v
        {
            // Both should converge to 100.
            assert_eq!(final_priorities.get("a"), Some(&100));
            assert_eq!(final_priorities.get("b"), Some(&100));
        }
    }

    #[test]
    fn diamond_propagation() {
        let base = [("root", 100), ("a", 10), ("b", 10), ("leaf", 5)];
        let edges = [("root", "a"), ("root", "b"), ("a", "leaf"), ("b", "leaf")];
        let v = propagate(&base, &edges);
        if let InheritanceVerdict::Ok {
            final_priorities, ..
        } = v
        {
            assert_eq!(final_priorities.get("leaf"), Some(&100));
        }
    }

    #[test]
    fn priority_already_max_unchanged() {
        let base = [("root", 100), ("child", 100)];
        let edges = [("root", "child")];
        let v = propagate(&base, &edges);
        if let InheritanceVerdict::Ok { promoted, .. } = v {
            assert!(promoted.is_empty());
        }
    }
}
