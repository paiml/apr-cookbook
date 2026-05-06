//! # Contracts-Macros Recipe Owner Round-Robin Assignment
//!
//! Round-robin assign N recipes to M owners. Returns assignments
//! plus per-owner load (max - min for fairness reporting).
//!
//! Demonstrates the **CMM.102** recipe for PMAT-191 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: round-robin scheduling (Silberschatz, OS Concepts §5.3);
//!  GitHub CODEOWNERS round-robin auto-assign.
//!
//! Run with: cargo run --example contracts_macros_recipe_owner_assignment
//!
//! Added by PMAT-191 (catalog 1342→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum AssignVerdict {
    Ok {
        assignments: Vec<(String, String)>,
        load_imbalance: u32,
    },
    InvalidConfig,
}

pub fn assign(recipes: &[&str], owners: &[&str]) -> AssignVerdict {
    if recipes.is_empty() || owners.is_empty() {
        return AssignVerdict::InvalidConfig;
    }
    let mut assignments: Vec<(String, String)> = Vec::with_capacity(recipes.len());
    let mut load: BTreeMap<String, u32> = BTreeMap::new();
    for (i, recipe) in recipes.iter().enumerate() {
        let owner = owners[i % owners.len()];
        assignments.push(((*recipe).to_string(), owner.to_string()));
        *load.entry(owner.to_string()).or_insert(0) += 1;
    }
    let max_load = load.values().max().copied().unwrap_or(0);
    let min_load = load.values().min().copied().unwrap_or(0);
    AssignVerdict::Ok {
        assignments,
        load_imbalance: max_load - min_load,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_owner_assignment")?;

    let recipes = ["r1", "r2", "r3", "r4"];
    let owners = ["alice", "bob"];
    println!("balanced: {:?}", assign(&recipes, &owners));
    println!("invalid: {:?}", assign(&[], &owners));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assigner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn balanced_assignment_zero_imbalance() {
        let recipes = ["r1", "r2", "r3", "r4"];
        let owners = ["alice", "bob"];
        let v = assign(&recipes, &owners);
        if let AssignVerdict::Ok { load_imbalance, .. } = v {
            assert_eq!(load_imbalance, 0);
        }
    }

    #[test]
    fn uneven_count_one_imbalance() {
        let recipes = ["r1", "r2", "r3"];
        let owners = ["alice", "bob"];
        let v = assign(&recipes, &owners);
        if let AssignVerdict::Ok { load_imbalance, .. } = v {
            assert_eq!(load_imbalance, 1);
        }
    }

    #[test]
    fn assignments_match_recipe_count() {
        let recipes = ["r1", "r2", "r3"];
        let owners = ["alice"];
        let v = assign(&recipes, &owners);
        if let AssignVerdict::Ok { assignments, .. } = v {
            assert_eq!(assignments.len(), 3);
        }
    }

    #[test]
    fn empty_recipes_rejected() {
        let owners = ["alice"];
        assert_eq!(assign(&[], &owners), AssignVerdict::InvalidConfig);
    }

    #[test]
    fn empty_owners_rejected() {
        let recipes = ["r"];
        assert_eq!(assign(&recipes, &[]), AssignVerdict::InvalidConfig);
    }

    #[test]
    fn round_robin_first_wraps() {
        let recipes = ["r1", "r2", "r3", "r4", "r5"];
        let owners = ["a", "b"];
        let v = assign(&recipes, &owners);
        if let AssignVerdict::Ok { assignments, .. } = v {
            assert_eq!(assignments[0].1, "a");
            assert_eq!(assignments[1].1, "b");
            assert_eq!(assignments[2].1, "a");
            assert_eq!(assignments[3].1, "b");
            assert_eq!(assignments[4].1, "a");
        }
    }

    #[test]
    fn deterministic() {
        let recipes = ["r1"];
        let owners = ["alice"];
        let r1 = assign(&recipes, &owners);
        let r2 = assign(&recipes, &owners);
        assert_eq!(r1, r2);
    }

    #[test]
    fn single_owner_takes_all() {
        let recipes = ["r1", "r2", "r3"];
        let owners = ["alice"];
        let v = assign(&recipes, &owners);
        if let AssignVerdict::Ok {
            assignments,
            load_imbalance,
        } = v
        {
            assert!(assignments.iter().all(|(_, o)| o == "alice"));
            assert_eq!(load_imbalance, 0);
        }
    }

    #[test]
    fn order_preserved() {
        let recipes = ["first", "second"];
        let owners = ["alice"];
        let v = assign(&recipes, &owners);
        if let AssignVerdict::Ok { assignments, .. } = v {
            assert_eq!(assignments[0].0, "first");
            assert_eq!(assignments[1].0, "second");
        }
    }

    #[test]
    fn load_imbalance_at_most_one_when_balanced() {
        let recipes = ["r1", "r2", "r3", "r4", "r5"];
        let owners = ["a", "b", "c"];
        let v = assign(&recipes, &owners);
        if let AssignVerdict::Ok { load_imbalance, .. } = v {
            assert!(load_imbalance <= 1);
        }
    }

    #[test]
    fn many_recipes_handled() {
        let recipes: Vec<&str> = vec!["r"; 100];
        let owners = ["a", "b", "c", "d"];
        let v = assign(&recipes, &owners);
        if let AssignVerdict::Ok { assignments, .. } = v {
            assert_eq!(assignments.len(), 100);
        }
    }
}
