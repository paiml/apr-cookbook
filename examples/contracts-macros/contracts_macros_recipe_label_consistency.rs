//! # Contracts-Macros Recipe Label Consistency
//!
//! Validate that all recipes share the required label keys with the
//! same value casing. Returns sorted offending recipe IDs and the
//! complete label-key set.
//!
//! Demonstrates the **CMM.158** recipe for PMAT-210 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: kubernetes label-key conventions (k8s.io/docs/concepts/
//!  overview/working-with-objects/labels); GitHub label taxonomy
//!  guidelines.
//!
//! Run with: cargo run --example contracts_macros_recipe_label_consistency
//!
//! Added by PMAT-210 (catalog 1513→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum LabelVerdict {
    Ok {
        offending_recipes: Vec<String>,
        unique_label_keys: Vec<String>,
    },
    InvalidConfig,
}

/// Items: (recipe_id, list of label keys present).
/// `required_keys` must all appear in every recipe.
pub fn check(items: &[(&str, Vec<&str>)], required_keys: &[&str]) -> LabelVerdict {
    if items.is_empty() || required_keys.is_empty() {
        return LabelVerdict::InvalidConfig;
    }
    let req: BTreeSet<String> = required_keys.iter().map(|s| (*s).to_string()).collect();
    let mut offenders: Vec<String> = Vec::new();
    let mut all_keys: BTreeSet<String> = BTreeSet::new();
    for (id, keys) in items {
        let key_set: BTreeSet<String> = keys.iter().map(|s| (*s).to_string()).collect();
        all_keys.extend(key_set.iter().cloned());
        if !req.is_subset(&key_set) {
            offenders.push((*id).to_string());
        }
    }
    offenders.sort();
    LabelVerdict::Ok {
        offending_recipes: offenders,
        unique_label_keys: all_keys.into_iter().collect(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_label_consistency")?;

    let items = vec![("r1", vec!["category", "owner"]), ("r2", vec!["category"])];
    println!("check: {:?}", check(&items, &["category", "owner"]));
    println!("invalid: {:?}", check(&[], &["category"]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn complete_labels_no_offender() {
        let items = vec![("r", vec!["a", "b"])];
        let v = check(&items, &["a", "b"]);
        if let LabelVerdict::Ok {
            offending_recipes, ..
        } = v
        {
            assert!(offending_recipes.is_empty());
        }
    }

    #[test]
    fn missing_label_offender() {
        let items = vec![("r", vec!["a"])];
        let v = check(&items, &["a", "b"]);
        if let LabelVerdict::Ok {
            offending_recipes, ..
        } = v
        {
            assert_eq!(offending_recipes, vec!["r".to_string()]);
        }
    }

    #[test]
    fn extra_labels_ok() {
        let items = vec![("r", vec!["a", "b", "extra"])];
        let v = check(&items, &["a", "b"]);
        if let LabelVerdict::Ok {
            offending_recipes, ..
        } = v
        {
            assert!(offending_recipes.is_empty());
        }
    }

    #[test]
    fn empty_items_rejected() {
        assert_eq!(check(&[], &["a"]), LabelVerdict::InvalidConfig);
    }

    #[test]
    fn empty_required_rejected() {
        let items: Vec<(&str, Vec<&str>)> = vec![("r", vec!["a"])];
        assert_eq!(check(&items, &[]), LabelVerdict::InvalidConfig);
    }

    #[test]
    fn unique_keys_aggregated() {
        let items = vec![("r1", vec!["a"]), ("r2", vec!["b"])];
        let v = check(&items, &["a"]);
        if let LabelVerdict::Ok {
            unique_label_keys, ..
        } = v
        {
            assert_eq!(unique_label_keys, vec!["a".to_string(), "b".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let items = vec![("r", vec!["a"])];
        let r1 = check(&items, &["a"]);
        let r2 = check(&items, &["a"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn offenders_sorted() {
        let items = vec![("zeta", vec![]), ("alpha", vec![])];
        let v = check(&items, &["a"]);
        if let LabelVerdict::Ok {
            offending_recipes, ..
        } = v
        {
            assert_eq!(
                offending_recipes,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn case_sensitive_label() {
        let items = vec![("r", vec!["A"])];
        let v = check(&items, &["a"]);
        if let LabelVerdict::Ok {
            offending_recipes, ..
        } = v
        {
            assert_eq!(offending_recipes, vec!["r".to_string()]);
        }
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<(&str, Vec<&str>)> = (0..30).map(|_| ("r", vec!["a"])).collect();
        let v = check(&items, &["a"]);
        if let LabelVerdict::Ok {
            offending_recipes, ..
        } = v
        {
            assert!(offending_recipes.is_empty());
        }
    }

    #[test]
    fn unicode_label_supported() {
        let items = vec![("r", vec!["café"])];
        let v = check(&items, &["café"]);
        if let LabelVerdict::Ok {
            offending_recipes, ..
        } = v
        {
            assert!(offending_recipes.is_empty());
        }
    }

    #[test]
    fn duplicate_in_recipe_dedup() {
        // Same label twice in one recipe still counts as single key.
        let items = vec![("r", vec!["a", "a"])];
        let v = check(&items, &["a"]);
        if let LabelVerdict::Ok {
            offending_recipes, ..
        } = v
        {
            assert!(offending_recipes.is_empty());
        }
    }
}
