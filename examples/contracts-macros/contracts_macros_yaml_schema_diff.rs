//! # Contracts-Macros YAML Schema Diff
//!
//! Diff two schema field lists: report fields added, removed, or
//! type-changed. Distinguishes safe (additive) from breaking changes.
//!
//! Demonstrates the **CMM.54** recipe for PMAT-175 (catalog crosses 1200).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GraphQL schema diff / OpenAPI breaking-change checker.
//!
//! Run with: cargo run --example contracts_macros_yaml_schema_diff
//!
//! Added by PMAT-175 (catalog 1198→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum SchemaDiffVerdict {
    NoChange,
    Additive {
        added: Vec<String>,
    },
    Breaking {
        removed: Vec<String>,
        type_changed: Vec<String>,
    },
    Mixed {
        added: Vec<String>,
        removed: Vec<String>,
        type_changed: Vec<String>,
    },
    EmptyOldAndNew,
}

pub fn diff(old: &[(&str, &str)], new: &[(&str, &str)]) -> SchemaDiffVerdict {
    if old.is_empty() && new.is_empty() {
        return SchemaDiffVerdict::EmptyOldAndNew;
    }
    let old_map: BTreeMap<&str, &str> = old.iter().copied().collect();
    let new_map: BTreeMap<&str, &str> = new.iter().copied().collect();
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut type_changed = Vec::new();
    for (name, ty) in &new_map {
        match old_map.get(name) {
            None => added.push((*name).to_string()),
            Some(old_ty) if old_ty != ty => type_changed.push((*name).to_string()),
            _ => {}
        }
    }
    for name in old_map.keys() {
        if !new_map.contains_key(name) {
            removed.push((*name).to_string());
        }
    }
    let has_breaking = !removed.is_empty() || !type_changed.is_empty();
    let has_additive = !added.is_empty();
    match (has_breaking, has_additive) {
        (false, false) => SchemaDiffVerdict::NoChange,
        (false, true) => SchemaDiffVerdict::Additive { added },
        (true, false) => SchemaDiffVerdict::Breaking {
            removed,
            type_changed,
        },
        (true, true) => SchemaDiffVerdict::Mixed {
            added,
            removed,
            type_changed,
        },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_schema_diff")?;

    let v0 = [("name", "string"), ("count", "u32")];
    let same = [("name", "string"), ("count", "u32")];
    println!("no change: {:?}", diff(&v0, &same));

    let added = [("name", "string"), ("count", "u32"), ("color", "string")];
    println!("additive: {:?}", diff(&v0, &added));

    let breaking = [("name", "string")];
    println!("breaking: {:?}", diff(&v0, &breaking));

    let type_change = [("name", "i64"), ("count", "u32")];
    println!("type change: {:?}", diff(&v0, &type_change));

    println!("both empty: {:?}", diff(&[], &[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn differ_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_change_recognized() {
        let v = diff(&[("a", "u32")], &[("a", "u32")]);
        assert_eq!(v, SchemaDiffVerdict::NoChange);
    }

    #[test]
    fn additive_only() {
        let v = diff(&[("a", "u32")], &[("a", "u32"), ("b", "u64")]);
        assert!(matches!(v, SchemaDiffVerdict::Additive { .. }));
    }

    #[test]
    fn breaking_removal() {
        let v = diff(&[("a", "u32"), ("b", "u32")], &[("a", "u32")]);
        assert!(matches!(v, SchemaDiffVerdict::Breaking { .. }));
    }

    #[test]
    fn breaking_type_change() {
        let v = diff(&[("a", "u32")], &[("a", "i64")]);
        if let SchemaDiffVerdict::Breaking { type_changed, .. } = v {
            assert_eq!(type_changed, vec!["a".to_string()]);
        }
    }

    #[test]
    fn mixed_breaking_and_additive() {
        let v = diff(&[("a", "u32")], &[("b", "u64")]);
        assert!(matches!(v, SchemaDiffVerdict::Mixed { .. }));
    }

    #[test]
    fn both_empty_special() {
        assert_eq!(diff(&[], &[]), SchemaDiffVerdict::EmptyOldAndNew);
    }

    #[test]
    fn empty_old_all_added() {
        let v = diff(&[], &[("a", "u32")]);
        if let SchemaDiffVerdict::Additive { added } = v {
            assert_eq!(added, vec!["a".to_string()]);
        }
    }

    #[test]
    fn empty_new_all_removed() {
        let v = diff(&[("a", "u32")], &[]);
        if let SchemaDiffVerdict::Breaking { removed, .. } = v {
            assert_eq!(removed, vec!["a".to_string()]);
        }
    }

    #[test]
    fn type_change_in_mixed() {
        let v = diff(
            &[("a", "u32"), ("b", "u32")],
            &[("a", "i64"), ("c", "string")],
        );
        if let SchemaDiffVerdict::Mixed {
            added,
            removed,
            type_changed,
        } = v
        {
            assert_eq!(added.len(), 1);
            assert_eq!(removed.len(), 1);
            assert_eq!(type_changed.len(), 1);
        }
    }

    #[test]
    fn deterministic() {
        let a = diff(&[("a", "u32")], &[("a", "u32"), ("b", "u64")]);
        let b = diff(&[("a", "u32")], &[("a", "u32"), ("b", "u64")]);
        assert_eq!(a, b);
    }
}
