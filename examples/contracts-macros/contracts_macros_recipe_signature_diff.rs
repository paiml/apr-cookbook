//! # Contracts-Macros Recipe Signature Diff
//!
//! Compare two recipe signatures (input/output sets) to determine
//! if the change is compatible (additive only) or incompatible
//! (removed or changed fields). Returns added/removed/changed sets.
//!
//! Demonstrates the **CMM.99** recipe for PMAT-190 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: protobuf field-number compatibility rules; gRPC API
//!  evolution best practices.
//!
//! Run with: cargo run --example contracts_macros_recipe_signature_diff
//!
//! Added by PMAT-190 (catalog 1333→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum DiffVerdict {
    Ok {
        added: Vec<String>,
        removed: Vec<String>,
        type_changed: Vec<String>,
        compatible: bool,
    },
    InvalidConfig,
}

pub fn diff(before: &[(&str, &str)], after: &[(&str, &str)]) -> DiffVerdict {
    if before.is_empty() && after.is_empty() {
        return DiffVerdict::InvalidConfig;
    }
    let before_map: BTreeMap<String, String> = before
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    let after_map: BTreeMap<String, String> = after
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    let before_keys: BTreeSet<&String> = before_map.keys().collect();
    let after_keys: BTreeSet<&String> = after_map.keys().collect();
    let added: Vec<String> = after_keys
        .difference(&before_keys)
        .map(|s| (*s).clone())
        .collect();
    let removed: Vec<String> = before_keys
        .difference(&after_keys)
        .map(|s| (*s).clone())
        .collect();
    let type_changed: Vec<String> = before_keys
        .intersection(&after_keys)
        .filter_map(|k| {
            let b = before_map.get(*k).unwrap();
            let a = after_map.get(*k).unwrap();
            if b == a {
                None
            } else {
                Some((*k).clone())
            }
        })
        .collect();
    let compatible = removed.is_empty() && type_changed.is_empty();
    DiffVerdict::Ok {
        added,
        removed,
        type_changed,
        compatible,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_signature_diff")?;

    let before = [("x", "i32"), ("y", "f64")];
    let after = [("x", "i32"), ("y", "f64"), ("z", "u32")];
    println!("compatible: {:?}", diff(&before, &after));
    let after2 = [("y", "f64")];
    println!("incompatible: {:?}", diff(&before, &after2));
    println!("invalid: {:?}", diff(&[], &[]));
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
    fn additive_change_compatible() {
        let before = [("x", "i32")];
        let after = [("x", "i32"), ("y", "f64")];
        let v = diff(&before, &after);
        if let DiffVerdict::Ok { compatible, .. } = v {
            assert!(compatible);
        }
    }

    #[test]
    fn removed_field_incompatible() {
        let before = [("x", "i32"), ("y", "f64")];
        let after = [("x", "i32")];
        let v = diff(&before, &after);
        if let DiffVerdict::Ok { compatible, .. } = v {
            assert!(!compatible);
        }
    }

    #[test]
    fn type_change_incompatible() {
        let before = [("x", "i32")];
        let after = [("x", "u32")];
        let v = diff(&before, &after);
        if let DiffVerdict::Ok { compatible, .. } = v {
            assert!(!compatible);
        }
    }

    #[test]
    fn no_change_compatible() {
        let fields = [("x", "i32")];
        let v = diff(&fields, &fields);
        if let DiffVerdict::Ok { compatible, .. } = v {
            assert!(compatible);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(diff(&[], &[]), DiffVerdict::InvalidConfig);
    }

    #[test]
    fn added_correctly() {
        let before = [("x", "i32")];
        let after = [("x", "i32"), ("y", "f64")];
        let v = diff(&before, &after);
        if let DiffVerdict::Ok { added, .. } = v {
            assert_eq!(added, vec!["y".to_string()]);
        }
    }

    #[test]
    fn removed_correctly() {
        let before = [("x", "i32"), ("y", "f64")];
        let after = [("x", "i32")];
        let v = diff(&before, &after);
        if let DiffVerdict::Ok { removed, .. } = v {
            assert_eq!(removed, vec!["y".to_string()]);
        }
    }

    #[test]
    fn type_changed_correctly() {
        let before = [("x", "i32")];
        let after = [("x", "u32")];
        let v = diff(&before, &after);
        if let DiffVerdict::Ok { type_changed, .. } = v {
            assert_eq!(type_changed, vec!["x".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let before = [("x", "i32")];
        let after = [("y", "f64")];
        let r1 = diff(&before, &after);
        let r2 = diff(&before, &after);
        assert_eq!(r1, r2);
    }

    #[test]
    fn empty_to_nonempty_only_added() {
        let after = [("x", "i32")];
        let v = diff(&[], &after);
        if let DiffVerdict::Ok {
            added,
            removed,
            compatible,
            ..
        } = v
        {
            assert_eq!(added.len(), 1);
            assert!(removed.is_empty());
            assert!(compatible);
        }
    }

    #[test]
    fn nonempty_to_empty_only_removed() {
        let before = [("x", "i32")];
        let v = diff(&before, &[]);
        if let DiffVerdict::Ok { added, removed, .. } = v {
            assert!(added.is_empty());
            assert_eq!(removed, vec!["x".to_string()]);
        }
    }
}
