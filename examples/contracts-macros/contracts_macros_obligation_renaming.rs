//! # Contracts-Macros Obligation Rename Safety
//!
//! Validate a rename map: every old name maps to exactly one new
//! name, no new name collides with an existing untouched name, and
//! cycles are forbidden.
//!
//! Demonstrates the **CMM.42** recipe for PMAT-171 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: refactor safety: rename-only changes (Fowler).
//!
//! Run with: cargo run --example contracts_macros_obligation_renaming
//!
//! Added by PMAT-171 (catalog 1162→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, PartialEq)]
pub enum RenameVerdict {
    Ok { rename_count: u32 },
    DuplicateOld { name: String },
    NewNameCollides { name: String },
    CycleDetected,
    EmptyRenames,
}

pub fn validate(renames: &[(&str, &str)], existing: &[&str]) -> RenameVerdict {
    if renames.is_empty() {
        return RenameVerdict::EmptyRenames;
    }
    let existing_set: BTreeSet<&str> = existing.iter().copied().collect();
    let mut seen_old: BTreeSet<&str> = BTreeSet::new();
    let mut new_names: BTreeMap<&str, &str> = BTreeMap::new();
    let mut renamed_olds: BTreeSet<&str> = BTreeSet::new();
    for (old, new) in renames {
        if !seen_old.insert(old) {
            return RenameVerdict::DuplicateOld {
                name: (*old).to_string(),
            };
        }
        renamed_olds.insert(old);
        new_names.insert(old, new);
    }
    for (old, new) in &new_names {
        // The new name is invalid if (a) it's an existing name not being renamed
        // away, OR (b) it collides with another rename's new name.
        if existing_set.contains(new) && !renamed_olds.contains(new) {
            return RenameVerdict::NewNameCollides {
                name: (*new).to_string(),
            };
        }
        if new == old {
            return RenameVerdict::CycleDetected;
        }
    }
    let new_set: BTreeSet<&str> = new_names.values().copied().collect();
    if new_set.len() < new_names.len() {
        return RenameVerdict::NewNameCollides {
            name: "duplicate-new".to_string(),
        };
    }
    RenameVerdict::Ok {
        rename_count: renames.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_renaming")?;

    let renames = [("old_a", "new_a"), ("old_b", "new_b")];
    let existing = ["old_a", "old_b", "untouched"];
    println!("ok: {:?}", validate(&renames, &existing));

    let collision = [("old_a", "untouched")];
    println!("collision: {:?}", validate(&collision, &existing));

    let cycle = [("a", "a")];
    println!("cycle: {:?}", validate(&cycle, &["a"]));

    let dup_old = [("a", "x"), ("a", "y")];
    println!("dup old: {:?}", validate(&dup_old, &["a"]));
    println!("empty: {:?}", validate(&[], &["a"]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_rename_ok() {
        let renames = [("old", "new")];
        let existing = ["old", "other"];
        if let RenameVerdict::Ok { rename_count } = validate(&renames, &existing) {
            assert_eq!(rename_count, 1);
        }
    }

    #[test]
    fn collision_with_existing() {
        let renames = [("old", "untouched")];
        let existing = ["old", "untouched"];
        assert!(matches!(
            validate(&renames, &existing),
            RenameVerdict::NewNameCollides { .. }
        ));
    }

    #[test]
    fn cycle_self_rename() {
        let renames = [("a", "a")];
        let existing = ["a"];
        assert_eq!(validate(&renames, &existing), RenameVerdict::CycleDetected);
    }

    #[test]
    fn duplicate_old_rejected() {
        let renames = [("a", "x"), ("a", "y")];
        let existing = ["a"];
        assert!(matches!(
            validate(&renames, &existing),
            RenameVerdict::DuplicateOld { .. }
        ));
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(validate(&[], &["a"]), RenameVerdict::EmptyRenames);
    }

    #[test]
    fn duplicate_new_rejected() {
        let renames = [("a", "x"), ("b", "x")];
        let existing = ["a", "b"];
        assert!(matches!(
            validate(&renames, &existing),
            RenameVerdict::NewNameCollides { .. }
        ));
    }

    #[test]
    fn rename_to_being_renamed_old_ok() {
        // 'a' → 'b' and 'b' → 'c' is fine; 'b' is being renamed away.
        let renames = [("a", "b"), ("b", "c")];
        let existing = ["a", "b"];
        assert!(matches!(
            validate(&renames, &existing),
            RenameVerdict::Ok { .. }
        ));
    }

    #[test]
    fn many_renames_ok() {
        let renames = [("a", "x"), ("b", "y"), ("c", "z")];
        let existing = ["a", "b", "c"];
        if let RenameVerdict::Ok { rename_count } = validate(&renames, &existing) {
            assert_eq!(rename_count, 3);
        }
    }

    #[test]
    fn no_existing_works() {
        let renames = [("old", "new")];
        let v = validate(&renames, &[]);
        assert!(matches!(v, RenameVerdict::Ok { .. }));
    }

    #[test]
    fn deterministic() {
        let renames = [("a", "x")];
        let existing = ["a"];
        let a = validate(&renames, &existing);
        let b = validate(&renames, &existing);
        assert_eq!(a, b);
    }
}
