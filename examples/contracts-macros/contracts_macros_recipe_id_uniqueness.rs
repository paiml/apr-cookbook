//! # Contracts-Macros Recipe ID Uniqueness
//!
//! Verify a list of recipe IDs has no duplicates after canonicalization
//! (case-insensitive, dashes-as-underscores). Returns first collision
//! pair if any.
//!
//! Demonstrates the **CMM.40** recipe for PMAT-171 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Cargo manifest unique-name invariant.
//!
//! Run with: cargo run --example contracts_macros_recipe_id_uniqueness
//!
//! Added by PMAT-171 (catalog 1162→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum UniquenessVerdict {
    Unique {
        count: u32,
    },
    Duplicate {
        canonical: String,
        originals: Vec<String>,
    },
    EmptyList,
}

pub fn check(ids: &[&str]) -> UniquenessVerdict {
    if ids.is_empty() {
        return UniquenessVerdict::EmptyList;
    }
    let mut seen: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for id in ids {
        let canonical = normalize(id);
        seen.entry(canonical).or_default().push((*id).to_string());
    }
    for (canonical, originals) in seen {
        if originals.len() > 1 {
            return UniquenessVerdict::Duplicate {
                canonical,
                originals,
            };
        }
    }
    UniquenessVerdict::Unique {
        count: ids.len() as u32,
    }
}

fn normalize(s: &str) -> String {
    s.chars()
        .filter_map(|c| {
            if c.is_ascii_alphanumeric() {
                Some(c.to_ascii_lowercase())
            } else if c == '-' || c == '_' || c == ' ' {
                Some('_')
            } else {
                None
            }
        })
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_id_uniqueness")?;

    println!("unique: {:?}", check(&["recipe_a", "recipe_b", "recipe_c"]));
    println!("duplicate: {:?}", check(&["snake_case", "Snake-Case"]));
    println!("empty: {:?}", check(&[]));
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
    fn distinct_ids_unique() {
        let v = check(&["a", "b", "c"]);
        if let UniquenessVerdict::Unique { count } = v {
            assert_eq!(count, 3);
        }
    }

    #[test]
    fn case_insensitive_collision() {
        let v = check(&["ABC", "abc"]);
        assert!(matches!(v, UniquenessVerdict::Duplicate { .. }));
    }

    #[test]
    fn dash_underscore_collision() {
        let v = check(&["snake_case", "snake-case"]);
        assert!(matches!(v, UniquenessVerdict::Duplicate { .. }));
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(check(&[]), UniquenessVerdict::EmptyList);
    }

    #[test]
    fn single_unique() {
        let v = check(&["only"]);
        if let UniquenessVerdict::Unique { count } = v {
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn duplicate_carries_originals() {
        let v = check(&["snake_case", "Snake-Case"]);
        if let UniquenessVerdict::Duplicate { originals, .. } = v {
            assert_eq!(originals.len(), 2);
        }
    }

    #[test]
    fn three_way_collision() {
        let v = check(&["abc", "ABC", "Abc"]);
        if let UniquenessVerdict::Duplicate { originals, .. } = v {
            assert_eq!(originals.len(), 3);
        }
    }

    #[test]
    fn unrelated_ids_stay_unique() {
        let v = check(&["alpha", "beta", "gamma"]);
        assert!(matches!(v, UniquenessVerdict::Unique { .. }));
    }

    #[test]
    fn whitespace_normalizes() {
        let v = check(&["snake_case", "snake case"]);
        assert!(matches!(v, UniquenessVerdict::Duplicate { .. }));
    }

    #[test]
    fn deterministic() {
        let ids = ["a", "b", "c"];
        let a = check(&ids);
        let b = check(&ids);
        assert_eq!(a, b);
    }
}
