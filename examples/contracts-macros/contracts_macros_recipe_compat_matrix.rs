//! # Contracts-Macros Recipe Compatibility Matrix
//!
//! Validate recipe spec-version against an allowed compatibility
//! matrix `(recipe_id, target_spec_version)` pairs. Returns sorted
//! incompatible IDs.
//!
//! Demonstrates the **CMM.161** recipe for PMAT-211 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SemVer 2.0 §11 precedence; npm peerDependencies version
//!  range matching.
//!
//! Run with: cargo run --example contracts_macros_recipe_compat_matrix
//!
//! Added by PMAT-211 (catalog 1522→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CompatVerdict {
    Ok {
        incompatible_ids: Vec<String>,
        compatible_count: u32,
    },
    InvalidConfig,
}

/// `recipes`: (id, declared_version). `allowed`: list of versions.
pub fn check(recipes: &[(&str, u32)], allowed: &[u32]) -> CompatVerdict {
    if recipes.is_empty() || allowed.is_empty() {
        return CompatVerdict::InvalidConfig;
    }
    let mut incompatible: Vec<String> = recipes
        .iter()
        .filter(|(_, v)| !allowed.contains(v))
        .map(|(id, _)| (*id).to_string())
        .collect();
    incompatible.sort();
    let compatible = recipes.len() as u32 - incompatible.len() as u32;
    CompatVerdict::Ok {
        incompatible_ids: incompatible,
        compatible_count: compatible,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_compat_matrix")?;

    let recipes = [("r1", 6), ("r2", 7), ("r3", 4)];
    println!("allowed [5,6]: {:?}", check(&recipes, &[5, 6]));
    println!("invalid: {:?}", check(&[], &[6]));
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
    fn allowed_version_compatible() {
        let v = check(&[("r", 6)], &[6]);
        if let CompatVerdict::Ok {
            incompatible_ids, ..
        } = v
        {
            assert!(incompatible_ids.is_empty());
        }
    }

    #[test]
    fn disallowed_version_incompatible() {
        let v = check(&[("r", 9)], &[6]);
        if let CompatVerdict::Ok {
            incompatible_ids, ..
        } = v
        {
            assert_eq!(incompatible_ids, vec!["r".to_string()]);
        }
    }

    #[test]
    fn multiple_allowed_handled() {
        let v = check(&[("r", 5)], &[5, 6, 7]);
        if let CompatVerdict::Ok {
            incompatible_ids, ..
        } = v
        {
            assert!(incompatible_ids.is_empty());
        }
    }

    #[test]
    fn empty_recipes_rejected() {
        assert_eq!(check(&[], &[6]), CompatVerdict::InvalidConfig);
    }

    #[test]
    fn empty_allowed_rejected() {
        assert_eq!(check(&[("r", 6)], &[]), CompatVerdict::InvalidConfig);
    }

    #[test]
    fn compatible_count_correct() {
        let v = check(&[("a", 6), ("b", 9), ("c", 6)], &[6]);
        if let CompatVerdict::Ok {
            compatible_count, ..
        } = v
        {
            assert_eq!(compatible_count, 2);
        }
    }

    #[test]
    fn incompatible_sorted() {
        let v = check(&[("zeta", 9), ("alpha", 9)], &[6]);
        if let CompatVerdict::Ok {
            incompatible_ids, ..
        } = v
        {
            assert_eq!(
                incompatible_ids,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check(&[("r", 6)], &[6]);
        let r2 = check(&[("r", 6)], &[6]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn many_recipes_handled() {
        let recipes: Vec<(&str, u32)> = (0..30).map(|_| ("r", 9)).collect();
        let v = check(&recipes, &[6]);
        if let CompatVerdict::Ok {
            incompatible_ids, ..
        } = v
        {
            assert_eq!(incompatible_ids.len(), 30);
        }
    }

    #[test]
    fn unicode_id_supported() {
        let v = check(&[("café", 9)], &[6]);
        if let CompatVerdict::Ok {
            incompatible_ids, ..
        } = v
        {
            assert_eq!(incompatible_ids, vec!["café".to_string()]);
        }
    }

    #[test]
    fn version_zero_handled() {
        let v = check(&[("r", 0)], &[0]);
        if let CompatVerdict::Ok {
            incompatible_ids, ..
        } = v
        {
            assert!(incompatible_ids.is_empty());
        }
    }
}
