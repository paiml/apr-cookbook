//! # Contracts-Macros Recipe Metadata Min
//!
//! Validate that each recipe declares the minimum required metadata
//! fields (e.g., owner, severity, citation). Returns sorted offending
//! recipe IDs and per-field missing-count.
//!
//! Demonstrates the **CMM.164** recipe for PMAT-212 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SBOM minimum-element rules (NTIA 2021); SLSA provenance
//!  predicate required-attestations.
//!
//! Run with: cargo run --example contracts_macros_recipe_metadata_min
//!
//! Added by PMAT-212 (catalog 1531→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum MetaVerdict {
    Ok {
        offending_recipes: Vec<String>,
        missing_per_field: BTreeMap<String, u32>,
    },
    InvalidConfig,
}

/// `recipes`: (id, list of fields present). `required`: fields all must declare.
pub fn check(recipes: &[(&str, Vec<&str>)], required: &[&str]) -> MetaVerdict {
    if recipes.is_empty() || required.is_empty() {
        return MetaVerdict::InvalidConfig;
    }
    let mut offenders: Vec<String> = Vec::new();
    let mut missing: BTreeMap<String, u32> = BTreeMap::new();
    for r in required {
        missing.insert((*r).to_string(), 0);
    }
    for (id, fields) in recipes {
        let mut is_offender = false;
        for r in required {
            if !fields.contains(r) {
                if let Some(c) = missing.get_mut(*r) {
                    *c += 1;
                }
                is_offender = true;
            }
        }
        if is_offender {
            offenders.push((*id).to_string());
        }
    }
    offenders.sort();
    MetaVerdict::Ok {
        offending_recipes: offenders,
        missing_per_field: missing,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_metadata_min")?;

    let recipes = vec![("r1", vec!["owner", "severity"]), ("r2", vec!["severity"])];
    println!("check: {:?}", check(&recipes, &["owner", "severity"]));
    println!("invalid: {:?}", check(&[], &["owner"]));
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
    fn complete_no_offender() {
        let recipes = vec![("r", vec!["a", "b"])];
        let v = check(&recipes, &["a", "b"]);
        if let MetaVerdict::Ok {
            offending_recipes, ..
        } = v
        {
            assert!(offending_recipes.is_empty());
        }
    }

    #[test]
    fn missing_field_offender() {
        let recipes = vec![("r", vec!["a"])];
        let v = check(&recipes, &["a", "b"]);
        if let MetaVerdict::Ok {
            offending_recipes, ..
        } = v
        {
            assert_eq!(offending_recipes, vec!["r".to_string()]);
        }
    }

    #[test]
    fn empty_recipes_rejected() {
        assert_eq!(check(&[], &["a"]), MetaVerdict::InvalidConfig);
    }

    #[test]
    fn empty_required_rejected() {
        let recipes: Vec<(&str, Vec<&str>)> = vec![("r", vec!["a"])];
        assert_eq!(check(&recipes, &[]), MetaVerdict::InvalidConfig);
    }

    #[test]
    fn missing_per_field_count() {
        let recipes = vec![("a", vec!["owner"]), ("b", vec!["severity"]), ("c", vec![])];
        let v = check(&recipes, &["owner", "severity"]);
        if let MetaVerdict::Ok {
            missing_per_field, ..
        } = v
        {
            assert_eq!(missing_per_field.get("owner"), Some(&2));
            assert_eq!(missing_per_field.get("severity"), Some(&2));
        }
    }

    #[test]
    fn deterministic() {
        let recipes = vec![("r", vec!["a"])];
        let r1 = check(&recipes, &["a"]);
        let r2 = check(&recipes, &["a"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn offenders_sorted() {
        let recipes = vec![("zeta", vec![]), ("alpha", vec![])];
        let v = check(&recipes, &["x"]);
        if let MetaVerdict::Ok {
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
    fn extra_fields_ok() {
        let recipes = vec![("r", vec!["a", "b", "extra"])];
        let v = check(&recipes, &["a"]);
        if let MetaVerdict::Ok {
            offending_recipes, ..
        } = v
        {
            assert!(offending_recipes.is_empty());
        }
    }

    #[test]
    fn case_sensitive_field() {
        let recipes = vec![("r", vec!["A"])];
        let v = check(&recipes, &["a"]);
        if let MetaVerdict::Ok {
            offending_recipes, ..
        } = v
        {
            assert_eq!(offending_recipes, vec!["r".to_string()]);
        }
    }

    #[test]
    fn many_recipes_handled() {
        let recipes: Vec<(&str, Vec<&str>)> = (0..30).map(|_| ("r", vec![])).collect();
        let v = check(&recipes, &["a"]);
        if let MetaVerdict::Ok {
            offending_recipes, ..
        } = v
        {
            // dedup since "r" repeats
            assert_eq!(offending_recipes.len(), 30);
        }
    }

    #[test]
    fn unicode_field_supported() {
        let recipes = vec![("r", vec!["café"])];
        let v = check(&recipes, &["café"]);
        if let MetaVerdict::Ok {
            offending_recipes, ..
        } = v
        {
            assert!(offending_recipes.is_empty());
        }
    }
}
