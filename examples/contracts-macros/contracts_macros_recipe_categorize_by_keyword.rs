//! # Contracts-Macros Recipe Categorize By Keyword
//!
//! Classify recipe names into action categories (verify / audit /
//! sim / transform / unknown) based on keyword prefixes. Returns
//! per-category counts.
//!
//! Demonstrates the **CMM.96** recipe for PMAT-189 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: convention-over-configuration (Rails ~2004); naming
//!  taxonomy in package management.
//!
//! Run with: cargo run --example contracts_macros_recipe_categorize_by_keyword
//!
//! Added by PMAT-189 (catalog 1324→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Category {
    Verify,
    Audit,
    Sim,
    Transform,
    Unknown,
}

#[derive(Debug, PartialEq)]
pub enum CategorizeVerdict {
    Ok {
        per_recipe: Vec<(String, Category)>,
        counts: BTreeMap<String, u32>,
    },
    InvalidConfig,
}

pub fn classify(recipes: &[&str]) -> CategorizeVerdict {
    if recipes.is_empty() {
        return CategorizeVerdict::InvalidConfig;
    }
    let mut per_recipe: Vec<(String, Category)> = Vec::with_capacity(recipes.len());
    let mut counts: BTreeMap<String, u32> = BTreeMap::new();
    for name in recipes {
        let cat = categorize(name);
        let key = format!("{cat:?}");
        *counts.entry(key).or_insert(0) += 1;
        per_recipe.push(((*name).to_string(), cat));
    }
    CategorizeVerdict::Ok { per_recipe, counts }
}

fn categorize(name: &str) -> Category {
    let lower = name.to_lowercase();
    if lower.contains("verify") || lower.contains("validate") || lower.contains("check") {
        Category::Verify
    } else if lower.contains("audit") || lower.contains("scan") || lower.contains("review") {
        Category::Audit
    } else if lower.contains("simulate") || lower.starts_with("mc_") {
        Category::Sim
    } else if lower.contains("convert") || lower.contains("normalize") || lower.contains("compile")
    {
        Category::Transform
    } else {
        Category::Unknown
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_categorize_by_keyword")?;

    let recipes = [
        "verify_signature",
        "audit_yaml_indent",
        "mc_packet_loss",
        "convert_format",
        "weird_thing",
    ];
    println!("classify: {:?}", classify(&recipes));
    println!("invalid: {:?}", classify(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn verify_recognized() {
        let v = classify(&["verify_x"]);
        if let CategorizeVerdict::Ok { per_recipe, .. } = v {
            assert_eq!(per_recipe[0].1, Category::Verify);
        }
    }

    #[test]
    fn audit_recognized() {
        let v = classify(&["audit_x"]);
        if let CategorizeVerdict::Ok { per_recipe, .. } = v {
            assert_eq!(per_recipe[0].1, Category::Audit);
        }
    }

    #[test]
    fn mc_prefix_is_sim() {
        let v = classify(&["mc_burst"]);
        if let CategorizeVerdict::Ok { per_recipe, .. } = v {
            assert_eq!(per_recipe[0].1, Category::Sim);
        }
    }

    #[test]
    fn convert_is_transform() {
        let v = classify(&["convert_x"]);
        if let CategorizeVerdict::Ok { per_recipe, .. } = v {
            assert_eq!(per_recipe[0].1, Category::Transform);
        }
    }

    #[test]
    fn unknown_unrecognized() {
        let v = classify(&["weird_thing"]);
        if let CategorizeVerdict::Ok { per_recipe, .. } = v {
            assert_eq!(per_recipe[0].1, Category::Unknown);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(classify(&[]), CategorizeVerdict::InvalidConfig);
    }

    #[test]
    fn counts_correct() {
        let v = classify(&["verify_a", "verify_b", "audit_c"]);
        if let CategorizeVerdict::Ok { counts, .. } = v {
            assert_eq!(counts.get("Verify"), Some(&2));
            assert_eq!(counts.get("Audit"), Some(&1));
        }
    }

    #[test]
    fn case_insensitive_match() {
        let v = classify(&["Verify_X", "VERIFY_Y"]);
        if let CategorizeVerdict::Ok { counts, .. } = v {
            assert_eq!(counts.get("Verify"), Some(&2));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = classify(&["verify_x"]);
        let r2 = classify(&["verify_x"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn validate_keyword_matches_verify() {
        let v = classify(&["validate_input"]);
        if let CategorizeVerdict::Ok { per_recipe, .. } = v {
            assert_eq!(per_recipe[0].1, Category::Verify);
        }
    }

    #[test]
    fn scan_keyword_matches_audit() {
        let v = classify(&["scan_dir"]);
        if let CategorizeVerdict::Ok { per_recipe, .. } = v {
            assert_eq!(per_recipe[0].1, Category::Audit);
        }
    }
}
