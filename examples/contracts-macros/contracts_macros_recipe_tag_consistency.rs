//! # Contracts-Macros Recipe Tag Consistency
//!
//! Validate that every recipe uses tags from the official taxonomy.
//! Returns sorted unknown-tag list and total tag-usage count.
//!
//! Demonstrates the **CMM.138** recipe for PMAT-203 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Dublin Core Metadata Initiative; taxonomy governance for
//!  controlled vocabularies (ANSI/NISO Z39.19).
//!
//! Run with: cargo run --example contracts_macros_recipe_tag_consistency
//!
//! Added by PMAT-203 (catalog 1450→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum TagConsistVerdict {
    Ok {
        unknown_tags: Vec<String>,
        total_tag_uses: u32,
    },
    InvalidConfig,
}

pub fn check(recipes: &[(&str, Vec<&str>)], taxonomy: &[&str]) -> TagConsistVerdict {
    if recipes.is_empty() || taxonomy.is_empty() {
        return TagConsistVerdict::InvalidConfig;
    }
    let valid: BTreeSet<String> = taxonomy.iter().map(|t| (*t).to_string()).collect();
    let mut unknown: BTreeSet<String> = BTreeSet::new();
    let mut total = 0u32;
    for (_, tags) in recipes {
        for t in tags {
            total += 1;
            if !valid.contains(*t) {
                unknown.insert((*t).to_string());
            }
        }
    }
    TagConsistVerdict::Ok {
        unknown_tags: unknown.into_iter().collect(),
        total_tag_uses: total,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_tag_consistency")?;

    let recipes = vec![
        ("r1", vec!["tui", "verified"]),
        ("r2", vec!["mc", "experimental"]),
    ];
    let taxonomy = ["tui", "mc", "verified"];
    println!("check: {:?}", check(&recipes, &taxonomy));
    println!("invalid: {:?}", check(&[], &taxonomy));
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
    fn all_known_tags_no_unknown() {
        let recipes = vec![("r", vec!["a", "b"])];
        let v = check(&recipes, &["a", "b"]);
        if let TagConsistVerdict::Ok { unknown_tags, .. } = v {
            assert!(unknown_tags.is_empty());
        }
    }

    #[test]
    fn unknown_tag_flagged() {
        let recipes = vec![("r", vec!["a", "x"])];
        let v = check(&recipes, &["a"]);
        if let TagConsistVerdict::Ok { unknown_tags, .. } = v {
            assert_eq!(unknown_tags, vec!["x".to_string()]);
        }
    }

    #[test]
    fn total_uses_correct() {
        let recipes = vec![("r1", vec!["a", "b"]), ("r2", vec!["a"])];
        let v = check(&recipes, &["a", "b"]);
        if let TagConsistVerdict::Ok { total_tag_uses, .. } = v {
            assert_eq!(total_tag_uses, 3);
        }
    }

    #[test]
    fn empty_recipes_rejected() {
        let v = check(&[], &["a"]);
        assert_eq!(v, TagConsistVerdict::InvalidConfig);
    }

    #[test]
    fn empty_taxonomy_rejected() {
        let recipes: Vec<(&str, Vec<&str>)> = vec![("r", vec!["a"])];
        let v = check(&recipes, &[]);
        assert_eq!(v, TagConsistVerdict::InvalidConfig);
    }

    #[test]
    fn unknown_dedupe() {
        let recipes = vec![("r1", vec!["x"]), ("r2", vec!["x"])];
        let v = check(&recipes, &["a"]);
        if let TagConsistVerdict::Ok { unknown_tags, .. } = v {
            assert_eq!(unknown_tags.len(), 1);
        }
    }

    #[test]
    fn unknown_sorted() {
        let recipes = vec![("r", vec!["zeta", "alpha"])];
        let v = check(&recipes, &["valid"]);
        if let TagConsistVerdict::Ok { unknown_tags, .. } = v {
            assert_eq!(unknown_tags, vec!["alpha".to_string(), "zeta".to_string()]);
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
    fn no_tags_zero_uses() {
        let recipes: Vec<(&str, Vec<&str>)> = vec![("r", vec![])];
        let v = check(&recipes, &["a"]);
        if let TagConsistVerdict::Ok { total_tag_uses, .. } = v {
            assert_eq!(total_tag_uses, 0);
        }
    }

    #[test]
    fn many_recipes_handled() {
        let recipes: Vec<(&str, Vec<&str>)> = (0..30).map(|_| ("r", vec!["a"])).collect();
        let v = check(&recipes, &["a"]);
        if let TagConsistVerdict::Ok { total_tag_uses, .. } = v {
            assert_eq!(total_tag_uses, 30);
        }
    }

    #[test]
    fn case_sensitive_tag() {
        let recipes = vec![("r", vec!["Tui"])];
        let v = check(&recipes, &["tui"]);
        if let TagConsistVerdict::Ok { unknown_tags, .. } = v {
            assert_eq!(unknown_tags, vec!["Tui".to_string()]);
        }
    }
}
