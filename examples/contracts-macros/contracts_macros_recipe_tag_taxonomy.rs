//! # Contracts-Macros Recipe Tag Taxonomy
//!
//! Validate recipe tags against a known taxonomy: flag unknown tags,
//! flag duplicate tags, and report tag-frequency histogram. Useful
//! for keeping the catalog's tag space disciplined.
//!
//! Demonstrates the **CMM.67** recipe for PMAT-180 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Dublin Core controlled vocabulary; Linnaean taxonomy.
//!
//! Run with: cargo run --example contracts_macros_recipe_tag_taxonomy
//!
//! Added by PMAT-180 (catalog 1243→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum TaxonomyVerdict {
    Ok {
        unknown_tags: Vec<String>,
        duplicate_tags: Vec<String>,
        histogram: BTreeMap<String, u32>,
    },
    InvalidConfig,
}

pub fn audit(allowed: &[&str], applied: &[&str]) -> TaxonomyVerdict {
    if allowed.is_empty() {
        return TaxonomyVerdict::InvalidConfig;
    }
    let allowed_set: BTreeSet<&str> = allowed.iter().copied().collect();
    let mut unknown_set: BTreeSet<&str> = BTreeSet::new();
    let mut histogram: BTreeMap<String, u32> = BTreeMap::new();
    for tag in applied {
        *histogram.entry((*tag).to_string()).or_insert(0) += 1;
        if !allowed_set.contains(tag) {
            unknown_set.insert(*tag);
        }
    }
    let unknown_tags: Vec<String> = unknown_set.into_iter().map(String::from).collect();
    let duplicate_tags: Vec<String> = histogram
        .iter()
        .filter(|(_, &c)| c > 1)
        .map(|(k, _)| k.clone())
        .collect();
    TaxonomyVerdict::Ok {
        unknown_tags,
        duplicate_tags,
        histogram,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_tag_taxonomy")?;

    let allowed = ["wasm", "gpu", "monte-carlo", "tui"];
    let applied = ["wasm", "gpu", "wasm", "unknown_tag"];
    println!("audit: {:?}", audit(&allowed, &applied));
    println!("invalid: {:?}", audit(&[], &applied));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auditor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn all_known_tags_pass() {
        let allowed = ["a", "b", "c"];
        let applied = ["a", "b"];
        let v = audit(&allowed, &applied);
        if let TaxonomyVerdict::Ok {
            unknown_tags,
            duplicate_tags,
            ..
        } = v
        {
            assert!(unknown_tags.is_empty());
            assert!(duplicate_tags.is_empty());
        }
    }

    #[test]
    fn unknown_tag_flagged() {
        let allowed = ["a", "b"];
        let applied = ["a", "z"];
        let v = audit(&allowed, &applied);
        if let TaxonomyVerdict::Ok { unknown_tags, .. } = v {
            assert_eq!(unknown_tags, vec!["z".to_string()]);
        }
    }

    #[test]
    fn duplicate_tag_flagged() {
        let allowed = ["a", "b"];
        let applied = ["a", "a", "b"];
        let v = audit(&allowed, &applied);
        if let TaxonomyVerdict::Ok { duplicate_tags, .. } = v {
            assert_eq!(duplicate_tags, vec!["a".to_string()]);
        }
    }

    #[test]
    fn histogram_correct() {
        let allowed = ["a"];
        let applied = ["a", "a", "a"];
        let v = audit(&allowed, &applied);
        if let TaxonomyVerdict::Ok { histogram, .. } = v {
            assert_eq!(histogram.get("a"), Some(&3));
        }
    }

    #[test]
    fn empty_allowed_rejected() {
        let applied = ["a"];
        assert_eq!(audit(&[], &applied), TaxonomyVerdict::InvalidConfig);
    }

    #[test]
    fn no_applied_tags_ok() {
        let allowed = ["a"];
        let v = audit(&allowed, &[]);
        if let TaxonomyVerdict::Ok {
            unknown_tags,
            duplicate_tags,
            histogram,
        } = v
        {
            assert!(unknown_tags.is_empty());
            assert!(duplicate_tags.is_empty());
            assert!(histogram.is_empty());
        }
    }

    #[test]
    fn unknown_tags_sorted() {
        let allowed = ["x"];
        let applied = ["zeta", "alpha", "mu"];
        let v = audit(&allowed, &applied);
        if let TaxonomyVerdict::Ok { unknown_tags, .. } = v {
            assert_eq!(
                unknown_tags,
                vec!["alpha".to_string(), "mu".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn unknown_collapses_duplicates() {
        let allowed = ["x"];
        let applied = ["zeta", "zeta", "zeta"];
        let v = audit(&allowed, &applied);
        if let TaxonomyVerdict::Ok { unknown_tags, .. } = v {
            assert_eq!(unknown_tags, vec!["zeta".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let allowed = ["a", "b"];
        let applied = ["a", "c"];
        let r1 = audit(&allowed, &applied);
        let r2 = audit(&allowed, &applied);
        assert_eq!(r1, r2);
    }

    #[test]
    fn case_sensitive_match() {
        let allowed = ["WASM"];
        let applied = ["wasm"];
        let v = audit(&allowed, &applied);
        if let TaxonomyVerdict::Ok { unknown_tags, .. } = v {
            assert_eq!(unknown_tags, vec!["wasm".to_string()]);
        }
    }

    #[test]
    fn duplicate_only_known_still_flagged() {
        let allowed = ["a"];
        let applied = ["a", "a"];
        let v = audit(&allowed, &applied);
        if let TaxonomyVerdict::Ok {
            unknown_tags,
            duplicate_tags,
            ..
        } = v
        {
            assert!(unknown_tags.is_empty());
            assert_eq!(duplicate_tags, vec!["a".to_string()]);
        }
    }
}
