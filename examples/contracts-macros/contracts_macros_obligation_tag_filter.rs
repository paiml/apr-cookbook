//! # Contracts-Macros Obligation Tag Filter
//!
//! Filter obligations by tag predicates (intersect / union / not).
//! Useful when running a subset of contract checks for a specific
//! release category.
//!
//! Demonstrates the **CMM.36** recipe for PMAT-169 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: pytest -k expression syntax.
//!
//! Run with: cargo run --example contracts_macros_obligation_tag_filter
//!
//! Added by PMAT-169 (catalog 1144→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode {
    All,
    Any,
    None,
}

#[derive(Debug, PartialEq)]
pub enum FilterVerdict {
    Ok { matched: Vec<String> },
    EmptyObligations,
}

pub fn filter(
    obligations: &[(&str, Vec<&str>)],
    selector: &[&str],
    mode: FilterMode,
) -> FilterVerdict {
    if obligations.is_empty() {
        return FilterVerdict::EmptyObligations;
    }
    let selector_set: BTreeSet<&str> = selector.iter().copied().collect();
    let mut matched = Vec::new();
    for (name, tags) in obligations {
        let tag_set: BTreeSet<&str> = tags.iter().copied().collect();
        let intersect: usize = selector_set.intersection(&tag_set).count();
        let included = match mode {
            FilterMode::All => intersect == selector_set.len(),
            FilterMode::Any => intersect > 0 || selector_set.is_empty(),
            FilterMode::None => intersect == 0,
        };
        if included {
            matched.push((*name).to_string());
        }
    }
    FilterVerdict::Ok { matched }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_tag_filter")?;

    let oblig = vec![
        ("o1", vec!["safety"]),
        ("o2", vec!["performance"]),
        ("o3", vec!["safety", "performance"]),
    ];
    println!(
        "any safety: {:?}",
        filter(&oblig, &["safety"], FilterMode::Any)
    );
    println!(
        "all safety+perf: {:?}",
        filter(&oblig, &["safety", "performance"], FilterMode::All)
    );
    println!(
        "none with safety: {:?}",
        filter(&oblig, &["safety"], FilterMode::None)
    );
    println!("empty: {:?}", filter(&[], &["x"], FilterMode::Any));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typical() -> Vec<(&'static str, Vec<&'static str>)> {
        vec![
            ("o1", vec!["safety"]),
            ("o2", vec!["performance"]),
            ("o3", vec!["safety", "performance"]),
        ]
    }

    #[test]
    fn filter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn any_mode_picks_overlap() {
        let v = filter(&typical(), &["safety"], FilterMode::Any);
        if let FilterVerdict::Ok { matched } = v {
            assert_eq!(matched.len(), 2);
        }
    }

    #[test]
    fn all_mode_requires_full_match() {
        let v = filter(&typical(), &["safety", "performance"], FilterMode::All);
        if let FilterVerdict::Ok { matched } = v {
            assert_eq!(matched, vec!["o3".to_string()]);
        }
    }

    #[test]
    fn none_mode_excludes_match() {
        let v = filter(&typical(), &["safety"], FilterMode::None);
        if let FilterVerdict::Ok { matched } = v {
            assert_eq!(matched, vec!["o2".to_string()]);
        }
    }

    #[test]
    fn empty_obligations_special() {
        assert_eq!(
            filter(&[], &["x"], FilterMode::Any),
            FilterVerdict::EmptyObligations
        );
    }

    #[test]
    fn empty_selector_any_includes_all() {
        let v = filter(&typical(), &[], FilterMode::Any);
        if let FilterVerdict::Ok { matched } = v {
            assert_eq!(matched.len(), 3);
        }
    }

    #[test]
    fn empty_selector_all_includes_all() {
        let v = filter(&typical(), &[], FilterMode::All);
        if let FilterVerdict::Ok { matched } = v {
            // All-mode with empty selector: every set has empty subset.
            assert_eq!(matched.len(), 3);
        }
    }

    #[test]
    fn empty_selector_none_includes_all() {
        let v = filter(&typical(), &[], FilterMode::None);
        if let FilterVerdict::Ok { matched } = v {
            // None-of-empty: nothing excluded.
            assert_eq!(matched.len(), 3);
        }
    }

    #[test]
    fn no_match_empty_result() {
        let v = filter(&typical(), &["security"], FilterMode::Any);
        if let FilterVerdict::Ok { matched } = v {
            assert!(matched.is_empty());
        }
    }

    #[test]
    fn untagged_obligation_excluded() {
        let oblig = vec![("untagged", vec![])];
        let v = filter(&oblig, &["safety"], FilterMode::Any);
        if let FilterVerdict::Ok { matched } = v {
            assert!(matched.is_empty());
        }
    }

    #[test]
    fn deterministic() {
        let o = typical();
        let a = filter(&o, &["safety"], FilterMode::Any);
        let b = filter(&o, &["safety"], FilterMode::Any);
        assert_eq!(a, b);
    }
}
