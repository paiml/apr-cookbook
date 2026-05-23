//! # Contracts-Macros Obligation Owner Filter
//!
//! Filter obligations by owner pattern (case-insensitive substring).
//! Returns matched ids and unmatched count.
//!
//! Demonstrates the **CMM.100** recipe for PMAT-191 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GNU `grep -i` semantics; on-call rotation routing
//!  conventions (Google SRE workbook ch.14).
//!
//! Run with: cargo run --example contracts_macros_obligation_owner_filter
//!
//! Added by PMAT-191 (catalog 1342→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FilterVerdict {
    Ok {
        matched_ids: Vec<String>,
        unmatched_count: u32,
    },
    InvalidConfig,
}

pub fn filter(obligations: &[(&str, &str)], pattern: &str) -> FilterVerdict {
    if obligations.is_empty() || pattern.is_empty() {
        return FilterVerdict::InvalidConfig;
    }
    let p_lower = pattern.to_lowercase();
    let mut matched_ids: Vec<String> = Vec::new();
    let mut unmatched_count = 0u32;
    for (id, owner) in obligations {
        if owner.to_lowercase().contains(&p_lower) {
            matched_ids.push((*id).to_string());
        } else {
            unmatched_count += 1;
        }
    }
    matched_ids.sort();
    FilterVerdict::Ok {
        matched_ids,
        unmatched_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_owner_filter")?;

    let obligations = [
        ("o1", "alice@example.com"),
        ("o2", "bob@example.com"),
        ("o3", "alice@other.org"),
    ];
    println!("alice: {:?}", filter(&obligations, "alice"));
    println!("invalid: {:?}", filter(&[], ""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn pattern_matches_substring() {
        let obs = [("o1", "alice"), ("o2", "bob")];
        let v = filter(&obs, "ali");
        if let FilterVerdict::Ok { matched_ids, .. } = v {
            assert_eq!(matched_ids, vec!["o1".to_string()]);
        }
    }

    #[test]
    fn case_insensitive_match() {
        let obs = [("o1", "ALICE")];
        let v = filter(&obs, "alice");
        if let FilterVerdict::Ok { matched_ids, .. } = v {
            assert_eq!(matched_ids, vec!["o1".to_string()]);
        }
    }

    #[test]
    fn no_match_unmatched_count() {
        let obs = [("o1", "alice"), ("o2", "bob")];
        let v = filter(&obs, "carol");
        if let FilterVerdict::Ok {
            matched_ids,
            unmatched_count,
        } = v
        {
            assert!(matched_ids.is_empty());
            assert_eq!(unmatched_count, 2);
        }
    }

    #[test]
    fn empty_obligations_rejected() {
        assert_eq!(filter(&[], "a"), FilterVerdict::InvalidConfig);
    }

    #[test]
    fn empty_pattern_rejected() {
        let obs = [("o", "a")];
        assert_eq!(filter(&obs, ""), FilterVerdict::InvalidConfig);
    }

    #[test]
    fn matched_ids_sorted() {
        let obs = [("zeta", "alice"), ("alpha", "alice")];
        let v = filter(&obs, "alice");
        if let FilterVerdict::Ok { matched_ids, .. } = v {
            assert_eq!(matched_ids, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let obs = [("o", "alice")];
        let r1 = filter(&obs, "alice");
        let r2 = filter(&obs, "alice");
        assert_eq!(r1, r2);
    }

    #[test]
    fn full_match_works() {
        let obs = [("o", "alice@example.com")];
        let v = filter(&obs, "alice@example.com");
        if let FilterVerdict::Ok { matched_ids, .. } = v {
            assert_eq!(matched_ids, vec!["o".to_string()]);
        }
    }

    #[test]
    fn unicode_owner_supported() {
        let obs = [("o", "café@example.com")];
        let v = filter(&obs, "café");
        if let FilterVerdict::Ok { matched_ids, .. } = v {
            assert_eq!(matched_ids, vec!["o".to_string()]);
        }
    }

    #[test]
    fn matched_plus_unmatched_equals_total() {
        let obs = [("o1", "alice"), ("o2", "bob"), ("o3", "alice")];
        let v = filter(&obs, "alice");
        if let FilterVerdict::Ok {
            matched_ids,
            unmatched_count,
        } = v
        {
            assert_eq!(matched_ids.len() + unmatched_count as usize, 3);
        }
    }

    #[test]
    fn many_matches() {
        let obs: Vec<(&str, &str)> = (0..20).map(|_| ("o", "alice")).collect();
        let v = filter(&obs, "alice");
        if let FilterVerdict::Ok { matched_ids, .. } = v {
            assert_eq!(matched_ids.len(), 20);
        }
    }
}
