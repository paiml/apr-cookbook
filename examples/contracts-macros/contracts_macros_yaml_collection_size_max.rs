//! # Contracts-Macros YAML Collection Size Max
//!
//! Cap the size of YAML mappings/sequences to a configured maximum.
//! Returns sorted offending collection names that exceed the cap.
//!
//! Demonstrates the **CMM.200** recipe for PMAT-224 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: yamllint `key-duplicates` and `truthy` rules; PyYAML
//!  loader collection-size limits.
//!
//! Run with: cargo run --example contracts_macros_yaml_collection_size_max
//!
//! Added by PMAT-224 (catalog 1639→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CollectionSizeVerdict {
    Ok {
        offending: Vec<String>,
        largest_observed: u32,
    },
    InvalidConfig,
}

/// Items: (name, size). max_allowed is the cap.
pub fn check(items: &[(&str, u32)], max_allowed: u32) -> CollectionSizeVerdict {
    if items.is_empty() || max_allowed == 0 {
        return CollectionSizeVerdict::InvalidConfig;
    }
    let mut offenders: Vec<String> = items
        .iter()
        .filter(|(_, sz)| *sz > max_allowed)
        .map(|(name, _)| (*name).to_string())
        .collect();
    offenders.sort();
    let largest = items.iter().map(|(_, sz)| *sz).max().unwrap_or(0);
    CollectionSizeVerdict::Ok {
        offending: offenders,
        largest_observed: largest,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_collection_size_max")?;

    let items = [("colors", 10), ("tags", 200), ("aliases", 5)];
    println!("max-100: {:?}", check(&items, 100));
    println!("invalid: {:?}", check(&[], 100));
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
    fn within_cap_no_offender() {
        let v = check(&[("a", 50)], 100);
        if let CollectionSizeVerdict::Ok { offending, .. } = v {
            assert!(offending.is_empty());
        }
    }

    #[test]
    fn over_cap_flagged() {
        let v = check(&[("a", 200)], 100);
        if let CollectionSizeVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["a".to_string()]);
        }
    }

    #[test]
    fn at_cap_no_offender() {
        let v = check(&[("a", 100)], 100);
        if let CollectionSizeVerdict::Ok { offending, .. } = v {
            assert!(offending.is_empty());
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(&[], 100), CollectionSizeVerdict::InvalidConfig);
    }

    #[test]
    fn zero_max_rejected() {
        assert_eq!(check(&[("a", 50)], 0), CollectionSizeVerdict::InvalidConfig);
    }

    #[test]
    fn largest_observed_correct() {
        let v = check(&[("a", 10), ("b", 200), ("c", 50)], 100);
        if let CollectionSizeVerdict::Ok {
            largest_observed, ..
        } = v
        {
            assert_eq!(largest_observed, 200);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check(&[("a", 50)], 100);
        let r2 = check(&[("a", 50)], 100);
        assert_eq!(r1, r2);
    }

    #[test]
    fn offenders_sorted() {
        let v = check(&[("zeta", 200), ("alpha", 200)], 100);
        if let CollectionSizeVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<(&str, u32)> = (0..30).map(|_| ("c", 200)).collect();
        let v = check(&items, 100);
        if let CollectionSizeVerdict::Ok { offending, .. } = v {
            assert_eq!(offending.len(), 30);
        }
    }

    #[test]
    fn unicode_name_supported() {
        let v = check(&[("café", 200)], 100);
        if let CollectionSizeVerdict::Ok { offending, .. } = v {
            assert_eq!(offending, vec!["café".to_string()]);
        }
    }

    #[test]
    fn no_offenders_returns_empty() {
        let v = check(&[("a", 5), ("b", 10)], 100);
        if let CollectionSizeVerdict::Ok { offending, .. } = v {
            assert!(offending.is_empty());
        }
    }

    #[test]
    fn high_cap_handled() {
        let v = check(&[("a", 1_000_000)], u32::MAX);
        if let CollectionSizeVerdict::Ok { offending, .. } = v {
            assert!(offending.is_empty());
        }
    }
}
