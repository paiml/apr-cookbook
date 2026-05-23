//! # Contracts-Macros YAML Canonical Form
//!
//! Canonicalize a YAML key-value list: sort keys ascending, dedup
//! by key (last write wins), strip surrounding whitespace from
//! values. Returns the canonical key-value list.
//!
//! Demonstrates the **CMM.157** recipe for PMAT-210 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 §10 canonical form; jq sort_keys + tojson
//!  output mode.
//!
//! Run with: cargo run --example contracts_macros_yaml_canonical_form
//!
//! Added by PMAT-210 (catalog 1513→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum CanonicalVerdict {
    Ok {
        canonical: Vec<(String, String)>,
        deduped_count: u32,
    },
    InvalidConfig,
}

pub fn canonicalize(items: &[(&str, &str)]) -> CanonicalVerdict {
    if items.is_empty() {
        return CanonicalVerdict::InvalidConfig;
    }
    let mut map: BTreeMap<String, String> = BTreeMap::new();
    let mut input_count = 0u32;
    for (k, v) in items {
        if k.is_empty() {
            return CanonicalVerdict::InvalidConfig;
        }
        input_count += 1;
        map.insert((*k).to_string(), v.trim().to_string());
    }
    let canonical: Vec<(String, String)> = map.into_iter().collect();
    let deduped = input_count - canonical.len() as u32;
    CanonicalVerdict::Ok {
        canonical,
        deduped_count: deduped,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_canonical_form")?;

    let items = [("zeta", "  z  "), ("alpha", "a"), ("zeta", "Z")];
    println!("canonical: {:?}", canonicalize(&items));
    println!("invalid: {:?}", canonicalize(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn keys_sorted_ascending() {
        let v = canonicalize(&[("zeta", "z"), ("alpha", "a")]);
        if let CanonicalVerdict::Ok { canonical, .. } = v {
            assert_eq!(canonical[0].0, "alpha");
            assert_eq!(canonical[1].0, "zeta");
        }
    }

    #[test]
    fn duplicate_key_last_wins() {
        let v = canonicalize(&[("k", "first"), ("k", "second")]);
        if let CanonicalVerdict::Ok { canonical, .. } = v {
            assert_eq!(canonical[0].1, "second");
        }
    }

    #[test]
    fn whitespace_trimmed() {
        let v = canonicalize(&[("k", "  trimmed  ")]);
        if let CanonicalVerdict::Ok { canonical, .. } = v {
            assert_eq!(canonical[0].1, "trimmed");
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(canonicalize(&[]), CanonicalVerdict::InvalidConfig);
    }

    #[test]
    fn empty_key_rejected() {
        assert_eq!(canonicalize(&[("", "v")]), CanonicalVerdict::InvalidConfig);
    }

    #[test]
    fn deduped_count_correct() {
        let v = canonicalize(&[("a", "1"), ("a", "2"), ("b", "3")]);
        if let CanonicalVerdict::Ok { deduped_count, .. } = v {
            assert_eq!(deduped_count, 1);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = canonicalize(&[("k", "v")]);
        let r2 = canonicalize(&[("k", "v")]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn no_duplicates_zero_deduped() {
        let v = canonicalize(&[("a", "1"), ("b", "2")]);
        if let CanonicalVerdict::Ok { deduped_count, .. } = v {
            assert_eq!(deduped_count, 0);
        }
    }

    #[test]
    fn case_sensitive_keys() {
        let v = canonicalize(&[("K", "1"), ("k", "2")]);
        if let CanonicalVerdict::Ok { canonical, .. } = v {
            assert_eq!(canonical.len(), 2);
        }
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<(&str, &str)> = (0..30).map(|_| ("k", "v")).collect();
        let v = canonicalize(&items);
        if let CanonicalVerdict::Ok { canonical, .. } = v {
            assert_eq!(canonical.len(), 1);
        }
    }

    #[test]
    fn unicode_key_supported() {
        let v = canonicalize(&[("café", "value")]);
        if let CanonicalVerdict::Ok { canonical, .. } = v {
            assert_eq!(canonical[0].0, "café");
        }
    }

    #[test]
    fn empty_value_preserved() {
        let v = canonicalize(&[("k", "")]);
        if let CanonicalVerdict::Ok { canonical, .. } = v {
            assert_eq!(canonical[0].1, "");
        }
    }
}
