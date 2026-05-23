//! # Contracts-Macros Witness Origin Trace
//!
//! Track which provenance source each witness came from (PR/CI/manual)
//! and aggregate counts. Returns counts per origin and the list of
//! witnesses without recorded origin.
//!
//! Demonstrates the **CMM.165** recipe for PMAT-212 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SLSA provenance levels (build-source attestation); in-toto
//!  attestation predicate format.
//!
//! Run with: cargo run --example contracts_macros_witness_origin_trace
//!
//! Added by PMAT-212 (catalog 1531→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum OriginVerdict {
    Ok {
        counts_by_origin: BTreeMap<String, u32>,
        unknown_origin_ids: Vec<String>,
    },
    InvalidConfig,
}

/// Items: (id, origin) where empty origin = unknown.
pub fn trace(items: &[(&str, &str)]) -> OriginVerdict {
    if items.is_empty() {
        return OriginVerdict::InvalidConfig;
    }
    let mut counts: BTreeMap<String, u32> = BTreeMap::new();
    let mut unknown: Vec<String> = Vec::new();
    for (id, origin) in items {
        if origin.is_empty() {
            unknown.push((*id).to_string());
        } else {
            *counts.entry((*origin).to_string()).or_insert(0) += 1;
        }
    }
    unknown.sort();
    OriginVerdict::Ok {
        counts_by_origin: counts,
        unknown_origin_ids: unknown,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_witness_origin_trace")?;

    let items = [("w1", "PR"), ("w2", "CI"), ("w3", "PR"), ("w4", "")];
    println!("trace: {:?}", trace(&items));
    println!("invalid: {:?}", trace(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn count_per_origin() {
        let v = trace(&[("a", "PR"), ("b", "PR"), ("c", "CI")]);
        if let OriginVerdict::Ok {
            counts_by_origin, ..
        } = v
        {
            assert_eq!(counts_by_origin.get("PR"), Some(&2));
            assert_eq!(counts_by_origin.get("CI"), Some(&1));
        }
    }

    #[test]
    fn empty_origin_in_unknown_list() {
        let v = trace(&[("a", ""), ("b", "PR")]);
        if let OriginVerdict::Ok {
            unknown_origin_ids, ..
        } = v
        {
            assert_eq!(unknown_origin_ids, vec!["a".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(trace(&[]), OriginVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = trace(&[("a", "PR")]);
        let r2 = trace(&[("a", "PR")]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn unknown_sorted() {
        let v = trace(&[("zeta", ""), ("alpha", "")]);
        if let OriginVerdict::Ok {
            unknown_origin_ids, ..
        } = v
        {
            assert_eq!(
                unknown_origin_ids,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn no_unknown_returns_empty() {
        let v = trace(&[("a", "PR"), ("b", "CI")]);
        if let OriginVerdict::Ok {
            unknown_origin_ids, ..
        } = v
        {
            assert!(unknown_origin_ids.is_empty());
        }
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<(&str, &str)> = (0..30).map(|_| ("w", "PR")).collect();
        let v = trace(&items);
        if let OriginVerdict::Ok {
            counts_by_origin, ..
        } = v
        {
            assert_eq!(counts_by_origin.get("PR"), Some(&30));
        }
    }

    #[test]
    fn unicode_origin_supported() {
        let v = trace(&[("a", "café")]);
        if let OriginVerdict::Ok {
            counts_by_origin, ..
        } = v
        {
            assert_eq!(counts_by_origin.get("café"), Some(&1));
        }
    }

    #[test]
    fn case_sensitive_origin() {
        let v = trace(&[("a", "PR"), ("b", "pr")]);
        if let OriginVerdict::Ok {
            counts_by_origin, ..
        } = v
        {
            assert_eq!(counts_by_origin.len(), 2);
        }
    }

    #[test]
    fn single_item_handled() {
        let v = trace(&[("w", "PR")]);
        if let OriginVerdict::Ok {
            counts_by_origin, ..
        } = v
        {
            assert_eq!(counts_by_origin.len(), 1);
        }
    }

    #[test]
    fn mixed_known_unknown() {
        let v = trace(&[("a", "PR"), ("b", ""), ("c", "CI")]);
        if let OriginVerdict::Ok {
            counts_by_origin,
            unknown_origin_ids,
        } = v
        {
            assert_eq!(counts_by_origin.len(), 2);
            assert_eq!(unknown_origin_ids.len(), 1);
        }
    }
}
