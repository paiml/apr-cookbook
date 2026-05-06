//! # Contracts-Macros Obligation Dedupe
//!
//! Collapse identical obligations across recipes — same `(kind, expr)`
//! pair counts as one canonical obligation. Returns unique set,
//! dedup count, and per-canonical mention count.
//!
//! Demonstrates the **CMM.76** recipe for PMAT-183 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: structural sharing in Hash Array Mapped Tries (Bagwell
//!  2001); BOM dedup in build systems.
//!
//! Run with: cargo run --example contracts_macros_obligation_dedupe
//!
//! Added by PMAT-183 (catalog 1270→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum DedupeVerdict {
    Ok {
        unique_count: u32,
        dedup_count: u32,
        mention_counts: BTreeMap<String, u32>,
    },
    InvalidConfig,
}

pub fn dedupe(obligations: &[(&str, &str)]) -> DedupeVerdict {
    if obligations.is_empty() {
        return DedupeVerdict::InvalidConfig;
    }
    let mut counts: BTreeMap<String, u32> = BTreeMap::new();
    for (kind, expr) in obligations {
        let key = format!("{kind}::{expr}");
        *counts.entry(key).or_insert(0) += 1;
    }
    let unique_count = counts.len() as u32;
    let total = obligations.len() as u32;
    DedupeVerdict::Ok {
        unique_count,
        dedup_count: total - unique_count,
        mention_counts: counts,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_dedupe")?;

    let obligations = [
        ("pre", "x > 0"),
        ("post", "y > 0"),
        ("pre", "x > 0"),
        ("pre", "x > 0"),
    ];
    println!("dedupe: {:?}", dedupe(&obligations));
    println!("invalid: {:?}", dedupe(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deduper_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_duplicates_zero_dedup() {
        let obligations = [("pre", "a"), ("post", "b")];
        let v = dedupe(&obligations);
        if let DedupeVerdict::Ok { dedup_count, .. } = v {
            assert_eq!(dedup_count, 0);
        }
    }

    #[test]
    fn all_duplicates_collapse_to_one() {
        let obligations = [("pre", "x"), ("pre", "x"), ("pre", "x")];
        let v = dedupe(&obligations);
        if let DedupeVerdict::Ok {
            unique_count,
            dedup_count,
            ..
        } = v
        {
            assert_eq!(unique_count, 1);
            assert_eq!(dedup_count, 2);
        }
    }

    #[test]
    fn different_kind_same_expr_not_dup() {
        let obligations = [("pre", "x"), ("post", "x")];
        let v = dedupe(&obligations);
        if let DedupeVerdict::Ok { unique_count, .. } = v {
            assert_eq!(unique_count, 2);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(dedupe(&[]), DedupeVerdict::InvalidConfig);
    }

    #[test]
    fn mention_counts_correct() {
        let obligations = [("pre", "x"), ("pre", "x"), ("post", "y")];
        let v = dedupe(&obligations);
        if let DedupeVerdict::Ok { mention_counts, .. } = v {
            assert_eq!(mention_counts.get("pre::x"), Some(&2));
            assert_eq!(mention_counts.get("post::y"), Some(&1));
        }
    }

    #[test]
    fn unique_plus_dedup_equals_total() {
        let obligations = [("pre", "a"), ("pre", "a"), ("post", "b")];
        let v = dedupe(&obligations);
        if let DedupeVerdict::Ok {
            unique_count,
            dedup_count,
            ..
        } = v
        {
            assert_eq!(unique_count + dedup_count, 3);
        }
    }

    #[test]
    fn deterministic() {
        let obligations = [("pre", "x")];
        let r1 = dedupe(&obligations);
        let r2 = dedupe(&obligations);
        assert_eq!(r1, r2);
    }

    #[test]
    fn large_input_works() {
        let obligations: Vec<(&str, &str)> = (0..100).map(|_| ("pre", "x")).collect();
        let v = dedupe(&obligations);
        if let DedupeVerdict::Ok {
            unique_count,
            dedup_count,
            ..
        } = v
        {
            assert_eq!(unique_count, 1);
            assert_eq!(dedup_count, 99);
        }
    }

    #[test]
    fn mention_counts_keys_use_separator() {
        let obligations = [("pre", "x")];
        let v = dedupe(&obligations);
        if let DedupeVerdict::Ok { mention_counts, .. } = v {
            assert!(mention_counts.contains_key("pre::x"));
        }
    }

    #[test]
    fn unicode_obligation_supported() {
        let obligations = [("pre", "café > 0")];
        let v = dedupe(&obligations);
        if let DedupeVerdict::Ok { unique_count, .. } = v {
            assert_eq!(unique_count, 1);
        }
    }

    #[test]
    fn whitespace_distinguishes() {
        // "x > 0" vs " x > 0" are different.
        let obligations = [("pre", "x > 0"), ("pre", " x > 0")];
        let v = dedupe(&obligations);
        if let DedupeVerdict::Ok { unique_count, .. } = v {
            assert_eq!(unique_count, 2);
        }
    }
}
