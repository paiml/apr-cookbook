//! # Advanced In-Flight Request Dedup Cache
//!
//! Two clients send identical requests in rapid succession. Server
//! deduplicates by request_hash: second waits for first's response.
//! This recipe is the bookkeeping logic, not the wait mechanism.
//!
//! Demonstrates the **ADV.29** recipe for PMAT-155 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Singleflight (Go stdlib) + Cloudflare cache coalescing.
//!
//! Run with: cargo run --example adv_request_dedup_cache
//!
//! Added by PMAT-155 (catalog 1018→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum DedupVerdict {
    NewRequest,
    AttachedToInflight { wait_count: u32 },
    InvalidHash,
}

pub fn classify(request_hash: &str, inflight: &BTreeMap<String, u32>) -> DedupVerdict {
    if request_hash.is_empty() {
        return DedupVerdict::InvalidHash;
    }
    match inflight.get(request_hash) {
        Some(&count) => DedupVerdict::AttachedToInflight {
            wait_count: count + 1,
        },
        None => DedupVerdict::NewRequest,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_request_dedup_cache")?;

    let mut inflight = BTreeMap::new();
    inflight.insert("hash_a".to_string(), 0);
    inflight.insert("hash_b".to_string(), 3);

    println!("new: {:?}", classify("hash_c", &inflight));
    println!("dedup: {:?}", classify("hash_a", &inflight));
    println!("dedup hot: {:?}", classify("hash_b", &inflight));
    println!("invalid: {:?}", classify("", &inflight));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build() -> BTreeMap<String, u32> {
        let mut m = BTreeMap::new();
        m.insert("a".to_string(), 0);
        m.insert("b".to_string(), 5);
        m
    }

    #[test]
    fn dedup_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn new_request_classified() {
        let v = classify("c", &build());
        assert_eq!(v, DedupVerdict::NewRequest);
    }

    #[test]
    fn existing_request_attached() {
        let v = classify("a", &build());
        assert!(matches!(v, DedupVerdict::AttachedToInflight { .. }));
    }

    #[test]
    fn empty_hash_rejected() {
        assert_eq!(classify("", &build()), DedupVerdict::InvalidHash);
    }

    #[test]
    fn wait_count_increments() {
        let v = classify("b", &build());
        if let DedupVerdict::AttachedToInflight { wait_count } = v {
            // b had 5 waiters; new request → 6.
            assert_eq!(wait_count, 6);
        }
    }

    #[test]
    fn first_attached_increments_zero() {
        let v = classify("a", &build());
        if let DedupVerdict::AttachedToInflight { wait_count } = v {
            assert_eq!(wait_count, 1);
        }
    }

    #[test]
    fn empty_inflight_new() {
        let empty: BTreeMap<String, u32> = BTreeMap::new();
        let v = classify("a", &empty);
        assert_eq!(v, DedupVerdict::NewRequest);
    }

    #[test]
    fn case_sensitive() {
        let mut m = BTreeMap::new();
        m.insert("hash".to_string(), 0);
        let v = classify("HASH", &m);
        assert_eq!(v, DedupVerdict::NewRequest);
    }

    #[test]
    fn long_hash_works() {
        let long = "a".repeat(64);
        let mut m = BTreeMap::new();
        m.insert(long.clone(), 2);
        let v = classify(&long, &m);
        assert!(matches!(v, DedupVerdict::AttachedToInflight { .. }));
    }

    #[test]
    fn many_inflight_works() {
        let mut m = BTreeMap::new();
        for i in 0..1000 {
            m.insert(format!("h{i}"), i);
        }
        let v = classify("h500", &m);
        if let DedupVerdict::AttachedToInflight { wait_count } = v {
            assert_eq!(wait_count, 501);
        }
    }

    #[test]
    fn deterministic() {
        let m = build();
        let a = classify("a", &m);
        let b = classify("a", &m);
        assert_eq!(a, b);
    }
}
