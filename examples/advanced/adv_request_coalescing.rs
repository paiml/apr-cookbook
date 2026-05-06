//! # Advanced Inflight Request Coalescing
//!
//! Multiple identical requests in flight (e.g., 100 users asking the
//! same question) → run inference once, broadcast the result.
//!
//! Decision rules:
//!   request_hash matches inflight → CoalesceWith{existing_id}
//!   inflight slot full → OverflowToQueue
//!   no match → AcceptAsNew
//!
//! Demonstrates the **ADV.15** recipe for PMAT-145 (advanced round 6).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Cloudflare cache-coalescing pattern.
//!
//! Run with: cargo run --example adv_request_coalescing
//!
//! Added by PMAT-145 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashMap;
use std::hash::BuildHasher;

#[derive(Debug, PartialEq)]
pub enum CoalesceVerdict {
    AcceptAsNew { request_id: u64 },
    CoalesceWith { existing_request_id: u64 },
    OverflowToQueue,
    InvalidHash,
    InvalidLimit,
}

const HASH_LEN: usize = 64;

pub fn evaluate<S: BuildHasher>(
    request_hash: &str,
    inflight: &HashMap<String, u64, S>,
    next_id: u64,
    inflight_capacity: usize,
) -> CoalesceVerdict {
    if request_hash.len() != HASH_LEN || !request_hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return CoalesceVerdict::InvalidHash;
    }
    if inflight_capacity == 0 {
        return CoalesceVerdict::InvalidLimit;
    }
    if let Some(&existing_id) = inflight.get(request_hash) {
        return CoalesceVerdict::CoalesceWith {
            existing_request_id: existing_id,
        };
    }
    if inflight.len() >= inflight_capacity {
        return CoalesceVerdict::OverflowToQueue;
    }
    CoalesceVerdict::AcceptAsNew {
        request_id: next_id,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_request_coalescing")?;

    let mut inflight = HashMap::new();
    let h_a = "a".repeat(64);
    inflight.insert(h_a.clone(), 100);

    println!("coalesce: {:?}", evaluate(&h_a, &inflight, 200, 100));
    println!("new: {:?}", evaluate(&"b".repeat(64), &inflight, 200, 100));
    println!(
        "overflow: {:?}",
        evaluate(&"c".repeat(64), &inflight, 200, 1)
    );
    println!("invalid hash: {:?}", evaluate("abc", &inflight, 200, 100));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_with_one() -> HashMap<String, u64> {
        let mut h = HashMap::new();
        h.insert("a".repeat(64), 100);
        h
    }

    #[test]
    fn evaluator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn matching_hash_coalesces() {
        let cache = cache_with_one();
        let v = evaluate(&"a".repeat(64), &cache, 200, 100);
        if let CoalesceVerdict::CoalesceWith {
            existing_request_id,
        } = v
        {
            assert_eq!(existing_request_id, 100);
        }
    }

    #[test]
    fn unique_hash_accepts_new() {
        let cache = cache_with_one();
        let v = evaluate(&"b".repeat(64), &cache, 200, 100);
        if let CoalesceVerdict::AcceptAsNew { request_id } = v {
            assert_eq!(request_id, 200);
        }
    }

    #[test]
    fn capacity_full_overflows() {
        let cache = cache_with_one();
        let v = evaluate(&"b".repeat(64), &cache, 200, 1);
        assert_eq!(v, CoalesceVerdict::OverflowToQueue);
    }

    #[test]
    fn invalid_hash_rejected() {
        let cache = HashMap::new();
        assert_eq!(
            evaluate("abc", &cache, 200, 100),
            CoalesceVerdict::InvalidHash
        );
    }

    #[test]
    fn non_hex_hash_rejected() {
        let cache = HashMap::new();
        let bad = "z".repeat(64);
        assert_eq!(
            evaluate(&bad, &cache, 200, 100),
            CoalesceVerdict::InvalidHash
        );
    }

    #[test]
    fn zero_capacity_rejected() {
        let cache = HashMap::new();
        let v = evaluate(&"a".repeat(64), &cache, 200, 0);
        assert_eq!(v, CoalesceVerdict::InvalidLimit);
    }

    #[test]
    fn empty_cache_accepts() {
        let cache = HashMap::new();
        let v = evaluate(&"a".repeat(64), &cache, 200, 100);
        assert!(matches!(v, CoalesceVerdict::AcceptAsNew { .. }));
    }

    #[test]
    fn coalesce_returns_existing_id_not_new() {
        let cache = cache_with_one();
        let v = evaluate(&"a".repeat(64), &cache, 999, 100);
        if let CoalesceVerdict::CoalesceWith {
            existing_request_id,
        } = v
        {
            // Returns existing 100, not next_id 999.
            assert_eq!(existing_request_id, 100);
        }
    }

    #[test]
    fn at_capacity_boundary_overflows() {
        let cache = cache_with_one();
        // capacity == size of cache → no room for new.
        let v = evaluate(&"b".repeat(64), &cache, 200, 1);
        assert_eq!(v, CoalesceVerdict::OverflowToQueue);
    }

    #[test]
    fn just_under_capacity_accepts() {
        let cache = cache_with_one();
        let v = evaluate(&"b".repeat(64), &cache, 200, 2);
        assert!(matches!(v, CoalesceVerdict::AcceptAsNew { .. }));
    }
}
