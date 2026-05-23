//! # API Idempotency-Key Dedup
//!
//! POST requests can include an `Idempotency-Key` header. On retry,
//! return the prior response instead of re-executing. Algorithm:
//! - hash the key + endpoint into a cache slot
//! - if hit + within TTL → ReplayPriorResponse
//! - if hit + body differs (collision) → ConflictRejected
//! - if miss → AcceptNew (caller must store result)
//!
//! Demonstrates the **API.13** recipe for PMAT-143 (api round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Stripe / PayPal idempotency-key conventions.
//!
//! Run with: cargo run --example api_idempotency_key_dedup
//!
//! Added by PMAT-143 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashMap;
use std::hash::BuildHasher;

const TTL_SECS: u64 = 86_400;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedRequest {
    pub key: String,
    pub endpoint: String,
    pub body_hash: String,
    pub stored_at_secs: u64,
}

#[derive(Debug, PartialEq)]
pub enum DedupVerdict {
    AcceptNew,
    ReplayPriorResponse { stored_at_secs: u64 },
    ConflictRejected { stored_body_hash: String },
    EmptyKey,
    EmptyEndpoint,
    EmptyBodyHash,
    KeyExpired,
}

pub fn check<S: BuildHasher>(
    key: &str,
    endpoint: &str,
    body_hash: &str,
    cache: &HashMap<String, CachedRequest, S>,
    now_secs: u64,
) -> DedupVerdict {
    if key.is_empty() {
        return DedupVerdict::EmptyKey;
    }
    if endpoint.is_empty() {
        return DedupVerdict::EmptyEndpoint;
    }
    if body_hash.is_empty() {
        return DedupVerdict::EmptyBodyHash;
    }
    let slot = format!("{key}:{endpoint}");
    let Some(cached) = cache.get(&slot) else {
        return DedupVerdict::AcceptNew;
    };
    if now_secs.saturating_sub(cached.stored_at_secs) > TTL_SECS {
        return DedupVerdict::KeyExpired;
    }
    if cached.body_hash != body_hash {
        return DedupVerdict::ConflictRejected {
            stored_body_hash: cached.body_hash.clone(),
        };
    }
    DedupVerdict::ReplayPriorResponse {
        stored_at_secs: cached.stored_at_secs,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("api_idempotency_key_dedup")?;

    let mut cache = HashMap::new();
    cache.insert(
        "abc:/predict".to_string(),
        CachedRequest {
            key: "abc".to_string(),
            endpoint: "/predict".to_string(),
            body_hash: "hash1".to_string(),
            stored_at_secs: 1000,
        },
    );

    println!(
        "replay: {:?}",
        check("abc", "/predict", "hash1", &cache, 1500)
    );
    println!(
        "conflict: {:?}",
        check("abc", "/predict", "hash2", &cache, 1500)
    );
    println!("new: {:?}", check("xyz", "/predict", "hash3", &cache, 1500));
    println!(
        "expired: {:?}",
        check("abc", "/predict", "hash1", &cache, 90_000 + 1000)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache_with_one() -> HashMap<String, CachedRequest> {
        let mut c = HashMap::new();
        c.insert(
            "key:/predict".to_string(),
            CachedRequest {
                key: "key".to_string(),
                endpoint: "/predict".to_string(),
                body_hash: "hash1".to_string(),
                stored_at_secs: 1000,
            },
        );
        c
    }

    #[test]
    fn dedup_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn matching_replay() {
        let v = check("key", "/predict", "hash1", &cache_with_one(), 2000);
        assert!(matches!(v, DedupVerdict::ReplayPriorResponse { .. }));
    }

    #[test]
    fn collision_conflict() {
        let v = check("key", "/predict", "different", &cache_with_one(), 2000);
        assert!(matches!(v, DedupVerdict::ConflictRejected { .. }));
    }

    #[test]
    fn missing_accepts_new() {
        let v = check("newkey", "/predict", "hash1", &cache_with_one(), 2000);
        assert_eq!(v, DedupVerdict::AcceptNew);
    }

    #[test]
    fn expired_returns_expired() {
        let v = check(
            "key",
            "/predict",
            "hash1",
            &cache_with_one(),
            1000 + TTL_SECS + 1,
        );
        assert_eq!(v, DedupVerdict::KeyExpired);
    }

    #[test]
    fn empty_key_rejected() {
        assert_eq!(
            check("", "/predict", "h", &HashMap::new(), 100),
            DedupVerdict::EmptyKey
        );
    }

    #[test]
    fn empty_endpoint_rejected() {
        assert_eq!(
            check("k", "", "h", &HashMap::new(), 100),
            DedupVerdict::EmptyEndpoint
        );
    }

    #[test]
    fn empty_body_hash_rejected() {
        assert_eq!(
            check("k", "/p", "", &HashMap::new(), 100),
            DedupVerdict::EmptyBodyHash
        );
    }

    #[test]
    fn at_ttl_boundary_still_replays() {
        // Exactly TTL_SECS old → still valid.
        let v = check(
            "key",
            "/predict",
            "hash1",
            &cache_with_one(),
            1000 + TTL_SECS,
        );
        assert!(matches!(v, DedupVerdict::ReplayPriorResponse { .. }));
    }

    #[test]
    fn just_past_ttl_expires() {
        let v = check(
            "key",
            "/predict",
            "hash1",
            &cache_with_one(),
            1000 + TTL_SECS + 1,
        );
        assert_eq!(v, DedupVerdict::KeyExpired);
    }

    #[test]
    fn different_endpoint_treated_separately() {
        // Same key, different endpoint → AcceptNew.
        let v = check("key", "/different", "hash1", &cache_with_one(), 2000);
        assert_eq!(v, DedupVerdict::AcceptNew);
    }
}
