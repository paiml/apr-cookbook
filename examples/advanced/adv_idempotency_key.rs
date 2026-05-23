//! # Advanced Idempotency Key Dedup
//!
//! Client provides Idempotency-Key header; server caches response by
//! key for TTL_seconds. Repeat requests with same key → return cached
//! response, do not re-execute. Returns ReplayCached or NewRequest.
//!
//! Demonstrates the **ADV.39** recipe for PMAT-158 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Stripe Idempotency-Key header (24h TTL).
//!
//! Run with: cargo run --example adv_idempotency_key
//!
//! Added by PMAT-158 (catalog 1045→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CachedEntry {
    pub key: String,
    pub created_at_secs: u64,
    pub response_hash: String,
}

#[derive(Debug, PartialEq)]
pub enum IdemVerdict {
    NewRequest,
    ReplayCached {
        response_hash: String,
        age_secs: u64,
    },
    KeyExpired {
        age_secs: u64,
        ttl_secs: u64,
    },
    InvalidKey,
}

pub fn classify(
    request_key: &str,
    cache: &[CachedEntry],
    now_secs: u64,
    ttl_secs: u64,
) -> IdemVerdict {
    if request_key.is_empty() || ttl_secs == 0 {
        return IdemVerdict::InvalidKey;
    }
    let Some(entry) = cache.iter().find(|e| e.key == request_key) else {
        return IdemVerdict::NewRequest;
    };
    let age_secs = now_secs.saturating_sub(entry.created_at_secs);
    if age_secs > ttl_secs {
        IdemVerdict::KeyExpired { age_secs, ttl_secs }
    } else {
        IdemVerdict::ReplayCached {
            response_hash: entry.response_hash.clone(),
            age_secs,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_idempotency_key")?;

    let cache = vec![
        CachedEntry {
            key: "key_a".to_string(),
            created_at_secs: 1000,
            response_hash: "hash_a".to_string(),
        },
        CachedEntry {
            key: "key_b".to_string(),
            created_at_secs: 100,
            response_hash: "hash_b".to_string(),
        },
    ];
    let ttl = 86_400;
    println!("new: {:?}", classify("key_c", &cache, 2000, ttl));
    println!("replay: {:?}", classify("key_a", &cache, 2000, ttl));
    println!("expired: {:?}", classify("key_b", &cache, 1_000_000, ttl));
    println!("invalid: {:?}", classify("", &cache, 2000, ttl));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cache() -> Vec<CachedEntry> {
        vec![CachedEntry {
            key: "k".to_string(),
            created_at_secs: 1000,
            response_hash: "h".to_string(),
        }]
    }

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn unknown_key_new_request() {
        let v = classify("unknown", &cache(), 2000, 86_400);
        assert_eq!(v, IdemVerdict::NewRequest);
    }

    #[test]
    fn known_within_ttl_replay() {
        let v = classify("k", &cache(), 2000, 86_400);
        assert!(matches!(v, IdemVerdict::ReplayCached { .. }));
    }

    #[test]
    fn known_past_ttl_expired() {
        let v = classify("k", &cache(), 1_000_000, 1000);
        assert!(matches!(v, IdemVerdict::KeyExpired { .. }));
    }

    #[test]
    fn empty_key_invalid() {
        assert_eq!(
            classify("", &cache(), 2000, 86_400),
            IdemVerdict::InvalidKey
        );
    }

    #[test]
    fn zero_ttl_invalid() {
        assert_eq!(classify("k", &cache(), 2000, 0), IdemVerdict::InvalidKey);
    }

    #[test]
    fn empty_cache_new_request() {
        let v = classify("k", &[], 2000, 86_400);
        assert_eq!(v, IdemVerdict::NewRequest);
    }

    #[test]
    fn now_before_creation_zero_age() {
        // Saturating-sub gives zero age if now < created.
        let v = classify("k", &cache(), 500, 86_400);
        if let IdemVerdict::ReplayCached { age_secs, .. } = v {
            assert_eq!(age_secs, 0);
        }
    }

    #[test]
    fn boundary_at_ttl_replays() {
        // age == ttl → still within (not >).
        let v = classify("k", &cache(), 1000 + 86_400, 86_400);
        assert!(matches!(v, IdemVerdict::ReplayCached { .. }));
    }

    #[test]
    fn just_past_ttl_expired() {
        let v = classify("k", &cache(), 1000 + 86_401, 86_400);
        assert!(matches!(v, IdemVerdict::KeyExpired { .. }));
    }

    #[test]
    fn response_hash_returned() {
        let v = classify("k", &cache(), 2000, 86_400);
        if let IdemVerdict::ReplayCached { response_hash, .. } = v {
            assert_eq!(response_hash, "h");
        }
    }

    #[test]
    fn deterministic() {
        let c = cache();
        let a = classify("k", &c, 2000, 86_400);
        let b = classify("k", &c, 2000, 86_400);
        assert_eq!(a, b);
    }
}
