//! # Monte-Carlo Caching with TTL Eviction
//!
//! Sim a cache where entries expire after `ttl_secs`. Random
//! arrivals + lookups. Reports hit rate and eviction count.
//!
//! Demonstrates the **MC.116** recipe for PMAT-197 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTTP cache TTL semantics (RFC 7234); Memcached LRU+TTL
//!  eviction.
//!
//! Run with: cargo run --example mc_caching_eviction_oldest
//!
//! Added by PMAT-197 (catalog 1396→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum CacheVerdict {
    Ok {
        hits: u32,
        misses: u32,
        evictions: u32,
        hit_rate: f64,
    },
    InvalidConfig,
}

pub fn simulate(seconds: u32, keyspace: u32, ttl_secs: u32, seed: u64) -> CacheVerdict {
    if seconds == 0 || keyspace == 0 || ttl_secs == 0 {
        return CacheVerdict::InvalidConfig;
    }
    let mut cache: BTreeMap<u32, u32> = BTreeMap::new(); // key → expiry_sec
    let mut hits = 0u32;
    let mut misses = 0u32;
    let mut evictions = 0u32;
    let mut rng_state = seed | 1;
    for sec in 0..seconds {
        let key = ((lcg(&mut rng_state) >> 32) as u32) % keyspace;
        // Evict expired.
        let to_remove: Vec<u32> = cache
            .iter()
            .filter(|(_, &exp)| exp <= sec)
            .map(|(k, _)| *k)
            .collect();
        for k in to_remove {
            cache.remove(&k);
            evictions += 1;
        }
        if let Entry::Vacant(slot) = cache.entry(key) {
            misses += 1;
            slot.insert(sec + ttl_secs);
        } else {
            hits += 1;
        }
    }
    let total = hits + misses;
    let hit_rate = if total > 0 {
        f64::from(hits) / f64::from(total)
    } else {
        0.0
    };
    CacheVerdict::Ok {
        hits,
        misses,
        evictions,
        hit_rate,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_caching_eviction_oldest")?;

    println!("hot small set: {:?}", simulate(10_000, 5, 100, 42));
    println!("wide cold set: {:?}", simulate(10_000, 100_000, 100, 42));
    println!("invalid: {:?}", simulate(0, 5, 100, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn hot_small_set_high_hit() {
        let v = simulate(10_000, 5, 1000, 42);
        if let CacheVerdict::Ok { hit_rate, .. } = v {
            assert!(hit_rate > 0.5);
        }
    }

    #[test]
    fn wide_cold_set_low_hit() {
        let v = simulate(1000, 100_000, 100, 42);
        if let CacheVerdict::Ok { hit_rate, .. } = v {
            assert!(hit_rate < 0.10);
        }
    }

    #[test]
    fn invalid_zero_seconds() {
        assert_eq!(simulate(0, 5, 100, 42), CacheVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_keyspace() {
        assert_eq!(simulate(1000, 0, 100, 42), CacheVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_ttl() {
        assert_eq!(simulate(1000, 5, 0, 42), CacheVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 10, 50, 42);
        let b = simulate(500, 10, 50, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn shorter_ttl_more_evictions() {
        let short = simulate(5000, 10, 1, 42);
        let long = simulate(5000, 10, 1000, 42);
        if let (CacheVerdict::Ok { evictions: s, .. }, CacheVerdict::Ok { evictions: l, .. }) =
            (short, long)
        {
            assert!(s > l);
        }
    }

    #[test]
    fn hits_plus_misses_equals_seconds() {
        let v = simulate(500, 10, 50, 42);
        if let CacheVerdict::Ok { hits, misses, .. } = v {
            assert_eq!(hits + misses, 500);
        }
    }

    #[test]
    fn rate_in_unit_range() {
        let v = simulate(500, 10, 50, 42);
        if let CacheVerdict::Ok { hit_rate, .. } = v {
            assert!((0.0..=1.0).contains(&hit_rate));
        }
    }

    #[test]
    fn first_query_always_miss() {
        let v = simulate(1, 100, 50, 42);
        if let CacheVerdict::Ok { hits, .. } = v {
            assert_eq!(hits, 0);
        }
    }

    #[test]
    fn longer_ttl_higher_hit() {
        let short = simulate(2000, 50, 5, 42);
        let long = simulate(2000, 50, 1000, 42);
        if let (CacheVerdict::Ok { hit_rate: s, .. }, CacheVerdict::Ok { hit_rate: l, .. }) =
            (short, long)
        {
            assert!(l >= s);
        }
    }
}
