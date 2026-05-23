//! # Monte-Carlo KV Cache Hit Rate Under Zipf Distribution
//!
//! Sim cache hit rate when access pattern follows Zipf with parameter
//! `alpha`. Higher alpha → more skewed distribution → higher hit rate
//! for same cache size.
//!
//! Demonstrates the **MC.63** recipe for PMAT-178 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Breslau et al. (1999) Web caching workload Zipf-like.
//!
//! Run with: cargo run --example mc_kv_zipf_hit_rate
//!
//! Added by PMAT-178 (catalog 1225→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum ZipfVerdict {
    Ok {
        hit_rate: f64,
        unique_keys_seen: u32,
    },
    InvalidConfig,
}

pub fn simulate(
    cache_size: u32,
    keyspace: u32,
    alpha: f64,
    requests: u32,
    seed: u64,
) -> ZipfVerdict {
    if cache_size == 0 || keyspace < 2 || requests == 0 || !alpha.is_finite() || alpha <= 0.0 {
        return ZipfVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut cache: BTreeSet<u32> = BTreeSet::new();
    let mut order: Vec<u32> = Vec::new();
    let mut hits = 0u32;
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    for _ in 0..requests {
        let u = unit(&mut rng_state).max(1e-12);
        let rank_f = (1.0 - u).powf(-1.0 / alpha) - 1.0;
        let key = (rank_f as u32).min(keyspace - 1);
        seen.insert(key);
        if cache.contains(&key) {
            hits += 1;
        } else {
            if cache.len() >= cache_size as usize {
                if let Some(evicted) = order.first().copied() {
                    cache.remove(&evicted);
                    order.remove(0);
                }
            }
            cache.insert(key);
            order.push(key);
        }
    }
    let hit_rate = f64::from(hits) / f64::from(requests);
    ZipfVerdict::Ok {
        hit_rate,
        unique_keys_seen: seen.len() as u32,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_kv_zipf_hit_rate")?;

    println!("low skew: {:?}", simulate(50, 1000, 0.5, 10_000, 42));
    println!("high skew: {:?}", simulate(50, 1000, 2.0, 10_000, 42));
    println!("invalid: {:?}", simulate(0, 1000, 1.0, 10_000, 42));
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
    fn higher_skew_higher_hit_rate() {
        let lo = simulate(50, 1000, 0.5, 10_000, 42);
        let hi = simulate(50, 1000, 2.0, 10_000, 42);
        if let (ZipfVerdict::Ok { hit_rate: l, .. }, ZipfVerdict::Ok { hit_rate: h, .. }) = (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn invalid_zero_cache() {
        assert_eq!(
            simulate(0, 1000, 1.0, 10_000, 42),
            ZipfVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_small_keyspace() {
        assert_eq!(simulate(50, 1, 1.0, 10_000, 42), ZipfVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_requests() {
        assert_eq!(simulate(50, 1000, 1.0, 0, 42), ZipfVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_alpha() {
        assert_eq!(
            simulate(50, 1000, 0.0, 10_000, 42),
            ZipfVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(50, 1000, f64::NAN, 10_000, 42),
            ZipfVerdict::InvalidConfig
        );
    }

    #[test]
    fn hit_rate_in_unit_range() {
        let v = simulate(50, 1000, 1.0, 1000, 42);
        if let ZipfVerdict::Ok { hit_rate, .. } = v {
            assert!((0.0..=1.0).contains(&hit_rate));
        }
    }

    #[test]
    fn unique_keys_bounded() {
        let v = simulate(50, 100, 1.0, 1000, 42);
        if let ZipfVerdict::Ok {
            unique_keys_seen, ..
        } = v
        {
            assert!(unique_keys_seen <= 100);
        }
    }

    #[test]
    fn deterministic() {
        let a = simulate(50, 1000, 1.0, 1000, 42);
        let b = simulate(50, 1000, 1.0, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn extreme_skew_high_hit() {
        let v = simulate(10, 1_000_000, 5.0, 10_000, 42);
        if let ZipfVerdict::Ok { hit_rate, .. } = v {
            assert!(hit_rate > 0.5);
        }
    }
}
