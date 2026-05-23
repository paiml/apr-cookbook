//! # Monte-Carlo DNS Resolver Cache
//!
//! Sim a DNS resolver with TTL-based cache. Lookups for the same
//! name within TTL are served from cache; otherwise refetched
//! (incurring upstream lookup cost). Reports hit-rate.
//!
//! Demonstrates the **MC.70** recipe for PMAT-182 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: RFC 1035 §3.2.1 (RR TTL); Mockapetris, Domain Names
//!  Implementation and Specification (1987).
//!
//! Run with: cargo run --example mc_dns_resolver_cache
//!
//! Added by PMAT-182 (catalog 1261→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum DnsVerdict {
    Ok {
        hits: u32,
        misses: u32,
        hit_rate: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    queries: u32,
    name_count: u32,
    ttl_seconds: u32,
    inter_arrival_secs: u32,
    seed: u64,
) -> DnsVerdict {
    if queries == 0 || name_count == 0 || ttl_seconds == 0 || inter_arrival_secs == 0 {
        return DnsVerdict::InvalidConfig;
    }
    let mut cache: BTreeMap<u32, u32> = BTreeMap::new(); // name → expires_at
    let mut hits: u32 = 0;
    let mut misses: u32 = 0;
    let mut now: u32 = 0;
    let mut rng_state = seed | 1;
    for _ in 0..queries {
        let name = ((lcg(&mut rng_state) >> 32) as u32) % name_count;
        let cached = cache.get(&name).copied().unwrap_or(0);
        if cached > now {
            hits += 1;
        } else {
            misses += 1;
            cache.insert(name, now + ttl_seconds);
        }
        let jitter = ((lcg(&mut rng_state) >> 32) as u32) % (2 * inter_arrival_secs).max(1);
        now += jitter;
    }
    let hit_rate = f64::from(hits) / f64::from(queries);
    DnsVerdict::Ok {
        hits,
        misses,
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
    let _ctx = RecipeContext::new("mc_dns_resolver_cache")?;

    println!("hot small set: {:?}", simulate(10_000, 5, 300, 1, 42));
    println!(
        "cold large set: {:?}",
        simulate(10_000, 100_000, 300, 1, 42)
    );
    println!("invalid: {:?}", simulate(0, 5, 300, 1, 42));
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
    fn small_namespace_high_hit_rate() {
        let v = simulate(10_000, 5, 300, 1, 42);
        if let DnsVerdict::Ok { hit_rate, .. } = v {
            assert!(hit_rate > 0.5);
        }
    }

    #[test]
    fn large_namespace_low_hit_rate() {
        let v = simulate(1000, 100_000, 60, 5, 42);
        if let DnsVerdict::Ok { hit_rate, .. } = v {
            assert!(hit_rate < 0.5);
        }
    }

    #[test]
    fn longer_ttl_higher_hits() {
        let short = simulate(5000, 50, 1, 5, 42);
        let long = simulate(5000, 50, 1000, 5, 42);
        if let (DnsVerdict::Ok { hit_rate: s, .. }, DnsVerdict::Ok { hit_rate: l, .. }) =
            (short, long)
        {
            assert!(l >= s);
        }
    }

    #[test]
    fn invalid_zero_queries() {
        assert_eq!(simulate(0, 5, 300, 1, 42), DnsVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_names() {
        assert_eq!(simulate(100, 0, 300, 1, 42), DnsVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_ttl() {
        assert_eq!(simulate(100, 5, 0, 1, 42), DnsVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_arrival() {
        assert_eq!(simulate(100, 5, 300, 0, 42), DnsVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 10, 60, 5, 42);
        let b = simulate(500, 10, 60, 5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn hits_plus_misses_equals_queries() {
        let v = simulate(500, 10, 60, 5, 42);
        if let DnsVerdict::Ok { hits, misses, .. } = v {
            assert_eq!(hits + misses, 500);
        }
    }

    #[test]
    fn hit_rate_in_unit_range() {
        let v = simulate(500, 10, 60, 5, 42);
        if let DnsVerdict::Ok { hit_rate, .. } = v {
            assert!((0.0..=1.0).contains(&hit_rate));
        }
    }

    #[test]
    fn first_query_always_misses() {
        let v = simulate(1, 1, 300, 5, 42);
        if let DnsVerdict::Ok { hits, .. } = v {
            assert_eq!(hits, 0);
        }
    }
}
