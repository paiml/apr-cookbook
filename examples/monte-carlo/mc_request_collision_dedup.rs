//! # Monte-Carlo Request Collision Dedup
//!
//! Sim content-hash dedup: identical request payloads share a cache
//! entry. Returns observed dedup ratio (cache hits / total requests).
//!
//! Demonstrates the **MC.40** recipe for PMAT-171 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: content-defined dedup (Cloudflare Argo).
//!
//! Run with: cargo run --example mc_request_collision_dedup
//!
//! Added by PMAT-171 (catalog 1162→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum DedupVerdict {
    Ok {
        dedup_ratio: f64,
        unique_count: u32,
        total_count: u32,
    },
    InvalidConfig,
}

pub fn simulate(request_count: u32, distinct_payloads: u32, seed: u64) -> DedupVerdict {
    if request_count == 0 || distinct_payloads == 0 {
        return DedupVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    let mut hits = 0u32;
    for _ in 0..request_count {
        let id = (lcg(&mut rng_state) % u64::from(distinct_payloads)) as u32;
        if !seen.insert(id) {
            hits += 1;
        }
    }
    let dedup_ratio = f64::from(hits) / f64::from(request_count);
    DedupVerdict::Ok {
        dedup_ratio,
        unique_count: seen.len() as u32,
        total_count: request_count,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_request_collision_dedup")?;

    println!("hot keys: {:?}", simulate(10_000, 100, 42));
    println!("cold keys: {:?}", simulate(10_000, 10_000, 42));
    println!("invalid: {:?}", simulate(0, 100, 42));
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
    fn many_distinct_payloads_low_dedup() {
        let v = simulate(100, 100_000, 42);
        if let DedupVerdict::Ok { dedup_ratio, .. } = v {
            assert!(dedup_ratio < 0.05);
        }
    }

    #[test]
    fn small_payload_set_high_dedup() {
        let v = simulate(10_000, 10, 42);
        if let DedupVerdict::Ok { dedup_ratio, .. } = v {
            assert!(dedup_ratio > 0.99);
        }
    }

    #[test]
    fn invalid_zero_requests() {
        assert_eq!(simulate(0, 100, 42), DedupVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_payloads() {
        assert_eq!(simulate(100, 0, 42), DedupVerdict::InvalidConfig);
    }

    #[test]
    fn unique_bounded_by_payloads() {
        let v = simulate(100, 5, 42);
        if let DedupVerdict::Ok { unique_count, .. } = v {
            assert!(unique_count <= 5);
        }
    }

    #[test]
    fn unique_bounded_by_requests() {
        let v = simulate(5, 100, 42);
        if let DedupVerdict::Ok { unique_count, .. } = v {
            assert!(unique_count <= 5);
        }
    }

    #[test]
    fn deterministic() {
        let a = simulate(1000, 100, 42);
        let b = simulate(1000, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn ratio_in_unit_range() {
        let v = simulate(1000, 100, 42);
        if let DedupVerdict::Ok { dedup_ratio, .. } = v {
            assert!((0.0..=1.0).contains(&dedup_ratio));
        }
    }

    #[test]
    fn one_payload_only_dedup_max() {
        let v = simulate(1000, 1, 42);
        if let DedupVerdict::Ok { unique_count, .. } = v {
            assert_eq!(unique_count, 1);
        }
    }

    #[test]
    fn total_count_matches_input() {
        let v = simulate(500, 50, 42);
        if let DedupVerdict::Ok { total_count, .. } = v {
            assert_eq!(total_count, 500);
        }
    }
}
