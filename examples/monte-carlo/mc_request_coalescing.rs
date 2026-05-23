//! # Monte-Carlo Request Coalescing (Single-Flight)
//!
//! Sim duplicate-request coalescing: if N requests for the same key
//! arrive within a coalesce window, only one upstream call fires.
//! Returns coalesce-rate (saved upstream calls / total requests).
//!
//! Demonstrates the **MC.61** recipe for PMAT-179 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Go's `sync/singleflight` package; thundering-herd
//! mitigation (Decandia et al., Dynamo SOSP 2007).
//!
//! Run with: cargo run --example mc_request_coalescing
//!
//! Added by PMAT-179 (catalog 1234→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum CoalesceVerdict {
    Ok {
        upstream_calls: u32,
        coalesce_rate: f64,
    },
    InvalidConfig,
}

pub fn simulate(requests: u32, keyspace: u32, coalesce_window: u32, seed: u64) -> CoalesceVerdict {
    if requests == 0 || keyspace == 0 || coalesce_window == 0 {
        return CoalesceVerdict::InvalidConfig;
    }
    // For each key, track the in-flight upstream call's start step.
    let mut in_flight: BTreeMap<u32, u32> = BTreeMap::new();
    let mut upstream_calls = 0u32;
    let mut rng_state = seed | 1;
    for step in 0..requests {
        let key = ((lcg(&mut rng_state) >> 32) as u32) % keyspace;
        // Evict expired in-flight entries.
        in_flight.retain(|_, start| step.saturating_sub(*start) < coalesce_window);
        if let Entry::Vacant(slot) = in_flight.entry(key) {
            slot.insert(step);
            upstream_calls += 1;
        }
    }
    let coalesced = requests.saturating_sub(upstream_calls);
    let coalesce_rate = f64::from(coalesced) / f64::from(requests);
    CoalesceVerdict::Ok {
        upstream_calls,
        coalesce_rate,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_request_coalescing")?;

    println!("hot key: {:?}", simulate(10_000, 5, 50, 42));
    println!("cold spread: {:?}", simulate(10_000, 10_000, 50, 42));
    println!("invalid: {:?}", simulate(0, 5, 50, 42));
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
    fn small_keyspace_high_coalesce() {
        let v = simulate(10_000, 5, 50, 42);
        if let CoalesceVerdict::Ok { coalesce_rate, .. } = v {
            assert!(coalesce_rate > 0.5);
        }
    }

    #[test]
    fn large_keyspace_low_coalesce() {
        let v = simulate(1000, 100_000, 5, 42);
        if let CoalesceVerdict::Ok { coalesce_rate, .. } = v {
            assert!(coalesce_rate < 0.5);
        }
    }

    #[test]
    fn invalid_zero_requests() {
        assert_eq!(simulate(0, 5, 50, 42), CoalesceVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_keyspace() {
        assert_eq!(simulate(100, 0, 50, 42), CoalesceVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_window() {
        assert_eq!(simulate(100, 5, 0, 42), CoalesceVerdict::InvalidConfig);
    }

    #[test]
    fn rate_in_unit_range() {
        let v = simulate(1000, 100, 20, 42);
        if let CoalesceVerdict::Ok { coalesce_rate, .. } = v {
            assert!((0.0..=1.0).contains(&coalesce_rate));
        }
    }

    #[test]
    fn upstream_le_requests() {
        let v = simulate(1000, 100, 20, 42);
        if let CoalesceVerdict::Ok { upstream_calls, .. } = v {
            assert!(upstream_calls <= 1000);
        }
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 50, 10, 42);
        let b = simulate(500, 50, 10, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn longer_window_more_coalesce() {
        let short = simulate(5000, 50, 5, 42);
        let long = simulate(5000, 50, 200, 42);
        if let (
            CoalesceVerdict::Ok {
                coalesce_rate: s, ..
            },
            CoalesceVerdict::Ok {
                coalesce_rate: l, ..
            },
        ) = (short, long)
        {
            assert!(l >= s);
        }
    }

    #[test]
    fn no_coalesce_with_distinct_keys() {
        // keyspace == requests with very long window: still mostly distinct keys.
        let v = simulate(100, 1000, 100, 42);
        if let CoalesceVerdict::Ok { upstream_calls, .. } = v {
            // Most requests should generate upstream calls.
            assert!(upstream_calls > 50);
        }
    }

    #[test]
    fn single_request_one_call() {
        let v = simulate(1, 5, 10, 42);
        if let CoalesceVerdict::Ok {
            upstream_calls,
            coalesce_rate,
        } = v
        {
            assert_eq!(upstream_calls, 1);
            assert_eq!(coalesce_rate, 0.0);
        }
    }
}
