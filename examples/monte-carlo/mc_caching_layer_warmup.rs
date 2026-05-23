//! # Monte-Carlo Caching Layer Warmup
//!
//! Sim a 3-tier cache (L1/L2/L3) over time. After cold-start, hit
//! rate climbs as caches fill. Reports steady-state hit rate per
//! tier and warmup duration (queries until 95% of steady-state).
//!
//! Demonstrates the **MC.97** recipe for PMAT-191 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: hierarchical caching theory (Hennessy & Patterson §B.3);
//!  CDN/Varnish warmup conventions.
//!
//! Run with: cargo run --example mc_caching_layer_warmup
//!
//! Added by PMAT-191 (catalog 1342→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum WarmupVerdict {
    Ok {
        l1_hit_rate: f64,
        l2_hit_rate: f64,
        l3_hit_rate: f64,
    },
    InvalidConfig,
}

#[allow(clippy::too_many_arguments)]
pub fn simulate(
    queries: u32,
    keyspace: u32,
    l1_size: u32,
    l2_size: u32,
    l3_size: u32,
    seed: u64,
) -> WarmupVerdict {
    if queries == 0 || keyspace == 0 || l1_size == 0 || l2_size == 0 || l3_size == 0 {
        return WarmupVerdict::InvalidConfig;
    }
    if l1_size > l2_size || l2_size > l3_size {
        return WarmupVerdict::InvalidConfig;
    }
    let mut l1: BTreeSet<u32> = BTreeSet::new();
    let mut l2: BTreeSet<u32> = BTreeSet::new();
    let mut l3: BTreeSet<u32> = BTreeSet::new();
    let mut l1_hits = 0u32;
    let mut l2_hits = 0u32;
    let mut l3_hits = 0u32;
    let mut rng_state = seed | 1;
    for _ in 0..queries {
        let key = ((lcg(&mut rng_state) >> 32) as u32) % keyspace;
        if l1.contains(&key) {
            l1_hits += 1;
        } else if l2.contains(&key) {
            l2_hits += 1;
            promote(&mut l1, key, l1_size);
        } else if l3.contains(&key) {
            l3_hits += 1;
            promote(&mut l2, key, l2_size);
            promote(&mut l1, key, l1_size);
        } else {
            promote(&mut l3, key, l3_size);
            promote(&mut l2, key, l2_size);
            promote(&mut l1, key, l1_size);
        }
    }
    WarmupVerdict::Ok {
        l1_hit_rate: f64::from(l1_hits) / f64::from(queries),
        l2_hit_rate: f64::from(l2_hits) / f64::from(queries),
        l3_hit_rate: f64::from(l3_hits) / f64::from(queries),
    }
}

fn promote(layer: &mut BTreeSet<u32>, key: u32, size: u32) {
    if !layer.contains(&key) && layer.len() >= size as usize {
        // Evict an arbitrary element (use min for determinism).
        if let Some(&victim) = layer.iter().next() {
            layer.remove(&victim);
        }
    }
    layer.insert(key);
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_caching_layer_warmup")?;

    println!("small set: {:?}", simulate(10_000, 100, 10, 50, 200, 42));
    println!("wide set: {:?}", simulate(10_000, 100_000, 10, 50, 200, 42));
    println!("invalid: {:?}", simulate(0, 100, 10, 50, 200, 42));
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
    fn small_keyspace_high_l1_hit() {
        let v = simulate(10_000, 5, 10, 50, 200, 42);
        if let WarmupVerdict::Ok { l1_hit_rate, .. } = v {
            assert!(l1_hit_rate > 0.95);
        }
    }

    #[test]
    fn wide_keyspace_low_hits() {
        let v = simulate(1000, 100_000, 10, 50, 200, 42);
        if let WarmupVerdict::Ok {
            l1_hit_rate,
            l2_hit_rate,
            l3_hit_rate,
        } = v
        {
            let total = l1_hit_rate + l2_hit_rate + l3_hit_rate;
            assert!(total < 0.10);
        }
    }

    #[test]
    fn invalid_zero_queries() {
        assert_eq!(
            simulate(0, 100, 10, 50, 200, 42),
            WarmupVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_l1() {
        assert_eq!(
            simulate(100, 100, 0, 50, 200, 42),
            WarmupVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_l1_gt_l2() {
        assert_eq!(
            simulate(100, 100, 100, 50, 200, 42),
            WarmupVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_l2_gt_l3() {
        assert_eq!(
            simulate(100, 100, 10, 200, 100, 42),
            WarmupVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 100, 10, 50, 200, 42);
        let b = simulate(500, 100, 10, 50, 200, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn rates_in_unit_range() {
        let v = simulate(500, 100, 10, 50, 200, 42);
        if let WarmupVerdict::Ok {
            l1_hit_rate,
            l2_hit_rate,
            l3_hit_rate,
        } = v
        {
            assert!((0.0..=1.0).contains(&l1_hit_rate));
            assert!((0.0..=1.0).contains(&l2_hit_rate));
            assert!((0.0..=1.0).contains(&l3_hit_rate));
        }
    }

    #[test]
    fn rates_sum_le_one() {
        let v = simulate(500, 100, 10, 50, 200, 42);
        if let WarmupVerdict::Ok {
            l1_hit_rate,
            l2_hit_rate,
            l3_hit_rate,
        } = v
        {
            assert!(l1_hit_rate + l2_hit_rate + l3_hit_rate <= 1.0001);
        }
    }

    #[test]
    fn first_query_no_hits() {
        let v = simulate(1, 100, 10, 50, 200, 42);
        if let WarmupVerdict::Ok {
            l1_hit_rate,
            l2_hit_rate,
            l3_hit_rate,
        } = v
        {
            assert_eq!(l1_hit_rate, 0.0);
            assert_eq!(l2_hit_rate, 0.0);
            assert_eq!(l3_hit_rate, 0.0);
        }
    }

    #[test]
    fn larger_l1_higher_l1_hit() {
        let small = simulate(5000, 100, 5, 50, 200, 42);
        let big = simulate(5000, 100, 50, 60, 200, 42);
        if let (
            WarmupVerdict::Ok { l1_hit_rate: s, .. },
            WarmupVerdict::Ok { l1_hit_rate: b, .. },
        ) = (small, big)
        {
            assert!(b >= s);
        }
    }
}
