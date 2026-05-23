//! # Monte-Carlo Skewed Load Distribution
//!
//! Generate Zipfian-distributed key access pattern (heavy hitters
//! dominate). Returns the top-K key share and full Gini coefficient.
//!
//! Demonstrates the **MC.36** recipe for PMAT-169 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Zipf's law (1949) + Gini coefficient (Gini 1912).
//!
//! Run with: cargo run --example mc_skewed_load_distribution
//!
//! Added by PMAT-169 (catalog 1144→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SkewVerdict {
    Ok {
        top_k_share_pct: f64,
        gini: f64,
        unique_keys_hit: u32,
    },
    InvalidConfig,
}

pub fn simulate(
    keyspace: u32,
    requests: u32,
    skew_alpha: f64,
    top_k: u32,
    seed: u64,
) -> SkewVerdict {
    if keyspace == 0
        || requests == 0
        || top_k == 0
        || top_k > keyspace
        || !skew_alpha.is_finite()
        || skew_alpha < 0.0
    {
        return SkewVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut counts = vec![0u32; keyspace as usize];
    for _ in 0..requests {
        // Approximate Zipfian: pick rank k with prob ~ 1/k^alpha.
        let u = unit(&mut rng_state).max(1e-12);
        let rank = ((u.powf(-1.0 / (skew_alpha + 1.0)) - 1.0) as u32).min(keyspace - 1);
        counts[rank as usize] += 1;
    }
    let total: u32 = counts.iter().sum();
    let mut sorted: Vec<u32> = counts.clone();
    sorted.sort_by(|a, b| b.cmp(a));
    let top_k_count: u32 = sorted.iter().take(top_k as usize).sum();
    let top_k_share_pct = (f64::from(top_k_count) / f64::from(total)) * 100.0;
    let gini = compute_gini(&sorted);
    let unique_keys_hit = counts.iter().filter(|c| **c > 0).count() as u32;
    SkewVerdict::Ok {
        top_k_share_pct,
        gini,
        unique_keys_hit,
    }
}

fn compute_gini(sorted_desc: &[u32]) -> f64 {
    let n = sorted_desc.len();
    if n == 0 {
        return 0.0;
    }
    let total: u64 = sorted_desc.iter().map(|c| u64::from(*c)).sum();
    if total == 0 {
        return 0.0;
    }
    // Lorenz/Gini for discrete counts.
    let mut acc: u64 = 0;
    let mut weighted: f64 = 0.0;
    for (i, c) in sorted_desc.iter().rev().enumerate() {
        acc += u64::from(*c);
        weighted += (i as f64 + 1.0) * acc as f64;
    }
    let gini = (2.0 * weighted) / (n as f64 * total as f64) - (n as f64 + 1.0) / n as f64;
    gini.clamp(0.0, 1.0)
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_skewed_load_distribution")?;

    println!("uniform: {:?}", simulate(100, 10_000, 0.0, 10, 42));
    println!("mild skew: {:?}", simulate(100, 10_000, 1.0, 10, 42));
    println!("heavy skew: {:?}", simulate(100, 10_000, 3.0, 10, 42));
    println!("invalid: {:?}", simulate(0, 10_000, 1.0, 10, 42));
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
    fn heavy_skew_high_top_k_share() {
        let v = simulate(100, 10_000, 3.0, 10, 42);
        if let SkewVerdict::Ok {
            top_k_share_pct, ..
        } = v
        {
            assert!(top_k_share_pct > 50.0);
        }
    }

    #[test]
    fn invalid_zero_keyspace() {
        assert_eq!(simulate(0, 10_000, 1.0, 10, 42), SkewVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_requests() {
        assert_eq!(simulate(100, 0, 1.0, 10, 42), SkewVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_topk() {
        assert_eq!(
            simulate(100, 10_000, 1.0, 0, 42),
            SkewVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_topk_over_keyspace() {
        assert_eq!(simulate(10, 100, 1.0, 100, 42), SkewVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_negative_alpha() {
        assert_eq!(
            simulate(100, 10_000, -0.5, 10, 42),
            SkewVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(100, 10_000, f64::NAN, 10, 42),
            SkewVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 1000, 1.0, 10, 42);
        let b = simulate(100, 1000, 1.0, 10, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn gini_in_unit_range() {
        let v = simulate(100, 10_000, 1.0, 10, 42);
        if let SkewVerdict::Ok { gini, .. } = v {
            assert!((0.0..=1.0).contains(&gini));
        }
    }

    #[test]
    fn unique_keys_bounded_by_keyspace() {
        let v = simulate(100, 10_000, 1.0, 10, 42);
        if let SkewVerdict::Ok {
            unique_keys_hit, ..
        } = v
        {
            assert!(unique_keys_hit <= 100);
        }
    }
}
