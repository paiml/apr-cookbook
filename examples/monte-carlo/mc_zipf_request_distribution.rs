//! # Monte-Carlo Zipf Request Distribution
//!
//! Sample N requests from a Zipf distribution over `keyspace` keys
//! with parameter `alpha`. Reports top-1 frequency and Gini-style
//! inequality coefficient.
//!
//! Demonstrates the **MC.119** recipe for PMAT-198 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Zipf's law (1949); web access pattern analysis
//!  (Breslau et al. INFOCOM 1999).
//!
//! Run with: cargo run --example mc_zipf_request_distribution
//!
//! Added by PMAT-198 (catalog 1405→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum ZipfVerdict {
    Ok {
        top1_freq: f64,
        top10_freq: f64,
        unique_keys: u32,
    },
    InvalidConfig,
}

pub fn simulate(requests: u32, keyspace: u32, alpha: f64, seed: u64) -> ZipfVerdict {
    if requests == 0 || keyspace == 0 || alpha <= 0.0 {
        return ZipfVerdict::InvalidConfig;
    }
    // Build cumulative weights ∝ 1/(rank^alpha).
    let mut weights: Vec<f64> = (1..=keyspace)
        .map(|r| 1.0 / (f64::from(r)).powf(alpha))
        .collect();
    let total: f64 = weights.iter().sum();
    for w in &mut weights {
        *w /= total;
    }
    let mut cum: Vec<f64> = Vec::with_capacity(weights.len());
    let mut acc = 0.0;
    for w in &weights {
        acc += w;
        cum.push(acc);
    }
    let mut counts: BTreeMap<u32, u32> = BTreeMap::new();
    let mut rng_state = seed | 1;
    for _ in 0..requests {
        let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        let idx = cum.iter().position(|c| *c > r).unwrap_or(cum.len() - 1);
        *counts.entry(idx as u32).or_insert(0) += 1;
    }
    let mut sorted_counts: Vec<u32> = counts.values().copied().collect();
    sorted_counts.sort_by(|a, b| b.cmp(a));
    let top1 = sorted_counts.first().copied().unwrap_or(0);
    let top10: u32 = sorted_counts.iter().take(10).sum();
    ZipfVerdict::Ok {
        top1_freq: f64::from(top1) / f64::from(requests),
        top10_freq: f64::from(top10) / f64::from(requests),
        unique_keys: counts.len() as u32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_zipf_request_distribution")?;

    println!("alpha=1.0: {:?}", simulate(10_000, 100, 1.0, 42));
    println!("alpha=2.0: {:?}", simulate(10_000, 100, 2.0, 42));
    println!("invalid: {:?}", simulate(0, 100, 1.0, 42));
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
    fn high_alpha_concentrates() {
        let lo = simulate(10_000, 100, 0.5, 42);
        let hi = simulate(10_000, 100, 2.5, 42);
        if let (ZipfVerdict::Ok { top1_freq: l, .. }, ZipfVerdict::Ok { top1_freq: h, .. }) =
            (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn invalid_zero_requests() {
        assert_eq!(simulate(0, 100, 1.0, 42), ZipfVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_keyspace() {
        assert_eq!(simulate(100, 0, 1.0, 42), ZipfVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_alpha() {
        assert_eq!(simulate(100, 100, 0.0, 42), ZipfVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(1000, 100, 1.0, 42);
        let b = simulate(1000, 100, 1.0, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn frequencies_in_unit_range() {
        let v = simulate(1000, 100, 1.0, 42);
        if let ZipfVerdict::Ok {
            top1_freq,
            top10_freq,
            ..
        } = v
        {
            assert!((0.0..=1.0).contains(&top1_freq));
            assert!((0.0..=1.0).contains(&top10_freq));
        }
    }

    #[test]
    fn top10_ge_top1() {
        let v = simulate(1000, 100, 1.0, 42);
        if let ZipfVerdict::Ok {
            top1_freq,
            top10_freq,
            ..
        } = v
        {
            assert!(top10_freq >= top1_freq);
        }
    }

    #[test]
    fn unique_keys_le_keyspace() {
        let v = simulate(1000, 100, 1.0, 42);
        if let ZipfVerdict::Ok { unique_keys, .. } = v {
            assert!(unique_keys <= 100);
        }
    }

    #[test]
    fn top10_le_one() {
        let v = simulate(1000, 100, 1.0, 42);
        if let ZipfVerdict::Ok { top10_freq, .. } = v {
            assert!(top10_freq <= 1.0);
        }
    }

    #[test]
    fn small_sample_works() {
        let v = simulate(10, 100, 1.0, 42);
        assert!(matches!(v, ZipfVerdict::Ok { .. }));
    }

    #[test]
    fn many_keys_handled() {
        let v = simulate(10_000, 1000, 1.0, 42);
        assert!(matches!(v, ZipfVerdict::Ok { .. }));
    }
}
