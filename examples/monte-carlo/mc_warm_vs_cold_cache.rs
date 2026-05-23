//! # Monte-Carlo Warm-vs-Cold Cache Load Test
//!
//! Sim N requests against a cache that warms over time. First M
//! requests pay cold-miss latency; subsequent ones pay warm-hit
//! with `hit_rate`. Returns observed mean latency before / after
//! warmup.
//!
//! Demonstrates the **MC.52** recipe for PMAT-175 (catalog crosses 1200).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: cache-warming literature (CDN edge tier).
//!
//! Run with: cargo run --example mc_warm_vs_cold_cache
//!
//! Added by PMAT-175 (catalog 1198→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CacheVerdict {
    Ok {
        cold_mean_ms: f64,
        warm_mean_ms: f64,
        speedup_pct: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    cold_latency_ms: f64,
    warm_latency_ms: f64,
    warm_hit_rate: f64,
    cold_requests: u32,
    warm_requests: u32,
    seed: u64,
) -> CacheVerdict {
    if !cold_latency_ms.is_finite()
        || cold_latency_ms < 0.0
        || !warm_latency_ms.is_finite()
        || warm_latency_ms < 0.0
        || warm_latency_ms > cold_latency_ms
        || !warm_hit_rate.is_finite()
        || !(0.0..=1.0).contains(&warm_hit_rate)
        || cold_requests == 0
        || warm_requests == 0
    {
        return CacheVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    // Cold phase: every request is a miss.
    let cold_mean_ms = cold_latency_ms;
    // Warm phase: hit_rate use warm_latency, miss uses cold.
    let mut warm_sum = 0.0;
    for _ in 0..warm_requests {
        let lat = if unit(&mut rng_state) < warm_hit_rate {
            warm_latency_ms
        } else {
            cold_latency_ms
        };
        warm_sum += lat;
    }
    let warm_mean_ms = warm_sum / f64::from(warm_requests);
    let speedup_pct = if cold_mean_ms > 0.0 {
        ((cold_mean_ms - warm_mean_ms) / cold_mean_ms) * 100.0
    } else {
        0.0
    };
    let _ = cold_requests; // Kept for API symmetry.
    CacheVerdict::Ok {
        cold_mean_ms,
        warm_mean_ms,
        speedup_pct,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_warm_vs_cold_cache")?;

    println!(
        "high hit rate: {:?}",
        simulate(50.0, 5.0, 0.95, 100, 1000, 42)
    );
    println!(
        "low hit rate: {:?}",
        simulate(50.0, 5.0, 0.30, 100, 1000, 42)
    );
    println!("invalid: {:?}", simulate(50.0, 100.0, 0.5, 100, 1000, 42));
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
    fn high_hit_rate_high_speedup() {
        let v = simulate(50.0, 5.0, 0.95, 100, 1000, 42);
        if let CacheVerdict::Ok { speedup_pct, .. } = v {
            assert!(speedup_pct > 70.0);
        }
    }

    #[test]
    fn low_hit_rate_low_speedup() {
        let v = simulate(50.0, 5.0, 0.10, 100, 1000, 42);
        if let CacheVerdict::Ok { speedup_pct, .. } = v {
            assert!(speedup_pct < 30.0);
        }
    }

    #[test]
    fn warm_mean_le_cold_mean() {
        let v = simulate(50.0, 5.0, 0.50, 100, 1000, 42);
        if let CacheVerdict::Ok {
            cold_mean_ms,
            warm_mean_ms,
            ..
        } = v
        {
            assert!(warm_mean_ms <= cold_mean_ms);
        }
    }

    #[test]
    fn invalid_warm_above_cold() {
        assert_eq!(
            simulate(50.0, 100.0, 0.5, 100, 1000, 42),
            CacheVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_negative_latency() {
        assert_eq!(
            simulate(-1.0, 5.0, 0.5, 100, 1000, 42),
            CacheVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_hit_rate_over_one() {
        assert_eq!(
            simulate(50.0, 5.0, 1.5, 100, 1000, 42),
            CacheVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_requests() {
        assert_eq!(
            simulate(50.0, 5.0, 0.5, 0, 1000, 42),
            CacheVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(f64::NAN, 5.0, 0.5, 100, 1000, 42),
            CacheVerdict::InvalidConfig
        );
    }

    #[test]
    fn full_hit_rate_warm_equals_warm() {
        let v = simulate(50.0, 5.0, 1.0, 100, 1000, 42);
        if let CacheVerdict::Ok { warm_mean_ms, .. } = v {
            assert!((warm_mean_ms - 5.0).abs() < 1e-9);
        }
    }

    #[test]
    fn deterministic() {
        let a = simulate(50.0, 5.0, 0.95, 100, 1000, 42);
        let b = simulate(50.0, 5.0, 0.95, 100, 1000, 42);
        assert_eq!(a, b);
    }
}
