//! # Monte-Carlo Log Aggregation Window
//!
//! Sim log lines arriving at varying rates; aggregate into
//! `window_secs` buckets. Returns max bucket size and aggregation
//! efficiency (small buckets are wasted overhead).
//!
//! Demonstrates the **MC.47** recipe for PMAT-173 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Vector / Logstash batching strategies.
//!
//! Run with: cargo run --example mc_log_aggregation_window
//!
//! Added by PMAT-173 (catalog 1180→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AggregateVerdict {
    Ok {
        max_bucket: u32,
        mean_bucket: f64,
        empty_buckets: u32,
    },
    InvalidConfig,
}

pub fn simulate(
    line_rate_per_sec: f64,
    window_secs: u32,
    duration_secs: u32,
    seed: u64,
) -> AggregateVerdict {
    if !line_rate_per_sec.is_finite()
        || line_rate_per_sec < 0.0
        || window_secs == 0
        || duration_secs == 0
    {
        return AggregateVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let bucket_count = duration_secs.div_ceil(window_secs);
    let mut buckets = vec![0u32; bucket_count as usize];
    let total_lines = (line_rate_per_sec * f64::from(duration_secs)) as u32;
    for _ in 0..total_lines {
        let t = (unit(&mut rng_state) * f64::from(duration_secs)) as u32;
        let idx = (t / window_secs).min(bucket_count - 1);
        buckets[idx as usize] += 1;
    }
    let max_bucket = *buckets.iter().max().unwrap_or(&0);
    let mean_bucket = buckets.iter().map(|c| f64::from(*c)).sum::<f64>() / f64::from(bucket_count);
    let empty_buckets = buckets.iter().filter(|c| **c == 0).count() as u32;
    AggregateVerdict::Ok {
        max_bucket,
        mean_bucket,
        empty_buckets,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_log_aggregation_window")?;

    println!("dense: {:?}", simulate(100.0, 1, 60, 42));
    println!("sparse: {:?}", simulate(1.0, 5, 60, 42));
    println!("invalid: {:?}", simulate(-1.0, 5, 60, 42));
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
    fn dense_no_empty() {
        let v = simulate(100.0, 5, 60, 42);
        if let AggregateVerdict::Ok { empty_buckets, .. } = v {
            assert_eq!(empty_buckets, 0);
        }
    }

    #[test]
    fn sparse_some_empty() {
        let v = simulate(0.05, 1, 100, 42);
        if let AggregateVerdict::Ok { empty_buckets, .. } = v {
            assert!(empty_buckets > 0);
        }
    }

    #[test]
    fn invalid_neg_rate() {
        assert_eq!(simulate(-1.0, 5, 60, 42), AggregateVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_window() {
        assert_eq!(simulate(10.0, 0, 60, 42), AggregateVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_duration() {
        assert_eq!(simulate(10.0, 5, 0, 42), AggregateVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(f64::NAN, 5, 60, 42),
            AggregateVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(50.0, 5, 60, 42);
        let b = simulate(50.0, 5, 60, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn max_at_least_mean() {
        let v = simulate(50.0, 5, 60, 42);
        if let AggregateVerdict::Ok {
            max_bucket,
            mean_bucket,
            ..
        } = v
        {
            assert!(f64::from(max_bucket) >= mean_bucket);
        }
    }

    #[test]
    fn zero_rate_all_empty() {
        let v = simulate(0.0, 5, 60, 42);
        if let AggregateVerdict::Ok { max_bucket, .. } = v {
            assert_eq!(max_bucket, 0);
        }
    }

    #[test]
    fn higher_rate_higher_mean() {
        let lo = simulate(1.0, 5, 60, 42);
        let hi = simulate(100.0, 5, 60, 42);
        if let (
            AggregateVerdict::Ok { mean_bucket: l, .. },
            AggregateVerdict::Ok { mean_bucket: h, .. },
        ) = (lo, hi)
        {
            assert!(h > l);
        }
    }
}
