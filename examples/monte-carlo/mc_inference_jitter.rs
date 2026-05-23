//! # Monte-Carlo Inference Jitter Generator
//!
//! Generate N synthetic latency samples = base_latency_ms + jitter
//! where jitter ~ uniform(-spread, +spread) clamped at zero. Returns
//! observed mean / max / std-dev.
//!
//! Demonstrates the **MC.12** recipe for PMAT-161 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Synthetic load benchmarking (TGI / DeepSpeed-MII).
//!
//! Run with: cargo run --example mc_inference_jitter
//!
//! Added by PMAT-161 (catalog 1072→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum JitterVerdict {
    Ok {
        mean_ms: f64,
        max_ms: f64,
        std_dev_ms: f64,
        sample_count: u32,
    },
    InvalidConfig,
}

pub fn generate(
    base_latency_ms: f64,
    jitter_spread_ms: f64,
    samples: u32,
    seed: u64,
) -> JitterVerdict {
    if !base_latency_ms.is_finite()
        || base_latency_ms < 0.0
        || !jitter_spread_ms.is_finite()
        || jitter_spread_ms < 0.0
        || samples == 0
    {
        return JitterVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut max_v = 0.0_f64;
    for _ in 0..samples {
        let jitter = (unit(&mut rng_state) * 2.0 - 1.0) * jitter_spread_ms;
        let v = (base_latency_ms + jitter).max(0.0);
        sum += v;
        sum_sq += v * v;
        if v > max_v {
            max_v = v;
        }
    }
    let n = f64::from(samples);
    let mean_ms = sum / n;
    let variance = (sum_sq / n) - mean_ms * mean_ms;
    let std_dev_ms = variance.max(0.0).sqrt();
    JitterVerdict::Ok {
        mean_ms,
        max_ms: max_v,
        std_dev_ms,
        sample_count: samples,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_inference_jitter")?;

    println!("typical: {:?}", generate(50.0, 10.0, 1000, 42));
    println!("no jitter: {:?}", generate(50.0, 0.0, 100, 42));
    println!("invalid: {:?}", generate(-1.0, 10.0, 100, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn mean_near_base() {
        let v = generate(50.0, 10.0, 10_000, 42);
        if let JitterVerdict::Ok { mean_ms, .. } = v {
            assert!((mean_ms - 50.0).abs() < 1.0);
        }
    }

    #[test]
    fn no_jitter_zero_std_dev() {
        let v = generate(50.0, 0.0, 100, 42);
        if let JitterVerdict::Ok { std_dev_ms, .. } = v {
            assert!(std_dev_ms < 1e-9);
        }
    }

    #[test]
    fn no_negative_samples() {
        let v = generate(5.0, 100.0, 1000, 42);
        if let JitterVerdict::Ok { mean_ms, .. } = v {
            // Mean clipped from below at 0.
            assert!(mean_ms >= 0.0);
        }
    }

    #[test]
    fn invalid_negative_base() {
        assert_eq!(generate(-1.0, 10.0, 100, 42), JitterVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_negative_spread() {
        assert_eq!(generate(50.0, -10.0, 100, 42), JitterVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(generate(50.0, 10.0, 0, 42), JitterVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            generate(f64::NAN, 10.0, 100, 42),
            JitterVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = generate(50.0, 10.0, 1000, 42);
        let b = generate(50.0, 10.0, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn sample_count_correct() {
        let v = generate(50.0, 10.0, 100, 42);
        if let JitterVerdict::Ok { sample_count, .. } = v {
            assert_eq!(sample_count, 100);
        }
    }

    #[test]
    fn higher_spread_higher_std_dev() {
        let low = generate(50.0, 1.0, 1000, 42);
        let high = generate(50.0, 20.0, 1000, 42);
        if let (
            JitterVerdict::Ok { std_dev_ms: s1, .. },
            JitterVerdict::Ok { std_dev_ms: s2, .. },
        ) = (low, high)
        {
            assert!(s2 > s1);
        }
    }
}
