//! # Monte-Carlo Inference p99 Latency Estimator
//!
//! Estimate p99 latency from a small sample using bootstrap resampling.
//! Returns point estimate and 95% confidence interval. Uses an LCG so
//! the recipe is fully deterministic with a fixed seed.
//!
//! Demonstrates the **MC.07** recipe for PMAT-160 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Efron & Tibshirani (1993). An Introduction to the Bootstrap.
//!
//! Run with: cargo run --example mc_inference_p99_estimator
//!
//! Added by PMAT-160 (catalog 1063→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum P99Verdict {
    Ok {
        point_estimate: f64,
        ci_low: f64,
        ci_high: f64,
    },
    EmptySample,
    InvalidConfig,
}

pub fn estimate(latencies_ms: &[f64], num_resamples: u32, seed: u64) -> P99Verdict {
    if latencies_ms.is_empty() {
        return P99Verdict::EmptySample;
    }
    if num_resamples == 0 {
        return P99Verdict::InvalidConfig;
    }
    if latencies_ms.iter().any(|l| !l.is_finite() || *l < 0.0) {
        return P99Verdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let n = latencies_ms.len();
    let mut p99_samples = Vec::with_capacity(num_resamples as usize);
    for _ in 0..num_resamples {
        let mut sample: Vec<f64> = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = (lcg(&mut rng_state) % n as u64) as usize;
            sample.push(latencies_ms[idx]);
        }
        sample.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = ((n as f64) * 0.99) as usize;
        p99_samples.push(sample[idx.min(n - 1)]);
    }
    p99_samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = p99_samples[p99_samples.len() / 2];
    let lo_idx = (p99_samples.len() as f64 * 0.025) as usize;
    let hi_idx = (p99_samples.len() as f64 * 0.975) as usize;
    P99Verdict::Ok {
        point_estimate: mid,
        ci_low: p99_samples[lo_idx],
        ci_high: p99_samples[hi_idx.min(p99_samples.len() - 1)],
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_inference_p99_estimator")?;

    let latencies: Vec<f64> = (1..=100).map(f64::from).collect();
    println!("typical: {:?}", estimate(&latencies, 1000, 42));
    println!("empty: {:?}", estimate(&[], 1000, 42));
    println!("zero resamples: {:?}", estimate(&latencies, 0, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_estimate_in_range() {
        let latencies: Vec<f64> = (1..=100).map(f64::from).collect();
        let v = estimate(&latencies, 1000, 42);
        if let P99Verdict::Ok { point_estimate, .. } = v {
            // True p99 of 1..100 = 99.
            assert!(point_estimate >= 95.0 && point_estimate <= 100.0);
        }
    }

    #[test]
    fn ci_bounds_ordered() {
        let latencies: Vec<f64> = (1..=100).map(f64::from).collect();
        let v = estimate(&latencies, 1000, 42);
        if let P99Verdict::Ok {
            ci_low,
            ci_high,
            point_estimate,
        } = v
        {
            assert!(ci_low <= point_estimate);
            assert!(point_estimate <= ci_high);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(estimate(&[], 1000, 42), P99Verdict::EmptySample);
    }

    #[test]
    fn zero_resamples_invalid() {
        assert_eq!(estimate(&[1.0], 0, 42), P99Verdict::InvalidConfig);
    }

    #[test]
    fn nan_rejected() {
        assert_eq!(estimate(&[f64::NAN], 1000, 42), P99Verdict::InvalidConfig);
    }

    #[test]
    fn negative_rejected() {
        assert_eq!(estimate(&[-1.0], 1000, 42), P99Verdict::InvalidConfig);
    }

    #[test]
    fn deterministic_for_same_seed() {
        let lat: Vec<f64> = (1..=50).map(f64::from).collect();
        let a = estimate(&lat, 100, 42);
        let b = estimate(&lat, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_both_succeed() {
        let lat: Vec<f64> = (1..=100).map(f64::from).collect();
        let a = estimate(&lat, 100, 1);
        let b = estimate(&lat, 100, 2);
        assert!(matches!(a, P99Verdict::Ok { .. }));
        assert!(matches!(b, P99Verdict::Ok { .. }));
    }

    #[test]
    fn single_latency_passes() {
        let v = estimate(&[42.0], 100, 1);
        if let P99Verdict::Ok { point_estimate, .. } = v {
            assert!((point_estimate - 42.0).abs() < 1e-9);
        }
    }

    #[test]
    fn small_sample_ci_works() {
        let lat: Vec<f64> = (1..=10).map(f64::from).collect();
        let v = estimate(&lat, 100, 1);
        assert!(matches!(v, P99Verdict::Ok { .. }));
    }
}
