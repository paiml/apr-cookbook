//! # Monte-Carlo Rejection Sampling for Normal-like Bell Curve
//!
//! Generate samples from a normal-like density `p(x) = exp(-x²/2)`
//! on [-3,3] using rejection sampling against a uniform proposal
//! `q(x) = 1/6 * M`. Returns mean, variance estimate, and accept rate.
//!
//! Demonstrates the **MC.148** recipe for PMAT-208 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: von Neumann, "Various techniques used in connection with
//!  random digits" NBS Symp. (1951); Devroye, Non-Uniform Random
//!  Variate Generation ch. II.3 (1986).
//!
//! Run with: cargo run --example mc_rejection_sampling_normal
//!
//! Added by PMAT-208 (catalog 1495→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RejectionVerdict {
    Ok {
        sample_mean_x1000: i32,
        sample_var_x1000: u32,
        accept_rate_x1000: u32,
    },
    InvalidConfig,
}

pub fn sample(target_n: u32, seed: u64) -> RejectionVerdict {
    if target_n < 100 {
        return RejectionVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut accepted: Vec<f64> = Vec::with_capacity(target_n as usize);
    let mut total = 0u32;
    while accepted.len() < target_n as usize && total < target_n.saturating_mul(100) {
        let u = (lcg(&mut state) as f64) / (u32::MAX as f64);
        let x = u * 6.0 - 3.0; // proposal: uniform on [-3, 3]
        let r = (lcg(&mut state) as f64) / (u32::MAX as f64);
        let p = (-x * x / 2.0).exp();
        if r < p {
            accepted.push(x);
        }
        total += 1;
    }
    if accepted.is_empty() {
        return RejectionVerdict::InvalidConfig;
    }
    let n = accepted.len() as f64;
    let mean = accepted.iter().sum::<f64>() / n;
    let var = accepted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    RejectionVerdict::Ok {
        sample_mean_x1000: (mean * 1000.0) as i32,
        sample_var_x1000: (var * 1000.0) as u32,
        accept_rate_x1000: ((accepted.len() as f64 / total as f64) * 1000.0) as u32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_rejection_sampling_normal")?;

    println!("sample-1k: {:?}", sample(1000, 42));
    println!("invalid: {:?}", sample(50, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn invalid_too_few_samples() {
        assert_eq!(sample(50, 42), RejectionVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = sample(500, 42);
        let b = sample(500, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn mean_near_zero_for_symmetric_target() {
        let v = sample(5000, 42);
        if let RejectionVerdict::Ok {
            sample_mean_x1000, ..
        } = v
        {
            // Mean of N(0,1) truncated to [-3,3] is ~0; allow ±0.1.
            assert!(sample_mean_x1000.abs() < 100);
        }
    }

    #[test]
    fn var_near_one_for_normal() {
        let v = sample(5000, 42);
        if let RejectionVerdict::Ok {
            sample_var_x1000, ..
        } = v
        {
            // Var(N(0,1) truncated to ±3) ≈ 0.97; allow [800, 1200].
            assert!((800..=1200).contains(&sample_var_x1000));
        }
    }

    #[test]
    fn accept_rate_in_zero_one() {
        let v = sample(500, 42);
        if let RejectionVerdict::Ok {
            accept_rate_x1000, ..
        } = v
        {
            assert!(accept_rate_x1000 <= 1000);
        }
    }

    #[test]
    fn accept_rate_positive() {
        let v = sample(500, 42);
        if let RejectionVerdict::Ok {
            accept_rate_x1000, ..
        } = v
        {
            // Expected accept rate ≈ √(2π)/6 ≈ 0.418 → ~418.
            assert!(accept_rate_x1000 > 200);
        }
    }

    #[test]
    fn min_samples_accepted() {
        let v = sample(100, 42);
        assert!(matches!(v, RejectionVerdict::Ok { .. }));
    }

    #[test]
    fn many_samples_handled() {
        let v = sample(10_000, 42);
        assert!(matches!(v, RejectionVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = sample(500, 42);
        let b = sample(500, 999);
        assert!(a != b);
    }

    #[test]
    fn var_finite() {
        let v = sample(500, 42);
        if let RejectionVerdict::Ok {
            sample_var_x1000, ..
        } = v
        {
            assert!(sample_var_x1000 < u32::MAX);
        }
    }

    #[test]
    fn mean_within_proposal_range() {
        // Samples lie in [-3, 3] → mean is bounded by ±3000.
        let v = sample(500, 42);
        if let RejectionVerdict::Ok {
            sample_mean_x1000, ..
        } = v
        {
            assert!(sample_mean_x1000.abs() <= 3000);
        }
    }
}
