//! # Monte-Carlo Inverse CDF Sampling
//!
//! Sample from an exponential distribution Exp(λ) using inverse-CDF
//! transform: x = -ln(1-u)/λ for u ~ Uniform[0,1]. Returns sample
//! mean estimate (×1000) and theoretical mean for comparison.
//!
//! Demonstrates the **MC.149** recipe for PMAT-208 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Devroye, Non-Uniform Random Variate Generation ch. II.2
//!  (1986); Knuth TAOCP §3.4.1.
//!
//! Run with: cargo run --example mc_inverse_cdf_sample
//!
//! Added by PMAT-208 (catalog 1495→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum InvCdfVerdict {
    Ok {
        sample_mean_x1000: u32,
        theoretical_mean_x1000: u32,
    },
    InvalidConfig,
}

pub fn sample_exp(lambda_x1000: u32, samples: u32, seed: u64) -> InvCdfVerdict {
    if lambda_x1000 == 0 || samples < 100 {
        return InvCdfVerdict::InvalidConfig;
    }
    let lambda = lambda_x1000 as f64 / 1000.0;
    let mut state = seed | 1;
    let mut sum = 0.0f64;
    for _ in 0..samples {
        // u in (0, 1) — avoid u=0 which would give log(1)=0 (fine) and u=1 which gives ln(0)=-∞
        let raw = lcg(&mut state) as f64;
        let u = (raw / (u32::MAX as f64 + 1.0)).max(1e-10);
        let x = -(1.0 - u).ln() / lambda;
        sum += x;
    }
    let mean = sum / samples as f64;
    let theoretical = 1.0 / lambda;
    InvCdfVerdict::Ok {
        sample_mean_x1000: (mean * 1000.0) as u32,
        theoretical_mean_x1000: (theoretical * 1000.0) as u32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_inverse_cdf_sample")?;

    println!("lambda=1: {:?}", sample_exp(1000, 10_000, 42));
    println!("lambda=2: {:?}", sample_exp(2000, 10_000, 42));
    println!("invalid: {:?}", sample_exp(0, 10_000, 42));
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
    fn invalid_zero_lambda() {
        assert_eq!(sample_exp(0, 1000, 42), InvCdfVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_samples() {
        assert_eq!(sample_exp(1000, 50, 42), InvCdfVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = sample_exp(1000, 500, 42);
        let b = sample_exp(1000, 500, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn mean_close_to_theoretical_lambda_1() {
        // E[Exp(1)] = 1.0 → 1000.
        let v = sample_exp(1000, 50_000, 42);
        if let InvCdfVerdict::Ok {
            sample_mean_x1000, ..
        } = v
        {
            assert!((950..=1050).contains(&sample_mean_x1000));
        }
    }

    #[test]
    fn mean_close_to_theoretical_lambda_2() {
        // E[Exp(2)] = 0.5 → 500.
        let v = sample_exp(2000, 50_000, 42);
        if let InvCdfVerdict::Ok {
            sample_mean_x1000, ..
        } = v
        {
            assert!((450..=550).contains(&sample_mean_x1000));
        }
    }

    #[test]
    fn theoretical_returned() {
        let v = sample_exp(2000, 1000, 42);
        if let InvCdfVerdict::Ok {
            theoretical_mean_x1000,
            ..
        } = v
        {
            assert_eq!(theoretical_mean_x1000, 500);
        }
    }

    #[test]
    fn sample_mean_finite() {
        let v = sample_exp(1000, 1000, 42);
        if let InvCdfVerdict::Ok {
            sample_mean_x1000, ..
        } = v
        {
            assert!(sample_mean_x1000 < u32::MAX);
        }
    }

    #[test]
    fn larger_lambda_smaller_mean() {
        let small = sample_exp(500, 5000, 42);
        let large = sample_exp(5000, 5000, 42);
        if let (
            InvCdfVerdict::Ok {
                sample_mean_x1000: s,
                ..
            },
            InvCdfVerdict::Ok {
                sample_mean_x1000: l,
                ..
            },
        ) = (small, large)
        {
            assert!(l < s);
        }
    }

    #[test]
    fn min_samples_accepted() {
        let v = sample_exp(1000, 100, 42);
        assert!(matches!(v, InvCdfVerdict::Ok { .. }));
    }

    #[test]
    fn many_samples_handled() {
        let v = sample_exp(1000, 100_000, 42);
        assert!(matches!(v, InvCdfVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_means() {
        let a = sample_exp(1000, 500, 42);
        let b = sample_exp(1000, 500, 999);
        assert!(a != b);
    }
}
