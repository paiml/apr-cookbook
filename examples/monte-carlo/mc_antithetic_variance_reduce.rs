//! # Monte-Carlo Antithetic Variates Variance Reduction
//!
//! Estimate E[f(U)] for U ~ Uniform[0,1] using antithetic variates:
//! pair each sample u with 1-u. Compares variance vs naive Monte
//! Carlo with same total samples.
//!
//! Demonstrates the **MC.141** recipe for PMAT-205 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hammersley & Morton, "A new Monte Carlo technique:
//!  antithetic variates" (1956); Glasserman, Monte Carlo Methods in
//!  Financial Engineering ch. 4.2.
//!
//! Run with: cargo run --example mc_antithetic_variance_reduce
//!
//! Added by PMAT-205 (catalog 1468→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AntitheticVerdict {
    Ok {
        naive_var_x1000: u32,
        antithetic_var_x1000: u32,
        reduction_ratio_x1000: u32,
    },
    InvalidConfig,
}

pub fn estimate(pairs: u32, seed: u64) -> AntitheticVerdict {
    if pairs < 100 {
        return AntitheticVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut naive_samples: Vec<f64> = Vec::with_capacity((pairs * 2) as usize);
    let mut antithetic_pair_means: Vec<f64> = Vec::with_capacity(pairs as usize);
    for _ in 0..pairs {
        let u = (lcg(&mut state) as f64) / (u32::MAX as f64);
        let v = (lcg(&mut state) as f64) / (u32::MAX as f64);
        naive_samples.push(f(u));
        naive_samples.push(f(v));
        antithetic_pair_means.push((f(u) + f(1.0 - u)) / 2.0);
    }
    let naive_var = variance(&naive_samples);
    let anti_var = variance(&antithetic_pair_means);
    // ratio = antithetic / naive (per-sample for naive, per-pair for anti)
    let reduction = if naive_var > 0.0 {
        (anti_var / naive_var * 1000.0) as u32
    } else {
        1000
    };
    AntitheticVerdict::Ok {
        naive_var_x1000: (naive_var * 1000.0) as u32,
        antithetic_var_x1000: (anti_var * 1000.0) as u32,
        reduction_ratio_x1000: reduction,
    }
}

fn f(u: f64) -> f64 {
    u
}

fn variance(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_antithetic_variance_reduce")?;

    println!("estimate: {:?}", estimate(10_000, 42));
    println!("invalid: {:?}", estimate(50, 42));
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
    fn invalid_too_few_pairs() {
        assert_eq!(estimate(50, 42), AntitheticVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = estimate(500, 42);
        let b = estimate(500, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn antithetic_reduces_variance_for_monotone_f() {
        // f(u)=u is monotone → antithetic should give zero variance
        // per pair (since (u + (1-u))/2 = 0.5 always).
        let v = estimate(10_000, 42);
        if let AntitheticVerdict::Ok {
            antithetic_var_x1000,
            ..
        } = v
        {
            assert_eq!(antithetic_var_x1000, 0);
        }
    }

    #[test]
    fn naive_var_positive_for_uniform() {
        let v = estimate(5000, 42);
        if let AntitheticVerdict::Ok {
            naive_var_x1000, ..
        } = v
        {
            // Var(U) = 1/12 ≈ 0.0833 → 83.
            assert!((50..=120).contains(&naive_var_x1000));
        }
    }

    #[test]
    fn reduction_ratio_under_one_for_monotone() {
        let v = estimate(5000, 42);
        if let AntitheticVerdict::Ok {
            reduction_ratio_x1000,
            ..
        } = v
        {
            // Antithetic var = 0 → ratio = 0.
            assert!(reduction_ratio_x1000 < 100);
        }
    }

    #[test]
    fn many_pairs_handled() {
        let v = estimate(100_000, 42);
        assert!(matches!(v, AntitheticVerdict::Ok { .. }));
    }

    #[test]
    fn variance_function_correct() {
        // Var of [0, 0, 0, 0] = 0
        assert_eq!(variance(&[0.0, 0.0, 0.0, 0.0]), 0.0);
        // Var of [1, -1] (sample variance) = 2.0
        assert_eq!(variance(&[1.0, -1.0]), 2.0);
    }

    #[test]
    fn minimum_pairs_accepted() {
        let v = estimate(100, 42);
        assert!(matches!(v, AntitheticVerdict::Ok { .. }));
    }

    #[test]
    fn naive_var_finite() {
        let v = estimate(500, 42);
        if let AntitheticVerdict::Ok {
            naive_var_x1000, ..
        } = v
        {
            assert!(naive_var_x1000 < u32::MAX);
        }
    }

    #[test]
    fn different_seed_different_naive_var() {
        let a = estimate(500, 42);
        let b = estimate(500, 999);
        assert!(a != b);
    }

    #[test]
    fn antithetic_var_le_naive_var() {
        // Antithetic variance per pair ≤ naive per-sample for monotone f.
        let v = estimate(5000, 42);
        if let AntitheticVerdict::Ok {
            naive_var_x1000,
            antithetic_var_x1000,
            ..
        } = v
        {
            assert!(antithetic_var_x1000 <= naive_var_x1000);
        }
    }
}
