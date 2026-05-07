//! # Monte-Carlo Bootstrap Resample Mean
//!
//! Estimate the standard error of the mean by bootstrap resampling
//! (with replacement). Returns mean and standard-error estimates
//! (×1000 fixed point).
//!
//! Demonstrates the **MC.140** recipe for PMAT-205 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Efron, "Bootstrap Methods: Another Look at the
//!  Jackknife" Annals of Statistics 7(1):1-26 (1979).
//!
//! Run with: cargo run --example mc_bootstrap_resample_mean
//!
//! Added by PMAT-205 (catalog 1468→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BootstrapVerdict {
    Ok {
        sample_mean_x1000: i32,
        bootstrap_se_x1000: u32,
    },
    InvalidConfig,
}

pub fn estimate(data: &[i32], resamples: u32, seed: u64) -> BootstrapVerdict {
    if data.is_empty() || resamples < 100 {
        return BootstrapVerdict::InvalidConfig;
    }
    let n = data.len();
    let mut state = seed | 1;
    let sample_mean: f64 = data.iter().map(|x| *x as f64).sum::<f64>() / n as f64;
    let mut boot_means: Vec<f64> = Vec::with_capacity(resamples as usize);
    for _ in 0..resamples {
        let mut sum = 0.0f64;
        for _ in 0..n {
            let idx = (lcg(&mut state) as usize) % n;
            sum += data[idx] as f64;
        }
        boot_means.push(sum / n as f64);
    }
    let boot_mean: f64 = boot_means.iter().sum::<f64>() / resamples as f64;
    let var: f64 = boot_means
        .iter()
        .map(|m| (m - boot_mean).powi(2))
        .sum::<f64>()
        / (resamples - 1) as f64;
    let se = var.sqrt();
    BootstrapVerdict::Ok {
        sample_mean_x1000: (sample_mean * 1000.0) as i32,
        bootstrap_se_x1000: (se * 1000.0) as u32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_bootstrap_resample_mean")?;

    let data = [10, 20, 30, 40, 50];
    println!("estimate: {:?}", estimate(&data, 1000, 42));
    println!("invalid: {:?}", estimate(&[], 1000, 42));
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
    fn invalid_empty_data() {
        assert_eq!(estimate(&[], 1000, 42), BootstrapVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_resamples() {
        assert_eq!(estimate(&[1, 2], 50, 42), BootstrapVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = estimate(&[1, 2, 3], 200, 42);
        let b = estimate(&[1, 2, 3], 200, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn sample_mean_correct() {
        let v = estimate(&[10, 20, 30], 200, 42);
        if let BootstrapVerdict::Ok {
            sample_mean_x1000, ..
        } = v
        {
            assert_eq!(sample_mean_x1000, 20_000);
        }
    }

    #[test]
    fn constant_data_zero_se() {
        let v = estimate(&[5, 5, 5, 5], 200, 42);
        if let BootstrapVerdict::Ok {
            bootstrap_se_x1000, ..
        } = v
        {
            assert_eq!(bootstrap_se_x1000, 0);
        }
    }

    #[test]
    fn larger_sample_same_distribution_smaller_se() {
        // SE ∝ 1/√n for samples from the same distribution.
        // Replicate [1..=5] 1× vs 10× → same population, larger n.
        let small = estimate(&[1, 2, 3, 4, 5], 500, 42);
        let big_data: Vec<i32> = (0..10).flat_map(|_| 1..=5).collect();
        let big = estimate(&big_data, 500, 42);
        if let (
            BootstrapVerdict::Ok {
                bootstrap_se_x1000: s,
                ..
            },
            BootstrapVerdict::Ok {
                bootstrap_se_x1000: l,
                ..
            },
        ) = (small, big)
        {
            assert!(l < s);
        }
    }

    #[test]
    fn se_finite_and_positive_for_varied_data() {
        let v = estimate(&[1, 100, 1, 100], 500, 42);
        if let BootstrapVerdict::Ok {
            bootstrap_se_x1000, ..
        } = v
        {
            assert!(bootstrap_se_x1000 > 0);
            assert!(bootstrap_se_x1000 < u32::MAX);
        }
    }

    #[test]
    fn single_value_handled() {
        let v = estimate(&[42], 200, 42);
        if let BootstrapVerdict::Ok {
            sample_mean_x1000, ..
        } = v
        {
            assert_eq!(sample_mean_x1000, 42_000);
        }
    }

    #[test]
    fn negative_values_handled() {
        let v = estimate(&[-10, -20, -30], 200, 42);
        if let BootstrapVerdict::Ok {
            sample_mean_x1000, ..
        } = v
        {
            assert_eq!(sample_mean_x1000, -20_000);
        }
    }

    #[test]
    fn many_resamples_handled() {
        let v = estimate(&[1, 2, 3], 10_000, 42);
        assert!(matches!(v, BootstrapVerdict::Ok { .. }));
    }

    #[test]
    fn diverse_data_se_in_reasonable_range() {
        let v = estimate(&[10, 20, 30, 40, 50], 5000, 42);
        if let BootstrapVerdict::Ok {
            bootstrap_se_x1000, ..
        } = v
        {
            // Population SD ≈ 14.14, true SE = SD/√5 ≈ 6.32.
            // Bootstrap SE should be in same ballpark.
            assert!((4_000..=9_000).contains(&bootstrap_se_x1000));
        }
    }
}
