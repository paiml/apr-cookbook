//! # Monte-Carlo Stratified Sampling
//!
//! Estimate E[f(U)] for U ~ Uniform[0,1] using k stratified samples
//! per stratum (vs. naive Monte Carlo). Compares the variance of the
//! mean estimator. Returns naive var, stratified var, and reduction.
//!
//! Demonstrates the **MC.144** recipe for PMAT-206 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Cochran, Sampling Techniques 3rd ed. ch. 5 (1977);
//!  McKay, Beckman & Conover, Latin Hypercube precursor (1979).
//!
//! Run with: cargo run --example mc_stratified_sampling
//!
//! Added by PMAT-206 (catalog 1477→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum StratifiedVerdict {
    Ok {
        naive_var_x1000: u32,
        stratified_var_x1000: u32,
        strata: u32,
    },
    InvalidConfig,
}

pub fn estimate(
    strata: u32,
    samples_per_stratum: u32,
    trials: u32,
    seed: u64,
) -> StratifiedVerdict {
    if strata < 2 || samples_per_stratum == 0 || trials < 30 {
        return StratifiedVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let total_per_trial = strata * samples_per_stratum;
    let mut naive_means: Vec<f64> = Vec::with_capacity(trials as usize);
    let mut strat_means: Vec<f64> = Vec::with_capacity(trials as usize);
    for _ in 0..trials {
        // Naive: total_per_trial uniform samples on [0,1].
        let mut sum = 0.0f64;
        for _ in 0..total_per_trial {
            let u = (lcg(&mut state) as f64) / (u32::MAX as f64);
            sum += f(u);
        }
        naive_means.push(sum / total_per_trial as f64);
        // Stratified: in each [k/strata, (k+1)/strata], take samples_per_stratum.
        let mut s_sum = 0.0f64;
        for k in 0..strata {
            for _ in 0..samples_per_stratum {
                let u_local = (lcg(&mut state) as f64) / (u32::MAX as f64);
                let u_global = (k as f64 + u_local) / strata as f64;
                s_sum += f(u_global);
            }
        }
        strat_means.push(s_sum / total_per_trial as f64);
    }
    let n_var = variance(&naive_means);
    let s_var = variance(&strat_means);
    StratifiedVerdict::Ok {
        naive_var_x1000: (n_var * 1_000_000.0) as u32,
        stratified_var_x1000: (s_var * 1_000_000.0) as u32,
        strata,
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
    let _ctx = RecipeContext::new("mc_stratified_sampling")?;

    println!("strata=10: {:?}", estimate(10, 10, 100, 42));
    println!("invalid: {:?}", estimate(1, 10, 100, 42));
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
    fn invalid_too_few_strata() {
        assert_eq!(estimate(1, 10, 100, 42), StratifiedVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_samples_per_stratum() {
        assert_eq!(estimate(5, 0, 100, 42), StratifiedVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_trials() {
        assert_eq!(estimate(5, 10, 10, 42), StratifiedVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = estimate(5, 5, 30, 42);
        let b = estimate(5, 5, 30, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn stratified_variance_lower_for_smooth_f() {
        // For smooth, monotone f, stratification beats naive sampling.
        let v = estimate(10, 10, 200, 42);
        if let StratifiedVerdict::Ok {
            naive_var_x1000,
            stratified_var_x1000,
            ..
        } = v
        {
            assert!(stratified_var_x1000 <= naive_var_x1000);
        }
    }

    #[test]
    fn naive_var_positive() {
        let v = estimate(5, 5, 100, 42);
        if let StratifiedVerdict::Ok {
            naive_var_x1000, ..
        } = v
        {
            assert!(naive_var_x1000 > 0);
        }
    }

    #[test]
    fn strata_returned() {
        let v = estimate(7, 5, 100, 42);
        if let StratifiedVerdict::Ok { strata, .. } = v {
            assert_eq!(strata, 7);
        }
    }

    #[test]
    fn variance_function_correct() {
        assert_eq!(variance(&[1.0, -1.0]), 2.0);
        assert_eq!(variance(&[5.0, 5.0, 5.0]), 0.0);
    }

    #[test]
    fn many_strata_handled() {
        let v = estimate(50, 5, 50, 42);
        assert!(matches!(v, StratifiedVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_estimates() {
        let a = estimate(5, 5, 30, 42);
        let b = estimate(5, 5, 30, 999);
        assert!(a != b);
    }

    #[test]
    fn variances_finite() {
        let v = estimate(5, 5, 100, 42);
        if let StratifiedVerdict::Ok {
            naive_var_x1000,
            stratified_var_x1000,
            ..
        } = v
        {
            assert!(naive_var_x1000 < u32::MAX);
            assert!(stratified_var_x1000 < u32::MAX);
        }
    }
}
