//! # Monte-Carlo Gibbs Sampler for Bivariate Distribution
//!
//! Sample from a bivariate normal-like distribution where conditionals
//! are simple to compute: x | y ~ centered at correlation ρ·y, and
//! vice versa. Returns sample means and the empirical correlation.
//!
//! Demonstrates the **MC.156** recipe for PMAT-210 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Geman & Geman, "Stochastic Relaxation, Gibbs
//!  Distributions" IEEE PAMI (1984); Casella & George (1992)
//!  Bayesian intro to Gibbs sampling.
//!
//! Run with: cargo run --example mc_gibbs_sampler_bivariate
//!
//! Added by PMAT-210 (catalog 1513→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GibbsVerdict {
    Ok {
        mean_x_x1000: i32,
        mean_y_x1000: i32,
        empirical_corr_x1000: i32,
    },
    InvalidConfig,
}

pub fn simulate(rho_x100: i32, samples: u32, burnin: u32, seed: u64) -> GibbsVerdict {
    if !(-99..=99).contains(&rho_x100) || samples < 100 {
        return GibbsVerdict::InvalidConfig;
    }
    let rho = rho_x100 as f64 / 100.0;
    let sigma_cond = (1.0 - rho * rho).sqrt();
    let mut state = seed | 1;
    let mut x;
    let mut y = 0.0f64;
    let mut xs: Vec<f64> = Vec::with_capacity(samples as usize);
    let mut ys: Vec<f64> = Vec::with_capacity(samples as usize);
    let total = burnin + samples;
    for i in 0..total {
        // Conditional draws via Box-Muller-like uniform-mean approximation
        x = rho * y + sigma_cond * box_muller(&mut state);
        y = rho * x + sigma_cond * box_muller(&mut state);
        if i >= burnin {
            xs.push(x);
            ys.push(y);
        }
    }
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let cov = xs
        .iter()
        .zip(ys.iter())
        .map(|(a, b)| (*a - mean_x) * (*b - mean_y))
        .sum::<f64>()
        / n;
    let var_x = xs.iter().map(|a| (a - mean_x).powi(2)).sum::<f64>() / n;
    let var_y = ys.iter().map(|b| (b - mean_y).powi(2)).sum::<f64>() / n;
    let corr = cov / (var_x.sqrt() * var_y.sqrt());
    GibbsVerdict::Ok {
        mean_x_x1000: (mean_x * 1000.0) as i32,
        mean_y_x1000: (mean_y * 1000.0) as i32,
        empirical_corr_x1000: (corr * 1000.0) as i32,
    }
}

fn box_muller(state: &mut u64) -> f64 {
    let u1 = (lcg(state) as f64 / u32::MAX as f64).max(1e-10);
    let u2 = lcg(state) as f64 / u32::MAX as f64;
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_gibbs_sampler_bivariate")?;

    println!("rho=0.5: {:?}", simulate(50, 5000, 500, 42));
    println!("invalid: {:?}", simulate(101, 5000, 500, 42));
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
    fn invalid_rho_too_high() {
        assert_eq!(simulate(100, 1000, 100, 42), GibbsVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_rho_too_low() {
        assert_eq!(simulate(-100, 1000, 100, 42), GibbsVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_samples() {
        assert_eq!(simulate(50, 50, 100, 42), GibbsVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(50, 500, 100, 42);
        let b = simulate(50, 500, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn means_near_zero() {
        let v = simulate(50, 10_000, 1000, 42);
        if let GibbsVerdict::Ok {
            mean_x_x1000,
            mean_y_x1000,
            ..
        } = v
        {
            assert!(mean_x_x1000.abs() < 200);
            assert!(mean_y_x1000.abs() < 200);
        }
    }

    #[test]
    fn empirical_corr_close_to_rho() {
        // ρ=0.5 → empirical correlation ~500 (×1000).
        let v = simulate(50, 20_000, 1000, 42);
        if let GibbsVerdict::Ok {
            empirical_corr_x1000,
            ..
        } = v
        {
            assert!((350..=650).contains(&empirical_corr_x1000));
        }
    }

    #[test]
    fn negative_rho_negative_corr() {
        let v = simulate(-50, 10_000, 1000, 42);
        if let GibbsVerdict::Ok {
            empirical_corr_x1000,
            ..
        } = v
        {
            assert!(empirical_corr_x1000 < 0);
        }
    }

    #[test]
    fn zero_rho_zero_corr() {
        let v = simulate(0, 10_000, 1000, 42);
        if let GibbsVerdict::Ok {
            empirical_corr_x1000,
            ..
        } = v
        {
            assert!(empirical_corr_x1000.abs() < 200);
        }
    }

    #[test]
    fn min_samples_accepted() {
        let v = simulate(50, 100, 0, 42);
        assert!(matches!(v, GibbsVerdict::Ok { .. }));
    }

    #[test]
    fn many_samples_handled() {
        let v = simulate(50, 50_000, 1000, 42);
        assert!(matches!(v, GibbsVerdict::Ok { .. }));
    }

    #[test]
    fn corr_in_minus_one_one() {
        let v = simulate(50, 5000, 100, 42);
        if let GibbsVerdict::Ok {
            empirical_corr_x1000,
            ..
        } = v
        {
            assert!((-1100..=1100).contains(&empirical_corr_x1000));
        }
    }
}
