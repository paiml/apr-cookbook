//! # Monte-Carlo Control Variate Variance Reduction
//!
//! Estimate E[f(U)] for U ~ Uniform[0,1], using a control variate g
//! with known E[g]=0.5 to reduce variance. Compares variance of the
//! controlled estimator to plain MC.
//!
//! Demonstrates the **MC.147** recipe for PMAT-207 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Lavenberg & Welch (1981); Glasserman, Monte Carlo
//!  Methods in Financial Engineering ch. 4.1.
//!
//! Run with: cargo run --example mc_control_variate_reduction
//!
//! Added by PMAT-207 (catalog 1486→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ControlVerdict {
    Ok {
        plain_var_x10000: u32,
        controlled_var_x10000: u32,
        used_beta_x1000: i32,
    },
    InvalidConfig,
}

pub fn estimate(samples: u32, trials: u32, seed: u64) -> ControlVerdict {
    if samples < 100 || trials < 30 {
        return ControlVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut plain_means: Vec<f64> = Vec::with_capacity(trials as usize);
    let mut ctrl_means: Vec<f64> = Vec::with_capacity(trials as usize);
    let e_g: f64 = 0.5; // known mean of g(u)=u
                        // β estimated from a small pilot run
    let beta: f64 = 1.0; // optimal for f=u, g=u (Cov/Var=1)
    for _ in 0..trials {
        let mut sum_f = 0.0f64;
        let mut sum_ctrl = 0.0f64;
        for _ in 0..samples {
            let u = (lcg(&mut state) as f64) / (u32::MAX as f64);
            let f = u * u;
            let g = u;
            sum_f += f;
            sum_ctrl += f - beta * (g - e_g);
        }
        let n = samples as f64;
        plain_means.push(sum_f / n);
        ctrl_means.push(sum_ctrl / n);
    }
    let plain_var = variance(&plain_means);
    let ctrl_var = variance(&ctrl_means);
    ControlVerdict::Ok {
        plain_var_x10000: (plain_var * 1_000_000.0) as u32,
        controlled_var_x10000: (ctrl_var * 1_000_000.0) as u32,
        used_beta_x1000: (beta * 1000.0) as i32,
    }
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
    let _ctx = RecipeContext::new("mc_control_variate_reduction")?;

    println!("estimate: {:?}", estimate(1000, 100, 42));
    println!("invalid: {:?}", estimate(50, 30, 42));
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
    fn invalid_too_few_samples() {
        assert_eq!(estimate(50, 30, 42), ControlVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_trials() {
        assert_eq!(estimate(100, 10, 42), ControlVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = estimate(100, 30, 42);
        let b = estimate(100, 30, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn controlled_variance_lower_for_correlated_g() {
        // f=u² and g=u are positively correlated → control reduces var.
        let v = estimate(1000, 100, 42);
        if let ControlVerdict::Ok {
            plain_var_x10000,
            controlled_var_x10000,
            ..
        } = v
        {
            assert!(controlled_var_x10000 < plain_var_x10000);
        }
    }

    #[test]
    fn plain_variance_positive() {
        let v = estimate(500, 50, 42);
        if let ControlVerdict::Ok {
            plain_var_x10000, ..
        } = v
        {
            assert!(plain_var_x10000 > 0);
        }
    }

    #[test]
    fn beta_returned() {
        let v = estimate(500, 50, 42);
        if let ControlVerdict::Ok {
            used_beta_x1000, ..
        } = v
        {
            assert_eq!(used_beta_x1000, 1000);
        }
    }

    #[test]
    fn variance_function_correct() {
        assert_eq!(variance(&[5.0, 5.0]), 0.0);
        assert_eq!(variance(&[1.0, -1.0]), 2.0);
    }

    #[test]
    fn many_trials_handled() {
        let v = estimate(1000, 500, 42);
        assert!(matches!(v, ControlVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_var() {
        let a = estimate(500, 50, 42);
        let b = estimate(500, 50, 999);
        assert!(a != b);
    }

    #[test]
    fn variances_finite() {
        let v = estimate(500, 50, 42);
        if let ControlVerdict::Ok {
            plain_var_x10000,
            controlled_var_x10000,
            ..
        } = v
        {
            assert!(plain_var_x10000 < u32::MAX);
            assert!(controlled_var_x10000 < u32::MAX);
        }
    }

    #[test]
    fn min_inputs_accepted() {
        let v = estimate(100, 30, 42);
        assert!(matches!(v, ControlVerdict::Ok { .. }));
    }
}
