//! # Monte-Carlo Negative Binomial Overdispersion
//!
//! Sample from a negative binomial via Gamma-Poisson mixture; verify
//! variance > mean (overdispersion). Returns mean and variance
//! estimates (×100 fixed).
//!
//! Demonstrates the **MC.177** recipe for PMAT-217 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Cameron & Trivedi, Regression Analysis of Count Data
//!  ch. 4 (2013); negative binomial as Gamma-Poisson mixture.
//!
//! Run with: cargo run --example mc_negative_binomial_overdisp
//!
//! Added by PMAT-217 (catalog 1576→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum NegBinVerdict {
    Ok {
        sample_mean_x100: u32,
        sample_var_x100: u32,
    },
    InvalidConfig,
}

pub fn simulate(mu_x100: u32, dispersion_x100: u32, samples: u32, seed: u64) -> NegBinVerdict {
    if mu_x100 == 0 || dispersion_x100 == 0 || samples < 100 {
        return NegBinVerdict::InvalidConfig;
    }
    let mu = mu_x100 as f64 / 100.0;
    let r = dispersion_x100 as f64 / 100.0;
    let mut state = seed | 1;
    let mut values: Vec<u32> = Vec::with_capacity(samples as usize);
    for _ in 0..samples {
        // Gamma(shape=r, scale=mu/r) → λ; then Poisson(λ).
        let lambda = sample_gamma(r, mu / r, &mut state);
        let k = sample_poisson(lambda, &mut state);
        values.push(k);
    }
    let n = values.len() as f64;
    let mean = values.iter().map(|v| *v as f64).sum::<f64>() / n;
    let var = values
        .iter()
        .map(|v| (*v as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    NegBinVerdict::Ok {
        sample_mean_x100: (mean * 100.0) as u32,
        sample_var_x100: (var * 100.0) as u32,
    }
}

fn sample_gamma(shape: f64, scale: f64, state: &mut u64) -> f64 {
    // Marsaglia-Tsang shape ≥ 1 approximation; for shape < 1, use shape+1 + adjust.
    let s = if shape >= 1.0 { shape } else { shape + 1.0 };
    let d = s - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = box_muller(state);
        let v_3 = 1.0 + c * x;
        if v_3 <= 0.0 {
            continue;
        }
        let v = v_3 * v_3 * v_3;
        let u = (lcg(state) as f64) / (u32::MAX as f64);
        if u < 1.0 - 0.0331 * x.powi(4) {
            let g = d * v * scale;
            if shape >= 1.0 {
                return g;
            }
            // Adjust for shape < 1
            let u2 = ((lcg(state) as f64) / (u32::MAX as f64)).max(1e-10);
            return g * u2.powf(1.0 / shape);
        }
        let lhs = 0.5 * x * x + d * (1.0 - v + v.ln());
        if u.ln() < lhs {
            let g = d * v * scale;
            if shape >= 1.0 {
                return g;
            }
            let u2 = ((lcg(state) as f64) / (u32::MAX as f64)).max(1e-10);
            return g * u2.powf(1.0 / shape);
        }
    }
}

fn sample_poisson(lambda: f64, state: &mut u64) -> u32 {
    // Knuth algorithm
    let l = (-lambda).exp();
    let mut k = 0u32;
    let mut p = 1.0f64;
    while p > l {
        k += 1;
        let u = (lcg(state) as f64) / (u32::MAX as f64);
        p *= u;
        if k > 10_000 {
            break;
        }
    }
    k - 1
}

fn box_muller(state: &mut u64) -> f64 {
    let u1 = ((lcg(state) as f64) / (u32::MAX as f64)).max(1e-10);
    let u2 = (lcg(state) as f64) / (u32::MAX as f64);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_negative_binomial_overdisp")?;

    println!("μ=5, r=2: {:?}", simulate(500, 200, 5000, 42));
    println!("invalid: {:?}", simulate(0, 200, 5000, 42));
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
    fn invalid_zero_mu() {
        assert_eq!(simulate(0, 200, 5000, 42), NegBinVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_dispersion() {
        assert_eq!(simulate(500, 0, 5000, 42), NegBinVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_samples() {
        assert_eq!(simulate(500, 200, 50, 42), NegBinVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 200, 1000, 42);
        let b = simulate(500, 200, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn variance_exceeds_mean_overdispersion() {
        // For NB: var = mu + mu^2/r > mu when r is finite.
        let v = simulate(500, 200, 5000, 42);
        if let NegBinVerdict::Ok {
            sample_mean_x100,
            sample_var_x100,
        } = v
        {
            // var should exceed mean for overdispersed data.
            assert!(sample_var_x100 > sample_mean_x100);
        }
    }

    #[test]
    fn mean_finite() {
        let v = simulate(500, 200, 1000, 42);
        if let NegBinVerdict::Ok {
            sample_mean_x100, ..
        } = v
        {
            assert!(sample_mean_x100 < u32::MAX);
        }
    }

    #[test]
    fn variance_finite() {
        let v = simulate(500, 200, 1000, 42);
        if let NegBinVerdict::Ok {
            sample_var_x100, ..
        } = v
        {
            assert!(sample_var_x100 < u32::MAX);
        }
    }

    #[test]
    fn min_samples_accepted() {
        let v = simulate(500, 200, 100, 42);
        assert!(matches!(v, NegBinVerdict::Ok { .. }));
    }

    #[test]
    fn many_samples_handled() {
        let v = simulate(500, 200, 50_000, 42);
        assert!(matches!(v, NegBinVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_outcomes() {
        let a = simulate(500, 200, 500, 42);
        let b = simulate(500, 200, 500, 999);
        assert!(a != b);
    }

    #[test]
    fn higher_dispersion_lower_var_ratio() {
        // Larger r → less overdispersion (var/mean closer to 1).
        let low_r = simulate(500, 100, 10_000, 42);
        let high_r = simulate(500, 1000, 10_000, 42);
        if let (
            NegBinVerdict::Ok {
                sample_mean_x100: m_low,
                sample_var_x100: v_low,
            },
            NegBinVerdict::Ok {
                sample_mean_x100: m_high,
                sample_var_x100: v_high,
            },
        ) = (low_r, high_r)
        {
            // Ratio var/mean for low-r should exceed ratio for high-r.
            let ratio_low = v_low as f64 / m_low.max(1) as f64;
            let ratio_high = v_high as f64 / m_high.max(1) as f64;
            assert!(ratio_low > ratio_high);
        }
    }
}
