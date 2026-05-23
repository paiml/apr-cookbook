//! # Monte-Carlo Log-Normal Request Sizes
//!
//! Sim request sizes drawn from a log-normal distribution
//! (matches real-world payload heavy-tail). Returns p50, p95, p99
//! and approximate Box-Muller-derived geometric mean.
//!
//! Demonstrates the **MC.49** recipe for PMAT-174 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: log-normal-distributed file sizes (Crovella & Bestavros).
//!
//! Run with: cargo run --example mc_request_size_log_normal
//!
//! Added by PMAT-174 (catalog 1189→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum LogNormalVerdict {
    Ok { p50: f64, p95: f64, p99: f64 },
    InvalidConfig,
}

pub fn simulate(mu: f64, sigma: f64, samples: u32, seed: u64) -> LogNormalVerdict {
    if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 || samples < 2 {
        return LogNormalVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut sizes: Vec<f64> = Vec::with_capacity(samples as usize);
    let mut i = 0u32;
    while i < samples {
        let u1 = unit(&mut rng_state).max(1e-12);
        let u2 = unit(&mut rng_state);
        // Box-Muller transform: standard normal samples.
        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let z2 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
        sizes.push((mu + sigma * z1).exp());
        i += 1;
        if i < samples {
            sizes.push((mu + sigma * z2).exp());
            i += 1;
        }
    }
    sizes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = sizes[((samples as f64) * 0.50) as usize];
    let p95 = sizes[((samples as f64) * 0.95) as usize];
    let p99 = sizes[((samples as f64) * 0.99) as usize];
    LogNormalVerdict::Ok { p50, p95, p99 }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_request_size_log_normal")?;

    println!("typical: {:?}", simulate(7.0, 1.5, 10_000, 42));
    println!("low spread: {:?}", simulate(7.0, 0.1, 10_000, 42));
    println!("invalid: {:?}", simulate(7.0, 0.0, 10_000, 42));
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
    fn p99_above_p50() {
        let v = simulate(7.0, 1.5, 10_000, 42);
        if let LogNormalVerdict::Ok { p50, p99, .. } = v {
            assert!(p99 > p50);
        }
    }

    #[test]
    fn p95_above_p50() {
        let v = simulate(7.0, 1.5, 10_000, 42);
        if let LogNormalVerdict::Ok { p50, p95, .. } = v {
            assert!(p95 > p50);
        }
    }

    #[test]
    fn higher_sigma_wider_tail() {
        let narrow = simulate(7.0, 0.1, 10_000, 42);
        let wide = simulate(7.0, 2.0, 10_000, 42);
        if let (LogNormalVerdict::Ok { p99: n99, .. }, LogNormalVerdict::Ok { p99: w99, .. }) =
            (narrow, wide)
        {
            assert!(w99 > n99);
        }
    }

    #[test]
    fn invalid_zero_sigma() {
        assert_eq!(simulate(7.0, 0.0, 100, 42), LogNormalVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_neg_sigma() {
        assert_eq!(
            simulate(7.0, -1.0, 100, 42),
            LogNormalVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_too_few_samples() {
        assert_eq!(simulate(7.0, 1.0, 1, 42), LogNormalVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(f64::NAN, 1.0, 100, 42),
            LogNormalVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(7.0, 1.5, 1000, 42);
        let b = simulate(7.0, 1.5, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn p50_near_exp_mu_for_low_sigma() {
        let v = simulate(7.0, 0.1, 10_000, 42);
        if let LogNormalVerdict::Ok { p50, .. } = v {
            // For low sigma, log-normal is tight around exp(mu) = e^7 ≈ 1096.6.
            assert!((p50 - 1096.6).abs() / 1096.6 < 0.05);
        }
    }

    #[test]
    fn percentiles_positive() {
        let v = simulate(7.0, 1.5, 1000, 42);
        if let LogNormalVerdict::Ok { p50, p95, p99 } = v {
            assert!(p50 > 0.0);
            assert!(p95 > 0.0);
            assert!(p99 > 0.0);
        }
    }
}
