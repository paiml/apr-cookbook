//! # Monte-Carlo Data-Freshness Lag
//!
//! Simulate the lag distribution between data ingest and inference
//! request. Lag = exponential(rate) + fixed_overhead. Returns p50,
//! p95, p99 lag observed.
//!
//! Demonstrates the **MC.27** recipe for PMAT-166 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Data-pipeline freshness SLOs (Datadog ingest lag).
//!
//! Run with: cargo run --example mc_data_freshness_lag
//!
//! Added by PMAT-166 (catalog 1117→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum LagVerdict {
    Ok { p50: f64, p95: f64, p99: f64 },
    InvalidConfig,
}

pub fn simulate(rate: f64, fixed_overhead_secs: f64, samples: u32, seed: u64) -> LagVerdict {
    if !rate.is_finite()
        || rate <= 0.0
        || !fixed_overhead_secs.is_finite()
        || fixed_overhead_secs < 0.0
        || samples == 0
    {
        return LagVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut lags: Vec<f64> = Vec::with_capacity(samples as usize);
    for _ in 0..samples {
        let u = unit(&mut rng_state).max(1e-12);
        let exp_part = -u.ln() / rate;
        lags.push(exp_part + fixed_overhead_secs);
    }
    lags.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = lags.len();
    let p50 = lags[(n as f64 * 0.50) as usize];
    let p95 = lags[((n as f64 * 0.95) as usize).min(n - 1)];
    let p99 = lags[((n as f64 * 0.99) as usize).min(n - 1)];
    LagVerdict::Ok { p50, p95, p99 }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_data_freshness_lag")?;

    println!("typical: {:?}", simulate(2.0, 0.5, 10_000, 42));
    println!("fast: {:?}", simulate(10.0, 0.05, 10_000, 42));
    println!("invalid: {:?}", simulate(0.0, 0.5, 10_000, 42));
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
        let v = simulate(2.0, 0.5, 10_000, 42);
        if let LagVerdict::Ok { p50, p99, .. } = v {
            assert!(p99 > p50);
        }
    }

    #[test]
    fn p95_above_p50() {
        let v = simulate(2.0, 0.5, 10_000, 42);
        if let LagVerdict::Ok { p50, p95, .. } = v {
            assert!(p95 > p50);
        }
    }

    #[test]
    fn p99_above_p95() {
        let v = simulate(2.0, 0.5, 10_000, 42);
        if let LagVerdict::Ok { p95, p99, .. } = v {
            assert!(p99 >= p95);
        }
    }

    #[test]
    fn fixed_overhead_floors_p50() {
        let v = simulate(2.0, 5.0, 10_000, 42);
        if let LagVerdict::Ok { p50, .. } = v {
            assert!(p50 >= 5.0);
        }
    }

    #[test]
    fn invalid_zero_rate() {
        assert_eq!(simulate(0.0, 0.5, 100, 42), LagVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_negative_overhead() {
        assert_eq!(simulate(2.0, -0.5, 100, 42), LagVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(simulate(2.0, 0.5, 0, 42), LagVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(simulate(f64::NAN, 0.5, 100, 42), LagVerdict::InvalidConfig);
    }

    #[test]
    fn higher_rate_lower_p50() {
        let slow = simulate(0.5, 0.0, 10_000, 42);
        let fast = simulate(10.0, 0.0, 10_000, 42);
        if let (LagVerdict::Ok { p50: s, .. }, LagVerdict::Ok { p50: f, .. }) = (slow, fast) {
            assert!(f < s);
        }
    }

    #[test]
    fn deterministic() {
        let a = simulate(2.0, 0.5, 1000, 42);
        let b = simulate(2.0, 0.5, 1000, 42);
        assert_eq!(a, b);
    }
}
