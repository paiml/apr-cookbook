//! # Monte-Carlo Replica Lag Distribution
//!
//! Sim async-replica lag (writer commits before reader catches up).
//! Lag = uniform(min, max) per replica per sample. Returns observed
//! mean / max and proportion exceeding the SLO threshold.
//!
//! Demonstrates the **MC.31** recipe for PMAT-168 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Async replication lag in distributed systems.
//!
//! Run with: cargo run --example mc_replica_lag_distribution
//!
//! Added by PMAT-168 (catalog 1135→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum LagVerdict {
    Ok {
        mean_lag_ms: f64,
        max_lag_ms: f64,
        breach_rate: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    min_lag_ms: f64,
    max_lag_ms: f64,
    slo_threshold_ms: f64,
    samples: u32,
    seed: u64,
) -> LagVerdict {
    if !min_lag_ms.is_finite()
        || !max_lag_ms.is_finite()
        || min_lag_ms < 0.0
        || max_lag_ms < min_lag_ms
        || !slo_threshold_ms.is_finite()
        || slo_threshold_ms < 0.0
        || samples == 0
    {
        return LagVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut sum = 0.0;
    let mut max_v: f64 = 0.0;
    let mut breach = 0u32;
    for _ in 0..samples {
        let lag = min_lag_ms + (max_lag_ms - min_lag_ms) * unit(&mut rng_state);
        sum += lag;
        if lag > max_v {
            max_v = lag;
        }
        if lag > slo_threshold_ms {
            breach += 1;
        }
    }
    let mean_lag_ms = sum / f64::from(samples);
    let breach_rate = f64::from(breach) / f64::from(samples);
    LagVerdict::Ok {
        mean_lag_ms,
        max_lag_ms: max_v,
        breach_rate,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_replica_lag_distribution")?;

    println!("typical: {:?}", simulate(10.0, 50.0, 100.0, 10_000, 42));
    println!(
        "frequent breach: {:?}",
        simulate(80.0, 200.0, 100.0, 10_000, 42)
    );
    println!("invalid: {:?}", simulate(50.0, 10.0, 100.0, 100, 42));
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
    fn typical_low_breach() {
        let v = simulate(10.0, 50.0, 100.0, 10_000, 42);
        if let LagVerdict::Ok { breach_rate, .. } = v {
            assert!(breach_rate < 0.01);
        }
    }

    #[test]
    fn frequent_breach_high_rate() {
        let v = simulate(80.0, 200.0, 100.0, 10_000, 42);
        if let LagVerdict::Ok { breach_rate, .. } = v {
            assert!(breach_rate > 0.5);
        }
    }

    #[test]
    fn mean_in_range() {
        let v = simulate(10.0, 50.0, 100.0, 10_000, 42);
        if let LagVerdict::Ok { mean_lag_ms, .. } = v {
            assert!(mean_lag_ms >= 10.0 && mean_lag_ms <= 50.0);
        }
    }

    #[test]
    fn invalid_max_below_min() {
        assert_eq!(
            simulate(50.0, 10.0, 100.0, 100, 42),
            LagVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(
            simulate(10.0, 50.0, 100.0, 0, 42),
            LagVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_neg_threshold() {
        assert_eq!(
            simulate(10.0, 50.0, -1.0, 100, 42),
            LagVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(f64::NAN, 50.0, 100.0, 100, 42),
            LagVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(10.0, 50.0, 100.0, 1000, 42);
        let b = simulate(10.0, 50.0, 100.0, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn breach_rate_in_unit_range() {
        let v = simulate(10.0, 50.0, 100.0, 1000, 42);
        if let LagVerdict::Ok { breach_rate, .. } = v {
            assert!((0.0..=1.0).contains(&breach_rate));
        }
    }

    #[test]
    fn equal_min_max_constant_lag() {
        let v = simulate(50.0, 50.0, 100.0, 1000, 42);
        if let LagVerdict::Ok {
            mean_lag_ms,
            max_lag_ms,
            ..
        } = v
        {
            assert!((mean_lag_ms - 50.0).abs() < 1e-9);
            assert!((max_lag_ms - 50.0).abs() < 1e-9);
        }
    }
}
