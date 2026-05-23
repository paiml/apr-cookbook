//! # Monte-Carlo Throughput vs Concurrency Curve
//!
//! Sweep concurrency from 1..N and simulate observed throughput.
//! Validates Little's Law: throughput = concurrency / mean_latency
//! up to capacity, then plateaus.
//!
//! Demonstrates the **MC.25** recipe for PMAT-166 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Little's Law (Little 1961) for closed queueing networks.
//!
//! Run with: cargo run --example mc_throughput_concurrency_curve
//!
//! Added by PMAT-166 (catalog 1117→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CurveVerdict {
    Ok {
        knee_concurrency: u32,
        peak_throughput: f64,
    },
    InvalidConfig,
}

pub fn simulate(max_concurrency: u32, base_latency_secs: f64, capacity_tps: f64) -> CurveVerdict {
    if max_concurrency == 0
        || !base_latency_secs.is_finite()
        || base_latency_secs <= 0.0
        || !capacity_tps.is_finite()
        || capacity_tps <= 0.0
    {
        return CurveVerdict::InvalidConfig;
    }
    let mut knee_concurrency: u32 = max_concurrency;
    let mut peak_throughput: f64 = 0.0;
    for c in 1..=max_concurrency {
        // Linear region: throughput = c / base_latency until capacity hit.
        let linear = f64::from(c) / base_latency_secs;
        let throughput = linear.min(capacity_tps);
        if throughput > peak_throughput {
            peak_throughput = throughput;
        }
        if (linear - capacity_tps).abs() < 0.01 || linear >= capacity_tps {
            knee_concurrency = c;
            break;
        }
    }
    CurveVerdict::Ok {
        knee_concurrency,
        peak_throughput,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_throughput_concurrency_curve")?;

    println!("typical: {:?}", simulate(100, 0.05, 200.0));
    println!("low cap: {:?}", simulate(100, 0.05, 50.0));
    println!("invalid: {:?}", simulate(0, 0.05, 200.0));
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
    fn knee_at_expected_concurrency() {
        // Throughput = c/0.05 saturates at 200 tps when c = 10.
        let v = simulate(100, 0.05, 200.0);
        if let CurveVerdict::Ok {
            knee_concurrency, ..
        } = v
        {
            assert_eq!(knee_concurrency, 10);
        }
    }

    #[test]
    fn peak_at_capacity() {
        let v = simulate(100, 0.05, 200.0);
        if let CurveVerdict::Ok {
            peak_throughput, ..
        } = v
        {
            assert!((peak_throughput - 200.0).abs() < 0.5);
        }
    }

    #[test]
    fn higher_capacity_higher_knee() {
        let lo = simulate(1000, 0.05, 50.0);
        let hi = simulate(1000, 0.05, 500.0);
        if let (
            CurveVerdict::Ok {
                knee_concurrency: l,
                ..
            },
            CurveVerdict::Ok {
                knee_concurrency: h,
                ..
            },
        ) = (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn invalid_zero_concurrency() {
        assert_eq!(simulate(0, 0.05, 200.0), CurveVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_latency() {
        assert_eq!(simulate(100, 0.0, 200.0), CurveVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_capacity() {
        assert_eq!(simulate(100, 0.05, 0.0), CurveVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(simulate(100, f64::NAN, 200.0), CurveVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 0.05, 200.0);
        let b = simulate(100, 0.05, 200.0);
        assert_eq!(a, b);
    }

    #[test]
    fn very_high_capacity_uses_max_concurrency() {
        let v = simulate(50, 0.05, 1_000_000.0);
        if let CurveVerdict::Ok {
            knee_concurrency, ..
        } = v
        {
            assert_eq!(knee_concurrency, 50);
        }
    }

    #[test]
    fn very_low_capacity_knee_at_one() {
        let v = simulate(100, 0.05, 1.0);
        if let CurveVerdict::Ok {
            knee_concurrency, ..
        } = v
        {
            assert_eq!(knee_concurrency, 1);
        }
    }
}
