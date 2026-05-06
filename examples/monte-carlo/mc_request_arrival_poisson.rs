//! # Monte-Carlo Poisson Arrival Simulator
//!
//! Generate inter-arrival times from an exponential distribution with
//! mean 1/lambda. Returns observed arrival rate, mean inter-arrival,
//! and total time elapsed.
//!
//! Demonstrates the **MC.16** recipe for PMAT-163 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Poisson process generation (Inverse-CDF method).
//!
//! Run with: cargo run --example mc_request_arrival_poisson
//!
//! Added by PMAT-163 (catalog 1090→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ArrivalVerdict {
    Ok {
        observed_rate: f64,
        mean_interarrival: f64,
        total_time_secs: f64,
    },
    InvalidConfig,
}

pub fn simulate(lambda: f64, n_arrivals: u32, seed: u64) -> ArrivalVerdict {
    if !lambda.is_finite() || lambda <= 0.0 || n_arrivals == 0 {
        return ArrivalVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut total_time = 0.0;
    let mut sum = 0.0;
    for _ in 0..n_arrivals {
        // Exponential(lambda) via inverse CDF: -ln(U)/lambda where U is uniform(0,1].
        let u = unit(&mut rng_state).max(1e-12);
        let interarrival = -u.ln() / lambda;
        total_time += interarrival;
        sum += interarrival;
    }
    let mean_interarrival = sum / f64::from(n_arrivals);
    let observed_rate = if total_time > 0.0 {
        f64::from(n_arrivals) / total_time
    } else {
        0.0
    };
    ArrivalVerdict::Ok {
        observed_rate,
        mean_interarrival,
        total_time_secs: total_time,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_request_arrival_poisson")?;

    println!("rate=10, n=1000: {:?}", simulate(10.0, 1000, 42));
    println!("rate=100, n=10000: {:?}", simulate(100.0, 10_000, 42));
    println!("invalid: {:?}", simulate(-1.0, 100, 42));
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
    fn observed_rate_near_lambda() {
        let v = simulate(10.0, 10_000, 42);
        if let ArrivalVerdict::Ok { observed_rate, .. } = v {
            // Law of large numbers: observed → lambda.
            assert!((observed_rate - 10.0).abs() < 0.5);
        }
    }

    #[test]
    fn mean_interarrival_near_inv_lambda() {
        let v = simulate(10.0, 10_000, 42);
        if let ArrivalVerdict::Ok {
            mean_interarrival, ..
        } = v
        {
            assert!((mean_interarrival - 0.1).abs() < 0.01);
        }
    }

    #[test]
    fn invalid_zero_lambda() {
        assert_eq!(simulate(0.0, 100, 42), ArrivalVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_negative_lambda() {
        assert_eq!(simulate(-1.0, 100, 42), ArrivalVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_arrivals() {
        assert_eq!(simulate(10.0, 0, 42), ArrivalVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(simulate(f64::NAN, 100, 42), ArrivalVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(10.0, 1000, 42);
        let b = simulate(10.0, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn higher_lambda_more_frequent() {
        let slow = simulate(1.0, 1000, 42);
        let fast = simulate(100.0, 1000, 42);
        if let (
            ArrivalVerdict::Ok {
                total_time_secs: ts,
                ..
            },
            ArrivalVerdict::Ok {
                total_time_secs: tf,
                ..
            },
        ) = (slow, fast)
        {
            assert!(ts > tf);
        }
    }

    #[test]
    fn total_time_positive() {
        let v = simulate(10.0, 100, 42);
        if let ArrivalVerdict::Ok {
            total_time_secs, ..
        } = v
        {
            assert!(total_time_secs > 0.0);
        }
    }

    #[test]
    fn small_n_works() {
        let v = simulate(10.0, 5, 42);
        assert!(matches!(v, ArrivalVerdict::Ok { .. }));
    }
}
