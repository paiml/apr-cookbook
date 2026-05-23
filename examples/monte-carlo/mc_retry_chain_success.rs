//! # Monte-Carlo Retry Chain Success
//!
//! Simulate N independent retries each with success probability p.
//! Returns observed success rate over R trials. Formula: 1-(1-p)^N.
//!
//! Demonstrates the **MC.11** recipe for PMAT-161 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Independent Bernoulli trials (Feller).
//!
//! Run with: cargo run --example mc_retry_chain_success
//!
//! Added by PMAT-161 (catalog 1072→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RetryVerdict {
    Ok {
        observed_success_rate: f64,
        analytical_rate: f64,
        avg_attempts_until_success: f64,
    },
    InvalidConfig,
}

pub fn simulate(max_retries: u32, success_prob: f64, num_trials: u32, seed: u64) -> RetryVerdict {
    if max_retries == 0
        || num_trials == 0
        || !success_prob.is_finite()
        || !(0.0..=1.0).contains(&success_prob)
    {
        return RetryVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut successes: u32 = 0;
    let mut total_attempts: u64 = 0;
    for _ in 0..num_trials {
        for attempt in 1..=max_retries {
            if unit(&mut rng_state) < success_prob {
                successes += 1;
                total_attempts += u64::from(attempt);
                break;
            }
            if attempt == max_retries {
                total_attempts += u64::from(max_retries);
            }
        }
    }
    let observed_success_rate = f64::from(successes) / f64::from(num_trials);
    let analytical_rate = 1.0 - (1.0 - success_prob).powi(max_retries as i32);
    let avg_attempts_until_success = if successes > 0 {
        total_attempts as f64 / f64::from(num_trials)
    } else {
        f64::from(max_retries)
    };
    RetryVerdict::Ok {
        observed_success_rate,
        analytical_rate,
        avg_attempts_until_success,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_retry_chain_success")?;

    println!("3 retries, p=0.5: {:?}", simulate(3, 0.5, 1000, 42));
    println!("5 retries, p=0.2: {:?}", simulate(5, 0.2, 1000, 42));
    println!("invalid: {:?}", simulate(0, 0.5, 1000, 42));
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
    fn observed_near_analytical() {
        let v = simulate(3, 0.5, 10_000, 42);
        if let RetryVerdict::Ok {
            observed_success_rate,
            analytical_rate,
            ..
        } = v
        {
            // Analytical = 1 - 0.125 = 0.875.
            assert!((observed_success_rate - analytical_rate).abs() < 0.05);
        }
    }

    #[test]
    fn p_one_always_succeeds_first_try() {
        let v = simulate(5, 1.0, 100, 42);
        if let RetryVerdict::Ok {
            observed_success_rate,
            avg_attempts_until_success,
            ..
        } = v
        {
            assert!((observed_success_rate - 1.0).abs() < 1e-9);
            assert!((avg_attempts_until_success - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn p_zero_never_succeeds() {
        let v = simulate(5, 0.0, 100, 42);
        if let RetryVerdict::Ok {
            observed_success_rate,
            ..
        } = v
        {
            assert!((observed_success_rate - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn invalid_zero_retries() {
        assert_eq!(simulate(0, 0.5, 100, 42), RetryVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_trials() {
        assert_eq!(simulate(5, 0.5, 0, 42), RetryVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_negative_p() {
        assert_eq!(simulate(5, -0.1, 100, 42), RetryVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_over_one_p() {
        assert_eq!(simulate(5, 1.5, 100, 42), RetryVerdict::InvalidConfig);
    }

    #[test]
    fn nan_p_invalid() {
        assert_eq!(simulate(5, f64::NAN, 100, 42), RetryVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(3, 0.5, 100, 42);
        let b = simulate(3, 0.5, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn more_retries_higher_success() {
        let few = simulate(1, 0.3, 1000, 42);
        let many = simulate(10, 0.3, 1000, 42);
        if let (
            RetryVerdict::Ok {
                observed_success_rate: f,
                ..
            },
            RetryVerdict::Ok {
                observed_success_rate: m,
                ..
            },
        ) = (few, many)
        {
            assert!(m > f);
        }
    }
}
