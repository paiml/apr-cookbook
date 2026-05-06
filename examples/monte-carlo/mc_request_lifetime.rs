//! # Monte-Carlo Request Lifetime Distribution
//!
//! Sim per-request lifetime: arrival → enqueue → execute → respond.
//! Returns mean lifetime, queue-time fraction, and execution-time
//! fraction.
//!
//! Demonstrates the **MC.35** recipe for PMAT-169 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Request-lifecycle modeling (USL — Universal Scalability Law).
//!
//! Run with: cargo run --example mc_request_lifetime
//!
//! Added by PMAT-169 (catalog 1144→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum LifetimeVerdict {
    Ok {
        mean_lifetime_ms: f64,
        queue_pct: f64,
        execute_pct: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    queue_wait_ms_mean: f64,
    execute_ms_mean: f64,
    samples: u32,
    seed: u64,
) -> LifetimeVerdict {
    if !queue_wait_ms_mean.is_finite()
        || queue_wait_ms_mean < 0.0
        || !execute_ms_mean.is_finite()
        || execute_ms_mean <= 0.0
        || samples == 0
    {
        return LifetimeVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut total_queue = 0.0;
    let mut total_exec = 0.0;
    for _ in 0..samples {
        // Exponential queue wait + uniform execute time.
        let u_q = unit(&mut rng_state).max(1e-12);
        let queue = -u_q.ln() * queue_wait_ms_mean;
        let exec = execute_ms_mean * (0.5 + unit(&mut rng_state));
        total_queue += queue;
        total_exec += exec;
    }
    let n = f64::from(samples);
    let mean_q = total_queue / n;
    let mean_e = total_exec / n;
    let mean_lifetime_ms = mean_q + mean_e;
    let queue_pct = if mean_lifetime_ms > 0.0 {
        (mean_q / mean_lifetime_ms) * 100.0
    } else {
        0.0
    };
    let execute_pct = if mean_lifetime_ms > 0.0 {
        (mean_e / mean_lifetime_ms) * 100.0
    } else {
        0.0
    };
    LifetimeVerdict::Ok {
        mean_lifetime_ms,
        queue_pct,
        execute_pct,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_request_lifetime")?;

    println!("light queue: {:?}", simulate(1.0, 50.0, 10_000, 42));
    println!("heavy queue: {:?}", simulate(100.0, 50.0, 10_000, 42));
    println!("invalid: {:?}", simulate(-1.0, 50.0, 100, 42));
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
    fn light_queue_low_pct() {
        let v = simulate(1.0, 50.0, 10_000, 42);
        if let LifetimeVerdict::Ok { queue_pct, .. } = v {
            assert!(queue_pct < 5.0);
        }
    }

    #[test]
    fn heavy_queue_high_pct() {
        let v = simulate(100.0, 50.0, 10_000, 42);
        if let LifetimeVerdict::Ok { queue_pct, .. } = v {
            assert!(queue_pct > 50.0);
        }
    }

    #[test]
    fn percentages_sum_to_100() {
        let v = simulate(10.0, 50.0, 10_000, 42);
        if let LifetimeVerdict::Ok {
            queue_pct,
            execute_pct,
            ..
        } = v
        {
            assert!((queue_pct + execute_pct - 100.0).abs() < 1e-6);
        }
    }

    #[test]
    fn invalid_neg_queue() {
        assert_eq!(
            simulate(-1.0, 50.0, 100, 42),
            LifetimeVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_execute() {
        assert_eq!(simulate(10.0, 0.0, 100, 42), LifetimeVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(simulate(10.0, 50.0, 0, 42), LifetimeVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(f64::NAN, 50.0, 100, 42),
            LifetimeVerdict::InvalidConfig
        );
    }

    #[test]
    fn zero_queue_all_execute() {
        let v = simulate(0.0, 50.0, 1000, 42);
        if let LifetimeVerdict::Ok {
            queue_pct,
            execute_pct,
            ..
        } = v
        {
            assert!(queue_pct < 0.5);
            assert!(execute_pct > 99.0);
        }
    }

    #[test]
    fn deterministic() {
        let a = simulate(10.0, 50.0, 1000, 42);
        let b = simulate(10.0, 50.0, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn lifetime_positive() {
        let v = simulate(10.0, 50.0, 100, 42);
        if let LifetimeVerdict::Ok {
            mean_lifetime_ms, ..
        } = v
        {
            assert!(mean_lifetime_ms > 0.0);
        }
    }
}
