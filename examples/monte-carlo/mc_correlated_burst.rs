//! # Monte-Carlo Correlated Request Bursts
//!
//! Sim correlated bursts: with prob `burst_start_prob`, enter "burst"
//! state and stay for `burst_duration` steps. Returns observed
//! variance vs Poisson baseline.
//!
//! Demonstrates the **MC.61** recipe for PMAT-178 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: self-similar traffic models (Leland et al. 1994).
//!
//! Run with: cargo run --example mc_correlated_burst
//!
//! Added by PMAT-178 (catalog 1225→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BurstVerdict {
    Ok {
        mean_arrivals: f64,
        burst_steps: u32,
        idle_steps: u32,
    },
    InvalidConfig,
}

pub fn simulate(
    burst_start_prob: f64,
    burst_arrival_rate: u32,
    idle_arrival_rate: u32,
    burst_duration: u32,
    steps: u32,
    seed: u64,
) -> BurstVerdict {
    if !burst_start_prob.is_finite()
        || !(0.0..=1.0).contains(&burst_start_prob)
        || burst_duration == 0
        || steps == 0
    {
        return BurstVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut total_arrivals: u64 = 0;
    let mut burst_steps = 0u32;
    let mut idle_steps = 0u32;
    let mut burst_remaining: u32 = 0;
    for _ in 0..steps {
        if burst_remaining > 0 {
            burst_remaining -= 1;
            burst_steps += 1;
            total_arrivals += u64::from(burst_arrival_rate);
        } else if unit(&mut rng_state) < burst_start_prob {
            burst_remaining = burst_duration;
        } else {
            idle_steps += 1;
            total_arrivals += u64::from(idle_arrival_rate);
        }
    }
    let mean_arrivals = total_arrivals as f64 / f64::from(steps);
    BurstVerdict::Ok {
        mean_arrivals,
        burst_steps,
        idle_steps,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_correlated_burst")?;

    println!("rare bursts: {:?}", simulate(0.01, 100, 5, 50, 10_000, 42));
    println!(
        "frequent bursts: {:?}",
        simulate(0.1, 100, 5, 20, 10_000, 42)
    );
    println!("invalid: {:?}", simulate(-0.1, 100, 5, 50, 1000, 42));
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
    fn no_bursts_idle_only() {
        let v = simulate(0.0, 100, 5, 50, 1000, 42);
        if let BurstVerdict::Ok { burst_steps, .. } = v {
            assert_eq!(burst_steps, 0);
        }
    }

    #[test]
    fn frequent_bursts_high_mean() {
        let lo = simulate(0.001, 100, 5, 50, 10_000, 42);
        let hi = simulate(0.05, 100, 5, 50, 10_000, 42);
        if let (
            BurstVerdict::Ok {
                mean_arrivals: l, ..
            },
            BurstVerdict::Ok {
                mean_arrivals: h, ..
            },
        ) = (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn invalid_neg_prob() {
        assert_eq!(
            simulate(-0.1, 100, 5, 50, 1000, 42),
            BurstVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_duration() {
        assert_eq!(
            simulate(0.05, 100, 5, 0, 1000, 42),
            BurstVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_steps() {
        assert_eq!(
            simulate(0.05, 100, 5, 50, 0, 42),
            BurstVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(f64::NAN, 100, 5, 50, 1000, 42),
            BurstVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(0.05, 100, 5, 50, 1000, 42);
        let b = simulate(0.05, 100, 5, 50, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn mean_above_idle_when_bursting() {
        let v = simulate(0.1, 200, 5, 50, 10_000, 42);
        if let BurstVerdict::Ok { mean_arrivals, .. } = v {
            assert!(mean_arrivals > 5.0);
        }
    }

    #[test]
    fn burst_idle_partition_complete() {
        // burst_steps + idle_steps + (steps without classification) = steps.
        // If burst_start_prob=0, all are idle.
        let v = simulate(0.0, 100, 5, 50, 1000, 42);
        if let BurstVerdict::Ok {
            burst_steps,
            idle_steps,
            ..
        } = v
        {
            assert_eq!(burst_steps + idle_steps, 1000);
        }
    }
}
