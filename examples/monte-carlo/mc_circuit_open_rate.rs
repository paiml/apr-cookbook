//! # Monte-Carlo Circuit Breaker Open Rate
//!
//! Inject random failures with `failure_rate`. Circuit opens after
//! `consecutive_failures` in a row, stays open for `cooldown`, then
//! tries again. Returns observed open-fraction of total time.
//!
//! Demonstrates the **MC.45** recipe for PMAT-172 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hystrix-style breaker model.
//!
//! Run with: cargo run --example mc_circuit_open_rate
//!
//! Added by PMAT-172 (catalog 1171→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BreakerVerdict {
    Ok {
        open_steps: u32,
        open_fraction: f64,
        trip_count: u32,
    },
    InvalidConfig,
}

pub fn simulate(
    duration_steps: u32,
    failure_rate: f64,
    consecutive_threshold: u32,
    cooldown: u32,
    seed: u64,
) -> BreakerVerdict {
    if duration_steps == 0
        || consecutive_threshold == 0
        || cooldown == 0
        || !failure_rate.is_finite()
        || !(0.0..=1.0).contains(&failure_rate)
    {
        return BreakerVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut consecutive_failures: u32 = 0;
    let mut open_remaining: u32 = 0;
    let mut open_steps: u32 = 0;
    let mut trip_count: u32 = 0;
    for _ in 0..duration_steps {
        if open_remaining > 0 {
            open_remaining -= 1;
            open_steps += 1;
            continue;
        }
        let failed = unit(&mut rng_state) < failure_rate;
        if failed {
            consecutive_failures += 1;
            if consecutive_failures >= consecutive_threshold {
                open_remaining = cooldown;
                trip_count += 1;
                consecutive_failures = 0;
            }
        } else {
            consecutive_failures = 0;
        }
    }
    let open_fraction = f64::from(open_steps) / f64::from(duration_steps);
    BreakerVerdict::Ok {
        open_steps,
        open_fraction,
        trip_count,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_circuit_open_rate")?;

    println!("low fail: {:?}", simulate(10_000, 0.01, 5, 50, 42));
    println!("high fail: {:?}", simulate(10_000, 0.5, 5, 50, 42));
    println!("invalid: {:?}", simulate(0, 0.5, 5, 50, 42));
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
    fn low_fail_few_trips() {
        let v = simulate(10_000, 0.01, 5, 50, 42);
        if let BreakerVerdict::Ok { trip_count, .. } = v {
            assert!(trip_count < 10);
        }
    }

    #[test]
    fn high_fail_many_trips() {
        let v = simulate(10_000, 0.7, 5, 50, 42);
        if let BreakerVerdict::Ok { trip_count, .. } = v {
            assert!(trip_count > 10);
        }
    }

    #[test]
    fn zero_fail_never_trips() {
        let v = simulate(10_000, 0.0, 5, 50, 42);
        if let BreakerVerdict::Ok { trip_count, .. } = v {
            assert_eq!(trip_count, 0);
        }
    }

    #[test]
    fn invalid_zero_duration() {
        assert_eq!(simulate(0, 0.5, 5, 50, 42), BreakerVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_threshold() {
        assert_eq!(simulate(100, 0.5, 0, 50, 42), BreakerVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_cooldown() {
        assert_eq!(simulate(100, 0.5, 5, 0, 42), BreakerVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_neg_rate() {
        assert_eq!(
            simulate(100, -0.1, 5, 50, 42),
            BreakerVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(100, f64::NAN, 5, 50, 42),
            BreakerVerdict::InvalidConfig
        );
    }

    #[test]
    fn open_fraction_in_unit() {
        let v = simulate(10_000, 0.5, 5, 50, 42);
        if let BreakerVerdict::Ok { open_fraction, .. } = v {
            assert!((0.0..=1.0).contains(&open_fraction));
        }
    }

    #[test]
    fn deterministic() {
        let a = simulate(1000, 0.5, 5, 50, 42);
        let b = simulate(1000, 0.5, 5, 50, 42);
        assert_eq!(a, b);
    }
}
