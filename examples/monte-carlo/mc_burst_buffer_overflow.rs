//! # Monte-Carlo Burst-Buffer Overflow Probability
//!
//! Sim P(buffer overflow) for a fixed-size burst buffer with mean
//! arrival rate λ and Gaussian arrival sizes. Returns observed
//! overflow rate.
//!
//! Demonstrates the **MC.58** recipe for PMAT-177 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: queueing-buffer overflow probability (Kleinrock).
//!
//! Run with: cargo run --example mc_burst_buffer_overflow
//!
//! Added by PMAT-177 (catalog 1216→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum OverflowVerdict {
    Ok {
        overflow_rate: f64,
        max_occupancy: u32,
    },
    InvalidConfig,
}

pub fn simulate(
    capacity: u32,
    arrival_mean: f64,
    arrival_jitter: f64,
    drain_per_step: u32,
    steps: u32,
    seed: u64,
) -> OverflowVerdict {
    if capacity == 0
        || !arrival_mean.is_finite()
        || arrival_mean < 0.0
        || !arrival_jitter.is_finite()
        || arrival_jitter < 0.0
        || steps == 0
    {
        return OverflowVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut occupancy: u32 = 0;
    let mut overflows: u32 = 0;
    let mut max_occupancy: u32 = 0;
    for _ in 0..steps {
        occupancy = occupancy.saturating_sub(drain_per_step);
        let jitter = (unit(&mut rng_state) - 0.5) * 2.0 * arrival_jitter;
        let arrivals = (arrival_mean + jitter).max(0.0) as u32;
        let proposed = occupancy.saturating_add(arrivals);
        if proposed > capacity {
            overflows += 1;
            occupancy = capacity;
        } else {
            occupancy = proposed;
        }
        max_occupancy = max_occupancy.max(occupancy);
    }
    let overflow_rate = f64::from(overflows) / f64::from(steps);
    OverflowVerdict::Ok {
        overflow_rate,
        max_occupancy,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_burst_buffer_overflow")?;

    println!("stable: {:?}", simulate(100, 5.0, 1.0, 10, 1000, 42));
    println!("overflow: {:?}", simulate(50, 10.0, 5.0, 5, 1000, 42));
    println!("invalid: {:?}", simulate(0, 5.0, 1.0, 10, 1000, 42));
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
    fn drain_above_arrival_no_overflow() {
        let v = simulate(100, 1.0, 0.1, 100, 1000, 42);
        if let OverflowVerdict::Ok { overflow_rate, .. } = v {
            assert!(overflow_rate < 0.01);
        }
    }

    #[test]
    fn arrival_above_drain_overflows() {
        let v = simulate(50, 20.0, 5.0, 1, 1000, 42);
        if let OverflowVerdict::Ok { overflow_rate, .. } = v {
            assert!(overflow_rate > 0.1);
        }
    }

    #[test]
    fn invalid_zero_capacity() {
        assert_eq!(
            simulate(0, 5.0, 1.0, 10, 1000, 42),
            OverflowVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_neg_arrival() {
        assert_eq!(
            simulate(100, -1.0, 1.0, 10, 1000, 42),
            OverflowVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_steps() {
        assert_eq!(
            simulate(100, 5.0, 1.0, 10, 0, 42),
            OverflowVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(100, f64::NAN, 1.0, 10, 1000, 42),
            OverflowVerdict::InvalidConfig
        );
    }

    #[test]
    fn rate_in_unit_range() {
        let v = simulate(100, 5.0, 1.0, 10, 1000, 42);
        if let OverflowVerdict::Ok { overflow_rate, .. } = v {
            assert!((0.0..=1.0).contains(&overflow_rate));
        }
    }

    #[test]
    fn max_bounded_by_capacity() {
        let v = simulate(50, 100.0, 1.0, 0, 100, 42);
        if let OverflowVerdict::Ok { max_occupancy, .. } = v {
            assert!(max_occupancy <= 50);
        }
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 5.0, 1.0, 10, 1000, 42);
        let b = simulate(100, 5.0, 1.0, 10, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn zero_arrival_no_overflow() {
        let v = simulate(100, 0.0, 0.0, 10, 1000, 42);
        if let OverflowVerdict::Ok { overflow_rate, .. } = v {
            assert!((overflow_rate - 0.0).abs() < 1e-9);
        }
    }
}
