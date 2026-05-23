//! # Monte-Carlo Streaming Buffer Underflow
//!
//! Sim streaming buffer underflow: producer tokens at `producer_rate`,
//! consumer drains at `consumer_rate`. Returns underflow rate
//! (steps where buffer < threshold) and average buffer level.
//!
//! Demonstrates the **MC.62** recipe for PMAT-178 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: producer-consumer queue stability (Kleinrock).
//!
//! Run with: cargo run --example mc_streaming_underflow
//!
//! Added by PMAT-178 (catalog 1225→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum UnderflowVerdict {
    Ok {
        underflow_rate: f64,
        mean_level: f64,
        ended_empty: bool,
    },
    InvalidConfig,
}

pub fn simulate(
    initial_buffer: u32,
    producer_rate: f64,
    consumer_rate: f64,
    underflow_threshold: u32,
    steps: u32,
    seed: u64,
) -> UnderflowVerdict {
    if !producer_rate.is_finite()
        || producer_rate < 0.0
        || !consumer_rate.is_finite()
        || consumer_rate < 0.0
        || steps == 0
    {
        return UnderflowVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut buffer = i64::from(initial_buffer);
    let mut underflow_steps = 0u32;
    let mut sum_level: i64 = 0;
    for _ in 0..steps {
        let producer_jitter = unit(&mut rng_state);
        let consumer_jitter = unit(&mut rng_state);
        let produced = (producer_rate * (0.5 + producer_jitter)) as i64;
        let consumed = (consumer_rate * (0.5 + consumer_jitter)) as i64;
        buffer = (buffer + produced - consumed).max(0);
        if buffer < i64::from(underflow_threshold) {
            underflow_steps += 1;
        }
        sum_level += buffer;
    }
    let underflow_rate = f64::from(underflow_steps) / f64::from(steps);
    let mean_level = sum_level as f64 / f64::from(steps);
    let ended_empty = buffer == 0;
    UnderflowVerdict::Ok {
        underflow_rate,
        mean_level,
        ended_empty,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_streaming_underflow")?;

    println!("balanced: {:?}", simulate(100, 10.0, 10.0, 5, 1000, 42));
    println!("consumer_fast: {:?}", simulate(100, 5.0, 20.0, 5, 1000, 42));
    println!("invalid: {:?}", simulate(0, -1.0, 10.0, 5, 1000, 42));
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
    fn balanced_low_underflow() {
        let v = simulate(100, 10.0, 10.0, 1, 10_000, 42);
        if let UnderflowVerdict::Ok { underflow_rate, .. } = v {
            assert!(underflow_rate < 0.5);
        }
    }

    #[test]
    fn fast_consumer_high_underflow() {
        let v = simulate(10, 5.0, 50.0, 5, 1000, 42);
        if let UnderflowVerdict::Ok { underflow_rate, .. } = v {
            assert!(underflow_rate > 0.5);
        }
    }

    #[test]
    fn invalid_neg_producer() {
        assert_eq!(
            simulate(0, -1.0, 10.0, 5, 1000, 42),
            UnderflowVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_steps() {
        assert_eq!(
            simulate(0, 10.0, 10.0, 5, 0, 42),
            UnderflowVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(0, f64::NAN, 10.0, 5, 1000, 42),
            UnderflowVerdict::InvalidConfig
        );
    }

    #[test]
    fn rate_in_unit_range() {
        let v = simulate(100, 10.0, 10.0, 5, 1000, 42);
        if let UnderflowVerdict::Ok { underflow_rate, .. } = v {
            assert!((0.0..=1.0).contains(&underflow_rate));
        }
    }

    #[test]
    fn deterministic() {
        let a = simulate(100, 10.0, 10.0, 5, 1000, 42);
        let b = simulate(100, 10.0, 10.0, 5, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn mean_level_non_negative() {
        let v = simulate(100, 10.0, 10.0, 5, 1000, 42);
        if let UnderflowVerdict::Ok { mean_level, .. } = v {
            assert!(mean_level >= 0.0);
        }
    }

    #[test]
    fn fast_producer_high_buffer() {
        let v = simulate(0, 50.0, 5.0, 5, 1000, 42);
        if let UnderflowVerdict::Ok { mean_level, .. } = v {
            assert!(mean_level > 100.0);
        }
    }

    #[test]
    fn zero_initial_starts_underflow() {
        let v = simulate(0, 1.0, 1.0, 5, 100, 42);
        if let UnderflowVerdict::Ok { underflow_rate, .. } = v {
            assert!(underflow_rate > 0.0);
        }
    }
}
