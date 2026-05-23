//! # Monte-Carlo Packet Loss Burst Pattern (Gilbert-Elliott)
//!
//! Sim packet loss using Gilbert-Elliott two-state Markov model:
//! `Good` (low loss) and `Bad` (high loss). Returns observed loss
//! rate and burst length distribution.
//!
//! Demonstrates the **MC.84** recipe for PMAT-187 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Gilbert, E.N., Bell Sys Tech J 39 (1960); Elliott, E.O.,
//!  Bell Sys Tech J 42 (1963).
//!
//! Run with: cargo run --example mc_packet_loss_burst_pattern
//!
//! Added by PMAT-187 (catalog 1306→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BurstVerdict {
    Ok {
        loss_rate: f64,
        max_burst_length: u32,
        burst_count: u32,
    },
    InvalidConfig,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Good,
    Bad,
}

#[allow(clippy::too_many_arguments)]
pub fn simulate(
    packets: u32,
    p_good_to_bad: f64,
    p_bad_to_good: f64,
    loss_in_good: f64,
    loss_in_bad: f64,
    seed: u64,
) -> BurstVerdict {
    if packets == 0
        || !(0.0..=1.0).contains(&p_good_to_bad)
        || !(0.0..=1.0).contains(&p_bad_to_good)
        || !(0.0..=1.0).contains(&loss_in_good)
        || !(0.0..=1.0).contains(&loss_in_bad)
    {
        return BurstVerdict::InvalidConfig;
    }
    let mut state = State::Good;
    let mut losses = 0u32;
    let mut current_burst: u32 = 0;
    let mut max_burst: u32 = 0;
    let mut burst_count: u32 = 0;
    let mut rng_state = seed | 1;
    for _ in 0..packets {
        // Loss?
        let p_loss = match state {
            State::Good => loss_in_good,
            State::Bad => loss_in_bad,
        };
        let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        if r < p_loss {
            losses += 1;
            current_burst += 1;
        } else if current_burst > 0 {
            if current_burst > max_burst {
                max_burst = current_burst;
            }
            burst_count += 1;
            current_burst = 0;
        }
        // State transition.
        let r2 = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        state = match state {
            State::Good if r2 < p_good_to_bad => State::Bad,
            State::Bad if r2 < p_bad_to_good => State::Good,
            other => other,
        };
    }
    if current_burst > 0 {
        if current_burst > max_burst {
            max_burst = current_burst;
        }
        burst_count += 1;
    }
    BurstVerdict::Ok {
        loss_rate: f64::from(losses) / f64::from(packets),
        max_burst_length: max_burst,
        burst_count,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_packet_loss_burst_pattern")?;

    println!(
        "low loss: {:?}",
        simulate(10_000, 0.01, 0.5, 0.001, 0.5, 42)
    );
    println!("high loss: {:?}", simulate(10_000, 0.1, 0.1, 0.01, 0.8, 42));
    println!("invalid: {:?}", simulate(0, 0.01, 0.5, 0.001, 0.5, 42));
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
    fn no_loss_in_good_state() {
        let v = simulate(1000, 0.0, 0.0, 0.0, 0.0, 42);
        if let BurstVerdict::Ok { loss_rate, .. } = v {
            assert_eq!(loss_rate, 0.0);
        }
    }

    #[test]
    fn always_loss_high_rate() {
        let v = simulate(1000, 0.0, 0.0, 1.0, 1.0, 42);
        if let BurstVerdict::Ok { loss_rate, .. } = v {
            assert_eq!(loss_rate, 1.0);
        }
    }

    #[test]
    fn invalid_zero_packets() {
        assert_eq!(
            simulate(0, 0.01, 0.5, 0.001, 0.5, 42),
            BurstVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_prob_out_of_range() {
        assert_eq!(
            simulate(100, 1.5, 0.5, 0.001, 0.5, 42),
            BurstVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 0.05, 0.5, 0.01, 0.5, 42);
        let b = simulate(500, 0.05, 0.5, 0.01, 0.5, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn loss_rate_in_unit_range() {
        let v = simulate(500, 0.05, 0.5, 0.01, 0.5, 42);
        if let BurstVerdict::Ok { loss_rate, .. } = v {
            assert!((0.0..=1.0).contains(&loss_rate));
        }
    }

    #[test]
    fn bursty_higher_max_burst() {
        let uniform = simulate(10_000, 0.0, 1.0, 0.05, 0.05, 42);
        let bursty = simulate(10_000, 0.05, 0.05, 0.0, 0.5, 42);
        if let (
            BurstVerdict::Ok {
                max_burst_length: u,
                ..
            },
            BurstVerdict::Ok {
                max_burst_length: b,
                ..
            },
        ) = (uniform, bursty)
        {
            assert!(b >= u);
        }
    }

    #[test]
    fn max_burst_le_packets() {
        let v = simulate(100, 0.5, 0.1, 0.5, 0.9, 42);
        if let BurstVerdict::Ok {
            max_burst_length, ..
        } = v
        {
            assert!(max_burst_length <= 100);
        }
    }

    #[test]
    fn always_good_zero_burst() {
        let v = simulate(1000, 0.0, 1.0, 0.0, 1.0, 42);
        if let BurstVerdict::Ok {
            max_burst_length, ..
        } = v
        {
            assert_eq!(max_burst_length, 0);
        }
    }

    #[test]
    fn burst_count_le_packets() {
        let v = simulate(100, 0.5, 0.5, 0.5, 0.5, 42);
        if let BurstVerdict::Ok { burst_count, .. } = v {
            assert!(burst_count <= 100);
        }
    }
}
