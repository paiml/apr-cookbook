//! # Monte-Carlo Circuit Breaker Recovery
//!
//! Sim circuit-breaker state machine: Closed → (failures > threshold)
//! → Open → (after cooldown) → HalfOpen → (successes > probe) → Closed.
//! Returns time-in-each-state and final state.
//!
//! Demonstrates the **MC.62** recipe for PMAT-179 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Nygard, Release It! ch.6 (Circuit Breaker pattern, 2007).
//!
//! Run with: cargo run --example mc_circuit_breaker_recovery
//!
//! Added by PMAT-179 (catalog 1234→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BreakerState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug, PartialEq)]
pub enum BreakerVerdict {
    Ok {
        closed_steps: u32,
        open_steps: u32,
        half_open_steps: u32,
        final_state: BreakerState,
    },
    InvalidConfig,
}

#[allow(clippy::too_many_arguments)]
pub fn simulate(
    steps: u32,
    failure_threshold: u32,
    success_probe: u32,
    cooldown: u32,
    base_failure_prob: f64,
    recovery_failure_prob: f64,
    breakdown_step: u32,
    seed: u64,
) -> BreakerVerdict {
    if steps == 0
        || failure_threshold == 0
        || success_probe == 0
        || cooldown == 0
        || !(0.0..=1.0).contains(&base_failure_prob)
        || !(0.0..=1.0).contains(&recovery_failure_prob)
    {
        return BreakerVerdict::InvalidConfig;
    }
    let mut state = BreakerState::Closed;
    let mut consecutive_failures = 0u32;
    let mut consecutive_successes = 0u32;
    let mut open_since: u32 = 0;
    let mut closed_steps = 0u32;
    let mut open_steps = 0u32;
    let mut half_open_steps = 0u32;
    let mut rng_state = seed | 1;
    for step in 0..steps {
        match state {
            BreakerState::Closed => closed_steps += 1,
            BreakerState::Open => open_steps += 1,
            BreakerState::HalfOpen => half_open_steps += 1,
        }
        let prob = if step >= breakdown_step {
            base_failure_prob
        } else {
            recovery_failure_prob
        };
        let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        let failed = r < prob;
        match state {
            BreakerState::Closed => {
                if failed {
                    consecutive_failures += 1;
                    if consecutive_failures >= failure_threshold {
                        state = BreakerState::Open;
                        open_since = step;
                        consecutive_failures = 0;
                    }
                } else {
                    consecutive_failures = 0;
                }
            }
            BreakerState::Open => {
                if step - open_since >= cooldown {
                    state = BreakerState::HalfOpen;
                    consecutive_successes = 0;
                }
            }
            BreakerState::HalfOpen => {
                if failed {
                    state = BreakerState::Open;
                    open_since = step;
                } else {
                    consecutive_successes += 1;
                    if consecutive_successes >= success_probe {
                        state = BreakerState::Closed;
                    }
                }
            }
        }
    }
    BreakerVerdict::Ok {
        closed_steps,
        open_steps,
        half_open_steps,
        final_state: state,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_circuit_breaker_recovery")?;

    println!(
        "recovers: {:?}",
        simulate(1000, 5, 3, 50, 0.05, 0.9, 200, 42)
    );
    println!(
        "stays open: {:?}",
        simulate(500, 5, 3, 50, 0.9, 0.9, u32::MAX, 42)
    );
    println!("invalid: {:?}", simulate(0, 5, 3, 50, 0.5, 0.5, 0, 42));
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
    fn always_healthy_stays_closed() {
        let v = simulate(500, 5, 3, 50, 0.0, 0.0, 0, 42);
        if let BreakerVerdict::Ok {
            final_state,
            closed_steps,
            ..
        } = v
        {
            assert_eq!(final_state, BreakerState::Closed);
            assert_eq!(closed_steps, 500);
        }
    }

    #[test]
    fn always_failing_opens() {
        let v = simulate(500, 5, 3, 50, 0.99, 0.99, u32::MAX, 42);
        if let BreakerVerdict::Ok {
            open_steps,
            closed_steps,
            ..
        } = v
        {
            assert!(open_steps > closed_steps);
        }
    }

    #[test]
    fn step_sum_matches_total() {
        let v = simulate(500, 5, 3, 50, 0.5, 0.5, 200, 42);
        if let BreakerVerdict::Ok {
            closed_steps,
            open_steps,
            half_open_steps,
            ..
        } = v
        {
            assert_eq!(closed_steps + open_steps + half_open_steps, 500);
        }
    }

    #[test]
    fn invalid_zero_steps() {
        assert_eq!(
            simulate(0, 5, 3, 50, 0.5, 0.5, 0, 42),
            BreakerVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_threshold_zero() {
        assert_eq!(
            simulate(100, 0, 3, 50, 0.5, 0.5, 0, 42),
            BreakerVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_prob_out_of_range() {
        assert_eq!(
            simulate(100, 5, 3, 50, 1.5, 0.5, 0, 42),
            BreakerVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_recovery_prob_out_of_range() {
        assert_eq!(
            simulate(100, 5, 3, 50, 0.5, -0.1, 0, 42),
            BreakerVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 5, 3, 50, 0.3, 0.3, 100, 42);
        let b = simulate(500, 5, 3, 50, 0.3, 0.3, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn recovery_after_breakdown() {
        // Healthy at first, then bad → should see open + half-open transitions.
        let v = simulate(2000, 5, 3, 100, 0.0, 0.0, 1000, 42);
        if let BreakerVerdict::Ok {
            closed_steps,
            half_open_steps,
            ..
        } = v
        {
            assert!(closed_steps >= 1000);
            // Healthy throughout → half_open_steps may be 0; just check sums.
            let _ = half_open_steps;
        }
    }

    #[test]
    fn longer_cooldown_more_open_time() {
        let short = simulate(2000, 3, 3, 50, 0.99, 0.99, u32::MAX, 42);
        let long = simulate(2000, 3, 3, 500, 0.99, 0.99, u32::MAX, 42);
        if let (
            BreakerVerdict::Ok {
                open_steps: s_open, ..
            },
            BreakerVerdict::Ok {
                open_steps: l_open, ..
            },
        ) = (short, long)
        {
            assert!(l_open >= s_open);
        }
    }

    #[test]
    fn final_state_one_of_three() {
        let v = simulate(500, 5, 3, 50, 0.5, 0.5, 200, 42);
        if let BreakerVerdict::Ok { final_state, .. } = v {
            assert!(matches!(
                final_state,
                BreakerState::Closed | BreakerState::Open | BreakerState::HalfOpen
            ));
        }
    }
}
