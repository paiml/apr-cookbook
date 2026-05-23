//! # API Circuit Breaker State Machine
//!
//! Circuit breaker has three states:
//!   Closed: normal traffic; track failure count over window
//!   Open: failures crossed threshold → reject all requests for cooldown_ms
//!   HalfOpen: cooldown elapsed → try one request; success → Closed, failure → Open
//!
//! This recipe builds the state-transition function.
//!
//! Demonstrates the **API.9** recipe for PMAT-138 (api coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Nygard (2007). Release It! § Stability Patterns.
//!
//! Run with: cargo run --example api_circuit_breaker_state
//!
//! Added by PMAT-138 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Success,
    Failure,
}

#[derive(Debug, PartialEq)]
pub enum TransitionVerdict {
    Stay(CircuitState),
    Transition {
        from: CircuitState,
        to: CircuitState,
    },
}

pub fn next_state(
    current: CircuitState,
    outcome: Outcome,
    failure_count_in_window: u32,
    failure_threshold: u32,
    elapsed_since_open_ms: u64,
    cooldown_ms: u64,
) -> TransitionVerdict {
    match current {
        CircuitState::Closed => {
            if outcome == Outcome::Failure && failure_count_in_window >= failure_threshold {
                TransitionVerdict::Transition {
                    from: CircuitState::Closed,
                    to: CircuitState::Open,
                }
            } else {
                TransitionVerdict::Stay(CircuitState::Closed)
            }
        }
        CircuitState::Open => {
            if elapsed_since_open_ms >= cooldown_ms {
                TransitionVerdict::Transition {
                    from: CircuitState::Open,
                    to: CircuitState::HalfOpen,
                }
            } else {
                TransitionVerdict::Stay(CircuitState::Open)
            }
        }
        CircuitState::HalfOpen => match outcome {
            Outcome::Success => TransitionVerdict::Transition {
                from: CircuitState::HalfOpen,
                to: CircuitState::Closed,
            },
            Outcome::Failure => TransitionVerdict::Transition {
                from: CircuitState::HalfOpen,
                to: CircuitState::Open,
            },
        },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("api_circuit_breaker_state")?;

    println!(
        "closed, threshold reached: {:?}",
        next_state(CircuitState::Closed, Outcome::Failure, 5, 5, 0, 30_000)
    );
    println!(
        "open, cooldown elapsed: {:?}",
        next_state(CircuitState::Open, Outcome::Success, 0, 5, 31_000, 30_000)
    );
    println!(
        "half-open success: {:?}",
        next_state(CircuitState::HalfOpen, Outcome::Success, 0, 5, 0, 30_000)
    );
    println!(
        "half-open failure: {:?}",
        next_state(CircuitState::HalfOpen, Outcome::Failure, 0, 5, 0, 30_000)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breaker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn closed_with_few_failures_stays() {
        let v = next_state(CircuitState::Closed, Outcome::Failure, 2, 5, 0, 30_000);
        assert_eq!(v, TransitionVerdict::Stay(CircuitState::Closed));
    }

    #[test]
    fn closed_at_threshold_opens() {
        let v = next_state(CircuitState::Closed, Outcome::Failure, 5, 5, 0, 30_000);
        assert_eq!(
            v,
            TransitionVerdict::Transition {
                from: CircuitState::Closed,
                to: CircuitState::Open
            }
        );
    }

    #[test]
    fn closed_with_success_stays() {
        let v = next_state(CircuitState::Closed, Outcome::Success, 4, 5, 0, 30_000);
        assert_eq!(v, TransitionVerdict::Stay(CircuitState::Closed));
    }

    #[test]
    fn open_in_cooldown_stays() {
        let v = next_state(CircuitState::Open, Outcome::Success, 0, 5, 5_000, 30_000);
        assert_eq!(v, TransitionVerdict::Stay(CircuitState::Open));
    }

    #[test]
    fn open_after_cooldown_to_half_open() {
        let v = next_state(CircuitState::Open, Outcome::Success, 0, 5, 31_000, 30_000);
        assert_eq!(
            v,
            TransitionVerdict::Transition {
                from: CircuitState::Open,
                to: CircuitState::HalfOpen
            }
        );
    }

    #[test]
    fn half_open_success_closes() {
        let v = next_state(CircuitState::HalfOpen, Outcome::Success, 0, 5, 0, 30_000);
        assert_eq!(
            v,
            TransitionVerdict::Transition {
                from: CircuitState::HalfOpen,
                to: CircuitState::Closed
            }
        );
    }

    #[test]
    fn half_open_failure_opens() {
        let v = next_state(CircuitState::HalfOpen, Outcome::Failure, 0, 5, 0, 30_000);
        assert_eq!(
            v,
            TransitionVerdict::Transition {
                from: CircuitState::HalfOpen,
                to: CircuitState::Open
            }
        );
    }

    #[test]
    fn cooldown_at_exact_threshold_transitions() {
        let v = next_state(CircuitState::Open, Outcome::Success, 0, 5, 30_000, 30_000);
        assert!(matches!(v, TransitionVerdict::Transition { .. }));
    }

    #[test]
    fn very_high_threshold_stays_closed() {
        let v = next_state(CircuitState::Closed, Outcome::Failure, 100, 1000, 0, 30_000);
        assert_eq!(v, TransitionVerdict::Stay(CircuitState::Closed));
    }

    #[test]
    fn open_threshold_above_count_stays() {
        // failure_count not consulted in Open state.
        let v = next_state(CircuitState::Open, Outcome::Success, 999, 5, 1_000, 30_000);
        assert_eq!(v, TransitionVerdict::Stay(CircuitState::Open));
    }
}
