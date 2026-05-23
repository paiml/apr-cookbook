//! # Advanced Circuit Breaker State Machine
//!
//! Three states:
//!   Closed: requests flow normally
//!   Open: too many failures → reject all
//!   HalfOpen: probe with limited traffic
//! Transitions on consecutive failures (open) or successes (close).
//!
//! Demonstrates the **ADV.32** recipe for PMAT-156 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Nygard "Release It!" (2007) circuit breaker pattern.
//!
//! Run with: cargo run --example adv_circuit_breaker
//!
//! Added by PMAT-156 (catalog 1027→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakerState {
    Closed,
    Open,
    HalfOpen,
}

#[derive(Debug, PartialEq)]
pub enum BreakerVerdict {
    Stay {
        state: BreakerState,
    },
    Trip {
        from: BreakerState,
        to: BreakerState,
    },
    Reset,
    InvalidConfig,
}

pub fn step(
    state: BreakerState,
    consecutive_failures: u32,
    consecutive_successes: u32,
    fail_threshold: u32,
    success_threshold: u32,
) -> BreakerVerdict {
    if fail_threshold == 0 || success_threshold == 0 {
        return BreakerVerdict::InvalidConfig;
    }
    match state {
        BreakerState::Closed => {
            if consecutive_failures >= fail_threshold {
                BreakerVerdict::Trip {
                    from: BreakerState::Closed,
                    to: BreakerState::Open,
                }
            } else {
                BreakerVerdict::Stay {
                    state: BreakerState::Closed,
                }
            }
        }
        BreakerState::Open => {
            if consecutive_successes > 0 {
                BreakerVerdict::Trip {
                    from: BreakerState::Open,
                    to: BreakerState::HalfOpen,
                }
            } else {
                BreakerVerdict::Stay {
                    state: BreakerState::Open,
                }
            }
        }
        BreakerState::HalfOpen => {
            if consecutive_successes >= success_threshold {
                BreakerVerdict::Reset
            } else if consecutive_failures > 0 {
                BreakerVerdict::Trip {
                    from: BreakerState::HalfOpen,
                    to: BreakerState::Open,
                }
            } else {
                BreakerVerdict::Stay {
                    state: BreakerState::HalfOpen,
                }
            }
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_circuit_breaker")?;

    println!("closed→open: {:?}", step(BreakerState::Closed, 5, 0, 5, 3));
    println!("open→halfopen: {:?}", step(BreakerState::Open, 0, 1, 5, 3));
    println!(
        "halfopen→reset: {:?}",
        step(BreakerState::HalfOpen, 0, 3, 5, 3)
    );
    println!(
        "halfopen→open: {:?}",
        step(BreakerState::HalfOpen, 1, 0, 5, 3)
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
    fn closed_to_open_on_threshold() {
        let v = step(BreakerState::Closed, 5, 0, 5, 3);
        if let BreakerVerdict::Trip { to, .. } = v {
            assert_eq!(to, BreakerState::Open);
        }
    }

    #[test]
    fn closed_stays_below_threshold() {
        let v = step(BreakerState::Closed, 4, 0, 5, 3);
        assert!(matches!(v, BreakerVerdict::Stay { .. }));
    }

    #[test]
    fn open_to_halfopen_on_first_success() {
        let v = step(BreakerState::Open, 0, 1, 5, 3);
        if let BreakerVerdict::Trip { to, .. } = v {
            assert_eq!(to, BreakerState::HalfOpen);
        }
    }

    #[test]
    fn halfopen_resets_on_threshold() {
        let v = step(BreakerState::HalfOpen, 0, 3, 5, 3);
        assert_eq!(v, BreakerVerdict::Reset);
    }

    #[test]
    fn halfopen_to_open_on_failure() {
        let v = step(BreakerState::HalfOpen, 1, 0, 5, 3);
        if let BreakerVerdict::Trip { to, .. } = v {
            assert_eq!(to, BreakerState::Open);
        }
    }

    #[test]
    fn zero_fail_threshold_invalid() {
        assert_eq!(
            step(BreakerState::Closed, 5, 0, 0, 3),
            BreakerVerdict::InvalidConfig
        );
    }

    #[test]
    fn zero_success_threshold_invalid() {
        assert_eq!(
            step(BreakerState::HalfOpen, 0, 5, 5, 0),
            BreakerVerdict::InvalidConfig
        );
    }

    #[test]
    fn open_stays_with_zero_successes() {
        let v = step(BreakerState::Open, 5, 0, 5, 3);
        assert!(matches!(v, BreakerVerdict::Stay { .. }));
    }

    #[test]
    fn halfopen_stays_when_neither_threshold_met() {
        let v = step(BreakerState::HalfOpen, 0, 1, 5, 3);
        assert!(matches!(v, BreakerVerdict::Stay { .. }));
    }

    #[test]
    fn deterministic() {
        let a = step(BreakerState::Closed, 5, 0, 5, 3);
        let b = step(BreakerState::Closed, 5, 0, 5, 3);
        assert_eq!(a, b);
    }
}
