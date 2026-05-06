//! # apr train watch — Crash + Hang Restart Policy
//!
//! `apr train watch <RUN_DIR>` monitors a training run and restarts on
//! either (a) process crash (non-zero exit) or (b) hang (no checkpoint
//! progress for ≥ N minutes). This recipe builds the restart decision
//! tree and asserts the contract: bounded retries (default 3), backoff
//! between retries (exp), explicit "give up" verdict surfaces the reason.
//!
//! Demonstrates the **TRAIN.14** recipe for PMAT-106 (apr train watch coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender TRAIN-WATCH-001 + exponential backoff convention
//!
//! Run with: cargo run --example cli_train_watch_restart_policy
//!
//! Added by PMAT-106 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RestartCause {
    Crash,
    Hang { idle_minutes: u32 },
    Healthy,
}

#[derive(Debug, PartialEq)]
pub enum RestartDecision {
    Continue, // healthy, no action
    RestartWithBackoff {
        wait_seconds: u32,
    },
    GiveUp {
        attempts_used: u32,
        reason: &'static str,
    },
}

const MAX_ATTEMPTS: u32 = 3;
const HANG_THRESHOLD_MIN: u32 = 30;
const BASE_BACKOFF_SEC: u32 = 30;

pub fn decide_restart(cause: RestartCause, attempts_used: u32) -> RestartDecision {
    if matches!(cause, RestartCause::Healthy) {
        return RestartDecision::Continue;
    }
    if attempts_used >= MAX_ATTEMPTS {
        return RestartDecision::GiveUp {
            attempts_used,
            reason: match cause {
                RestartCause::Crash => "max retries exceeded after repeated crashes",
                RestartCause::Hang { .. } => "max retries exceeded after repeated hangs",
                RestartCause::Healthy => "unreachable",
            },
        };
    }
    if let RestartCause::Hang { idle_minutes } = cause {
        if idle_minutes < HANG_THRESHOLD_MIN {
            return RestartDecision::Continue;
        }
    }
    let wait = BASE_BACKOFF_SEC * (1 << attempts_used);
    RestartDecision::RestartWithBackoff { wait_seconds: wait }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_train_watch_restart_policy")?;

    let cases = [
        ("healthy", RestartCause::Healthy, 0),
        ("crash 1st", RestartCause::Crash, 0),
        ("crash 2nd", RestartCause::Crash, 1),
        ("crash 3rd", RestartCause::Crash, 2),
        ("crash 4th (give up)", RestartCause::Crash, 3),
        (
            "hang 5min (continue)",
            RestartCause::Hang { idle_minutes: 5 },
            0,
        ),
        (
            "hang 30min (restart)",
            RestartCause::Hang { idle_minutes: 30 },
            0,
        ),
    ];
    for (label, cause, attempts) in cases {
        println!("{label:>22}  →  {:?}", decide_restart(cause, attempts));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn healthy_continues() {
        assert_eq!(
            decide_restart(RestartCause::Healthy, 0),
            RestartDecision::Continue
        );
    }

    #[test]
    fn first_crash_restarts_with_30s_backoff() {
        let d = decide_restart(RestartCause::Crash, 0);
        assert_eq!(d, RestartDecision::RestartWithBackoff { wait_seconds: 30 });
    }

    #[test]
    fn second_crash_doubles_backoff() {
        let d = decide_restart(RestartCause::Crash, 1);
        assert_eq!(d, RestartDecision::RestartWithBackoff { wait_seconds: 60 });
    }

    #[test]
    fn third_crash_doubles_again() {
        let d = decide_restart(RestartCause::Crash, 2);
        assert_eq!(d, RestartDecision::RestartWithBackoff { wait_seconds: 120 });
    }

    #[test]
    fn fourth_crash_gives_up() {
        let d = decide_restart(RestartCause::Crash, 3);
        assert!(matches!(d, RestartDecision::GiveUp { .. }));
    }

    #[test]
    fn hang_below_threshold_continues() {
        let d = decide_restart(RestartCause::Hang { idle_minutes: 5 }, 0);
        assert_eq!(d, RestartDecision::Continue);
    }

    #[test]
    fn hang_at_threshold_restarts() {
        let d = decide_restart(
            RestartCause::Hang {
                idle_minutes: HANG_THRESHOLD_MIN,
            },
            0,
        );
        assert!(matches!(d, RestartDecision::RestartWithBackoff { .. }));
    }

    #[test]
    fn give_up_reason_distinguishes_crash_from_hang() {
        let crash = decide_restart(RestartCause::Crash, 3);
        let hang = decide_restart(RestartCause::Hang { idle_minutes: 100 }, 3);
        if let (
            RestartDecision::GiveUp { reason: cr, .. },
            RestartDecision::GiveUp { reason: hr, .. },
        ) = (crash, hang)
        {
            assert_ne!(cr, hr);
        } else {
            panic!("expected both GiveUp");
        }
    }
}
