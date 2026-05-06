//! # Distributed Failure Recovery Replay Window
//!
//! When a worker dies mid-step, the rest of the cluster restarts from
//! the most recent checkpoint. Replay window = (current_step −
//! checkpoint_step) × num_workers — wasted compute. Tradeoff:
//! frequent checkpoints reduce loss but cost I/O. This recipe builds
//! the calc + an idempotency check (replayed steps must be deterministic).
//!
//! Demonstrates the **DIST.4** recipe for PMAT-124 (distributed coverage —
//! closing F-invariant gap from 1 → 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Elnozahy et al. (2002). A survey of rollback-recovery protocols in message-passing systems.
//!
//! Run with: cargo run --example distributed_failure_recovery
//!
//! Added by PMAT-124 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ReplayVerdict {
    Ok {
        wasted_step_seconds: u64,
        wasted_compute_seconds: u64,
    },
    NoCheckpointAhead,
    InvalidWorkerCount,
}

pub fn calc_replay_cost(
    current_step: u32,
    last_checkpoint_step: u32,
    seconds_per_step: u32,
    num_workers: u32,
) -> ReplayVerdict {
    if num_workers == 0 {
        return ReplayVerdict::InvalidWorkerCount;
    }
    if last_checkpoint_step > current_step {
        return ReplayVerdict::NoCheckpointAhead;
    }
    let steps_to_replay = current_step - last_checkpoint_step;
    let wasted_step_seconds = u64::from(steps_to_replay) * u64::from(seconds_per_step);
    let wasted_compute_seconds = wasted_step_seconds * u64::from(num_workers);
    ReplayVerdict::Ok {
        wasted_step_seconds,
        wasted_compute_seconds,
    }
}

#[derive(Debug, PartialEq)]
pub enum IdempotencyVerdict {
    Deterministic,
    NonDeterministic { differing_steps: Vec<u32> },
    EmptyComparison,
}

pub fn check_idempotency(replay_a: &[(u32, u64)], replay_b: &[(u32, u64)]) -> IdempotencyVerdict {
    if replay_a.is_empty() && replay_b.is_empty() {
        return IdempotencyVerdict::EmptyComparison;
    }
    if replay_a.len() != replay_b.len() {
        let max_step = replay_a
            .iter()
            .chain(replay_b.iter())
            .map(|(s, _)| *s)
            .max()
            .unwrap_or(0);
        return IdempotencyVerdict::NonDeterministic {
            differing_steps: vec![max_step],
        };
    }
    let mut differing = Vec::new();
    for ((sa, ha), (sb, hb)) in replay_a.iter().zip(replay_b.iter()) {
        if sa != sb || ha != hb {
            differing.push(*sa);
        }
    }
    if differing.is_empty() {
        IdempotencyVerdict::Deterministic
    } else {
        IdempotencyVerdict::NonDeterministic {
            differing_steps: differing,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distributed_failure_recovery")?;

    let cases = [
        (1000u32, 950u32, 30u32, 8u32),
        (1000, 100, 30, 8),
        (50, 100, 30, 8),
        (1000, 950, 30, 0),
    ];
    for (cur, ckpt, secs, workers) in cases {
        println!(
            "step={cur} ckpt={ckpt} sec/step={secs} W={workers}  →  {:?}",
            calc_replay_cost(cur, ckpt, secs, workers)
        );
    }

    let a = [(0u32, 100u64), (1, 200), (2, 300)];
    let b = [(0u32, 100u64), (1, 200), (2, 300)];
    println!("identical: {:?}", check_idempotency(&a, &b));

    let c = [(0u32, 100u64), (1, 200), (2, 999)];
    println!("divergent: {:?}", check_idempotency(&a, &c));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calc_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_replay_cost() {
        // 50 steps × 30 sec/step × 8 workers = 12,000 worker-sec.
        let v = calc_replay_cost(1000, 950, 30, 8);
        assert!(matches!(
            v,
            ReplayVerdict::Ok {
                wasted_step_seconds: 1500,
                wasted_compute_seconds: 12000,
            }
        ));
    }

    #[test]
    fn checkpoint_ahead_rejected() {
        assert_eq!(
            calc_replay_cost(50, 100, 30, 8),
            ReplayVerdict::NoCheckpointAhead
        );
    }

    #[test]
    fn zero_workers_invalid() {
        assert_eq!(
            calc_replay_cost(1000, 950, 30, 0),
            ReplayVerdict::InvalidWorkerCount
        );
    }

    #[test]
    fn zero_distance_no_waste() {
        // Failed exactly at checkpoint → no replay cost.
        let v = calc_replay_cost(1000, 1000, 30, 8);
        assert!(matches!(
            v,
            ReplayVerdict::Ok {
                wasted_step_seconds: 0,
                ..
            }
        ));
    }

    #[test]
    fn identical_replays_deterministic() {
        let a = [(0u32, 100u64), (1, 200)];
        let b = [(0u32, 100u64), (1, 200)];
        assert_eq!(check_idempotency(&a, &b), IdempotencyVerdict::Deterministic);
    }

    #[test]
    fn divergent_hash_flagged() {
        let a = [(0u32, 100u64), (1, 200)];
        let b = [(0u32, 100u64), (1, 999)];
        let v = check_idempotency(&a, &b);
        assert!(matches!(v, IdempotencyVerdict::NonDeterministic { .. }));
    }

    #[test]
    fn length_mismatch_non_deterministic() {
        let a = [(0u32, 100u64), (1, 200)];
        let b = [(0u32, 100u64)];
        let v = check_idempotency(&a, &b);
        assert!(matches!(v, IdempotencyVerdict::NonDeterministic { .. }));
    }

    #[test]
    fn both_empty_treated_as_empty() {
        assert_eq!(
            check_idempotency(&[], &[]),
            IdempotencyVerdict::EmptyComparison
        );
    }

    #[test]
    fn compute_seconds_scales_with_workers() {
        let one = calc_replay_cost(1000, 950, 30, 1);
        let eight = calc_replay_cost(1000, 950, 30, 8);
        if let (
            ReplayVerdict::Ok {
                wasted_compute_seconds: a,
                ..
            },
            ReplayVerdict::Ok {
                wasted_compute_seconds: b,
                ..
            },
        ) = (one, eight)
        {
            assert_eq!(b, a * 8);
        }
    }
}
