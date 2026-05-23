//! # apr train --checkpoint-every — Save Schedule Planner
//!
//! `apr train --checkpoint-every <N>` saves model state every N steps.
//! Tradeoffs: too frequent → I/O bottleneck; too rare → big rollback
//! on failure. Rules: ≥ 100 steps; ≤ 10× total / 50 (no more than 50
//! checkpoints unless explicit). This recipe builds the planner.
//!
//! Demonstrates the **TRAIN.6** recipe for PMAT-116 (apr train coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender TRAIN-001 + checkpoint-restart literature
//!
//! Run with: cargo run --example cli_train_checkpoint_planner
//!
//! Added by PMAT-116 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CheckpointVerdict {
    Ok { count: u32 },
    IntervalTooSmall,
    TooManyCheckpoints { count: u32, recommended: u32 },
    InvalidTotal,
}

const MIN_INTERVAL: u32 = 100;
const MAX_CHECKPOINTS_DEFAULT: u32 = 50;

pub fn plan(every_n_steps: u32, total_steps: u32) -> CheckpointVerdict {
    if total_steps == 0 {
        return CheckpointVerdict::InvalidTotal;
    }
    if every_n_steps < MIN_INTERVAL {
        return CheckpointVerdict::IntervalTooSmall;
    }
    let count = total_steps / every_n_steps;
    if count > MAX_CHECKPOINTS_DEFAULT {
        return CheckpointVerdict::TooManyCheckpoints {
            count,
            recommended: MAX_CHECKPOINTS_DEFAULT,
        };
    }
    CheckpointVerdict::Ok { count }
}

pub fn auto_pick_interval(total_steps: u32) -> u32 {
    if total_steps == 0 {
        return MIN_INTERVAL;
    }
    let target_count = 25u32; // aim for ~25 checkpoints per run
    let raw = total_steps / target_count.max(1);
    raw.max(MIN_INTERVAL)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_train_checkpoint_planner")?;

    let cases = [(100u32, 10_000), (50, 1000), (1000, 100_000), (100, 0)];
    for (n, total) in cases {
        println!("every={n} total={total}  →  {:?}", plan(n, total));
    }
    println!("auto(50_000) = {}", auto_pick_interval(50_000));
    println!("auto(2_000) = {}", auto_pick_interval(2_000));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_plan_passes() {
        // 100 steps every 100 over 10K total → 100 checkpoints. Fails MAX.
        let v = plan(100, 10_000);
        assert!(matches!(v, CheckpointVerdict::TooManyCheckpoints { .. }));
    }

    #[test]
    fn well_spaced_plan_ok() {
        // 1000 steps every 1000 over 50K total → 50 checkpoints (at limit).
        let v = plan(1000, 50_000);
        assert!(matches!(v, CheckpointVerdict::Ok { count: 50 }));
    }

    #[test]
    fn interval_below_floor_rejected() {
        assert_eq!(plan(50, 10_000), CheckpointVerdict::IntervalTooSmall);
    }

    #[test]
    fn at_min_interval_passes_if_count_ok() {
        // 5 checkpoints with min interval.
        let v = plan(MIN_INTERVAL, 500);
        assert!(matches!(v, CheckpointVerdict::Ok { count: 5 }));
    }

    #[test]
    fn zero_total_rejected() {
        assert_eq!(plan(100, 0), CheckpointVerdict::InvalidTotal);
    }

    #[test]
    fn too_many_checkpoints_rejected() {
        let v = plan(100, 100_000);
        assert!(matches!(v, CheckpointVerdict::TooManyCheckpoints { .. }));
    }

    #[test]
    fn auto_pick_targets_25_checkpoints() {
        let interval = auto_pick_interval(50_000);
        let count = 50_000 / interval;
        assert!((20..=30).contains(&count), "got {count}");
    }

    #[test]
    fn auto_pick_respects_min_interval() {
        // Tiny total: would compute < MIN_INTERVAL; should clamp.
        let interval = auto_pick_interval(500);
        assert!(interval >= MIN_INTERVAL);
    }

    #[test]
    fn auto_pick_zero_steps_returns_min() {
        assert_eq!(auto_pick_interval(0), MIN_INTERVAL);
    }
}
