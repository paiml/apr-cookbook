//! # Distillation Continual Replay Buffer
//!
//! Continual learning needs replay buffer of older samples to avoid
//! catastrophic forgetting. Picker:
//!   buffer_size: 10% of new task data (scaled)
//!   sampling: balanced across past tasks
//!   replay_ratio: 0.5 (half new, half old)
//!
//! Demonstrates the **DIST.27** recipe for PMAT-154 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Rolnick et al. (2019). Experience Replay for Continual Learning.
//!
//! Run with: cargo run --example distill_continual_replay
//!
//! Added by PMAT-154 (catalog 1009→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ReplayVerdict {
    Ok {
        buffer_size: u32,
        per_task_samples: u32,
        replay_ratio: f64,
    },
    InvalidTaskCount,
    InvalidNewSamples,
}

pub fn plan(num_past_tasks: u32, new_task_samples: u32) -> ReplayVerdict {
    if num_past_tasks == 0 {
        return ReplayVerdict::InvalidTaskCount;
    }
    if new_task_samples == 0 {
        return ReplayVerdict::InvalidNewSamples;
    }
    let buffer_size = (new_task_samples / 10).max(100);
    let per_task_samples = buffer_size.div_ceil(num_past_tasks).max(1);
    let actual_buffer = per_task_samples * num_past_tasks;
    ReplayVerdict::Ok {
        buffer_size: actual_buffer,
        per_task_samples,
        replay_ratio: 0.5,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_continual_replay")?;

    println!("3 tasks, 10k new: {:?}", plan(3, 10_000));
    println!("10 tasks, 5k new: {:?}", plan(10, 5_000));
    println!("invalid: {:?}", plan(0, 1000));
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
    fn typical_buffer_sized() {
        let v = plan(3, 10_000);
        if let ReplayVerdict::Ok { buffer_size, .. } = v {
            // 10000 / 10 = 1000; 1000 / 3 = 333; final = 333 × 3 = 999.
            assert!(buffer_size >= 900);
            assert!(buffer_size <= 1100);
        }
    }

    #[test]
    fn per_task_balanced() {
        let v = plan(10, 5000);
        if let ReplayVerdict::Ok {
            per_task_samples,
            buffer_size,
            ..
        } = v
        {
            assert_eq!(per_task_samples * 10, buffer_size);
        }
    }

    #[test]
    fn invalid_zero_tasks() {
        assert_eq!(plan(0, 1000), ReplayVerdict::InvalidTaskCount);
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(plan(3, 0), ReplayVerdict::InvalidNewSamples);
    }

    #[test]
    fn min_buffer_100() {
        // Tiny new sample count → buffer floors at 100.
        let v = plan(3, 100);
        if let ReplayVerdict::Ok { buffer_size, .. } = v {
            assert!(buffer_size >= 100);
        }
    }

    #[test]
    fn replay_ratio_half() {
        let v = plan(3, 10_000);
        if let ReplayVerdict::Ok { replay_ratio, .. } = v {
            assert!((replay_ratio - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn many_tasks_smaller_per_task() {
        let v_few = plan(2, 10_000);
        let v_many = plan(50, 10_000);
        if let (
            ReplayVerdict::Ok {
                per_task_samples: f,
                ..
            },
            ReplayVerdict::Ok {
                per_task_samples: m,
                ..
            },
        ) = (v_few, v_many)
        {
            assert!(f > m);
        }
    }

    #[test]
    fn per_task_at_least_one() {
        // Many tasks, small buffer.
        let v = plan(1000, 10_000);
        if let ReplayVerdict::Ok {
            per_task_samples, ..
        } = v
        {
            assert!(per_task_samples >= 1);
        }
    }

    #[test]
    fn one_task_full_buffer() {
        let v = plan(1, 10_000);
        if let ReplayVerdict::Ok {
            per_task_samples,
            buffer_size,
            ..
        } = v
        {
            assert_eq!(per_task_samples, buffer_size);
        }
    }

    #[test]
    fn buffer_proportional_to_new_samples() {
        let v_small = plan(5, 1_000);
        let v_large = plan(5, 10_000);
        if let (
            ReplayVerdict::Ok { buffer_size: s, .. },
            ReplayVerdict::Ok { buffer_size: l, .. },
        ) = (v_small, v_large)
        {
            assert!(l >= s);
        }
    }
}
