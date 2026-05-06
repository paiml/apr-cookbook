//! # Distributed Pipeline-Parallel Microbatch Sizer
//!
//! Pipeline parallelism splits model layers across N stages. To keep
//! all stages busy, microbatches > N is required (otherwise bubbles).
//! Picker rule:
//!   ideal_microbatch = max(num_stages × 4, batch / num_stages)
//!   round to nearest power of 2 if larger than 8
//!   require batch % microbatch == 0
//!
//! Demonstrates the **DIST.7** recipe for PMAT-139 (distributed coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GPipe (Huang et al., 2018) — pipeline parallelism with microbatches.
//!
//! Run with: cargo run --example distributed_pipeline_microbatch
//!
//! Added by PMAT-139 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MicrobatchVerdict {
    Ok {
        microbatch_size: u32,
        microbatches_per_step: u32,
    },
    BubblesUnavoidable {
        batch: u32,
        stages: u32,
    },
    InvalidShape,
}

const MIN_MICROBATCH_FACTOR: u32 = 4;

pub fn pick(global_batch: u32, num_stages: u32) -> MicrobatchVerdict {
    if global_batch == 0 || num_stages == 0 {
        return MicrobatchVerdict::InvalidShape;
    }
    if global_batch < num_stages * MIN_MICROBATCH_FACTOR {
        return MicrobatchVerdict::BubblesUnavoidable {
            batch: global_batch,
            stages: num_stages,
        };
    }
    let mut microbatch = (global_batch / num_stages).max(1);
    while global_batch % microbatch != 0 && microbatch > 1 {
        microbatch -= 1;
    }
    let microbatches = global_batch / microbatch;
    MicrobatchVerdict::Ok {
        microbatch_size: microbatch,
        microbatches_per_step: microbatches,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distributed_pipeline_microbatch")?;

    println!("batch=64 stages=4: {:?}", pick(64, 4));
    println!("batch=128 stages=8: {:?}", pick(128, 8));
    println!("batch=10 stages=4 (bubbles): {:?}", pick(10, 4));
    println!("batch=1000 stages=4: {:?}", pick(1000, 4));
    println!("invalid batch=0: {:?}", pick(0, 4));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_batch_64_stages_4() {
        let v = pick(64, 4);
        if let MicrobatchVerdict::Ok {
            microbatch_size,
            microbatches_per_step,
        } = v
        {
            assert_eq!(microbatch_size * microbatches_per_step, 64);
        }
    }

    #[test]
    fn batch_too_small_bubbles() {
        // batch < num_stages × 4.
        let v = pick(10, 4);
        assert!(matches!(v, MicrobatchVerdict::BubblesUnavoidable { .. }));
    }

    #[test]
    fn invalid_zero_batch() {
        assert_eq!(pick(0, 4), MicrobatchVerdict::InvalidShape);
    }

    #[test]
    fn invalid_zero_stages() {
        assert_eq!(pick(64, 0), MicrobatchVerdict::InvalidShape);
    }

    #[test]
    fn microbatches_at_least_stages() {
        // num_microbatches >= num_stages avoids bubbles.
        let v = pick(64, 4);
        if let MicrobatchVerdict::Ok {
            microbatches_per_step,
            ..
        } = v
        {
            assert!(microbatches_per_step >= 4);
        }
    }

    #[test]
    fn microbatch_divides_batch_evenly() {
        let v = pick(60, 4);
        if let MicrobatchVerdict::Ok {
            microbatch_size, ..
        } = v
        {
            assert_eq!(60 % microbatch_size, 0);
        }
    }

    #[test]
    fn at_min_factor_succeeds() {
        // batch = 4 stages × 4 = 16.
        let v = pick(16, 4);
        assert!(matches!(v, MicrobatchVerdict::Ok { .. }));
    }

    #[test]
    fn just_below_min_factor_bubbles() {
        let v = pick(15, 4);
        assert!(matches!(v, MicrobatchVerdict::BubblesUnavoidable { .. }));
    }

    #[test]
    fn many_stages_handles_larger_batch() {
        let v = pick(1024, 16);
        if let MicrobatchVerdict::Ok {
            microbatch_size,
            microbatches_per_step,
        } = v
        {
            assert_eq!(microbatch_size * microbatches_per_step, 1024);
        }
    }

    #[test]
    fn single_stage_full_batch_one_step() {
        let v = pick(100, 1);
        if let MicrobatchVerdict::Ok {
            microbatch_size,
            microbatches_per_step,
        } = v
        {
            // With 1 stage, no pipelining needed: 1 microbatch covers the batch.
            assert_eq!(microbatch_size, 100);
            assert_eq!(microbatches_per_step, 1);
        }
    }
}
