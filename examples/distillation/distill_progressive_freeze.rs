//! # Distillation Progressive Layer Freeze Schedule
//!
//! Freeze trained layers, train new ones. Schedule:
//!   step 0: train all
//!   step warmup: freeze layers 0..k
//!   step total/2: freeze layers 0..2k
//!   etc., gradually freezing bottom-up.
//!
//! Picker returns the set of trainable layer indices at given step.
//!
//! Demonstrates the **DIST.22** recipe for PMAT-152 (distillation milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Felbo et al. (2017). Progressive layer freezing in transfer learning.
//!
//! Run with: cargo run --example distill_progressive_freeze
//!
//! Added by PMAT-152 (catalog crosses 1000 recipes).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FreezeVerdict {
    Ok {
        frozen_below: u32,
        trainable_count: u32,
    },
    InvalidStep,
    InvalidLayerCount,
}

pub fn pick(current_step: u32, total_steps: u32, n_layers: u32) -> FreezeVerdict {
    if total_steps == 0 {
        return FreezeVerdict::InvalidStep;
    }
    if n_layers == 0 {
        return FreezeVerdict::InvalidLayerCount;
    }
    let progress = f64::from(current_step.min(total_steps)) / f64::from(total_steps);
    let frozen_below = (progress * f64::from(n_layers)).floor() as u32;
    let frozen_below = frozen_below.min(n_layers - 1);
    let trainable_count = n_layers - frozen_below;
    FreezeVerdict::Ok {
        frozen_below,
        trainable_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_progressive_freeze")?;

    println!("step 0/100: {:?}", pick(0, 100, 12));
    println!("step 25/100: {:?}", pick(25, 100, 12));
    println!("step 50/100: {:?}", pick(50, 100, 12));
    println!("step 100/100: {:?}", pick(100, 100, 12));
    println!("invalid: {:?}", pick(0, 0, 12));
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
    fn step_zero_no_frozen() {
        let v = pick(0, 100, 12);
        if let FreezeVerdict::Ok { frozen_below, .. } = v {
            assert_eq!(frozen_below, 0);
        }
    }

    #[test]
    fn end_step_max_frozen() {
        let v = pick(100, 100, 12);
        if let FreezeVerdict::Ok { frozen_below, .. } = v {
            // Capped at n_layers - 1 to keep at least one trainable.
            assert_eq!(frozen_below, 11);
        }
    }

    #[test]
    fn at_least_one_trainable() {
        for step in [0u32, 50, 100, 200] {
            if let FreezeVerdict::Ok {
                trainable_count, ..
            } = pick(step, 100, 12)
            {
                assert!(trainable_count >= 1);
            }
        }
    }

    #[test]
    fn invalid_zero_total() {
        assert_eq!(pick(0, 0, 12), FreezeVerdict::InvalidStep);
    }

    #[test]
    fn invalid_zero_layers() {
        assert_eq!(pick(0, 100, 0), FreezeVerdict::InvalidLayerCount);
    }

    #[test]
    fn freezing_progresses() {
        let v0 = pick(0, 100, 12);
        let v50 = pick(50, 100, 12);
        let v100 = pick(100, 100, 12);
        if let (
            FreezeVerdict::Ok {
                frozen_below: f0, ..
            },
            FreezeVerdict::Ok {
                frozen_below: f50, ..
            },
            FreezeVerdict::Ok {
                frozen_below: f100, ..
            },
        ) = (v0, v50, v100)
        {
            assert!(f0 <= f50);
            assert!(f50 <= f100);
        }
    }

    #[test]
    fn step_clamped_above_total() {
        let v = pick(200, 100, 12);
        assert!(matches!(
            v,
            FreezeVerdict::Ok {
                frozen_below: 11,
                ..
            }
        ));
    }

    #[test]
    fn small_model_handled() {
        let v = pick(50, 100, 1);
        if let FreezeVerdict::Ok {
            trainable_count, ..
        } = v
        {
            assert_eq!(trainable_count, 1);
        }
    }

    #[test]
    fn large_model_handled() {
        let v = pick(50, 100, 96);
        assert!(matches!(v, FreezeVerdict::Ok { .. }));
    }

    #[test]
    fn trainable_plus_frozen_equals_n_layers() {
        let v = pick(50, 100, 12);
        if let FreezeVerdict::Ok {
            frozen_below,
            trainable_count,
        } = v
        {
            assert_eq!(frozen_below + trainable_count, 12);
        }
    }
}
