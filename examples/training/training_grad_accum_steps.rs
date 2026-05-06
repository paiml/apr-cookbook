//! # Training Gradient-Accumulation Steps Picker
//!
//! Effective batch = micro_batch × accum_steps × num_gpus.
//! When desired_effective_batch > micro_batch × num_gpus, accumulate
//! gradients over multiple micro-steps before optimizer step.
//!
//! Picker: returns accum_steps + actual_effective_batch + accuracy_warning
//! when accum > 32 (approx error grows).
//!
//! Demonstrates the **TRAIN.13** recipe for PMAT-144 (training round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: PyTorch Lightning gradient-accumulation guide.
//!
//! Run with: cargo run --example training_grad_accum_steps
//!
//! Added by PMAT-144 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const ACCURACY_WARNING_THRESHOLD: u32 = 32;

#[derive(Debug, PartialEq)]
pub enum AccumVerdict {
    Ok {
        accum_steps: u32,
        actual_effective_batch: u32,
        accuracy_warning: bool,
    },
    InvalidShape,
    UnreachableTarget {
        per_device_max: u32,
    },
}

pub fn pick(
    desired_effective_batch: u32,
    micro_batch_per_device: u32,
    num_devices: u32,
) -> AccumVerdict {
    if desired_effective_batch == 0 || micro_batch_per_device == 0 || num_devices == 0 {
        return AccumVerdict::InvalidShape;
    }
    let per_device = micro_batch_per_device * num_devices;
    if desired_effective_batch < per_device {
        return AccumVerdict::UnreachableTarget {
            per_device_max: per_device,
        };
    }
    let accum_steps = desired_effective_batch.div_ceil(per_device);
    let actual_effective_batch = accum_steps * per_device;
    let accuracy_warning = accum_steps > ACCURACY_WARNING_THRESHOLD;
    AccumVerdict::Ok {
        accum_steps,
        actual_effective_batch,
        accuracy_warning,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("training_grad_accum_steps")?;

    println!("4096 / 32 / 8: {:?}", pick(4096, 32, 8));
    println!("256 / 32 / 8: {:?}", pick(256, 32, 8));
    println!("128 / 32 / 8 unreachable: {:?}", pick(128, 32, 8));
    println!("huge accum 65536/2/1 (warn): {:?}", pick(65536, 2, 1));
    println!("invalid: {:?}", pick(0, 32, 8));
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
    fn typical_8_gpu_training() {
        // 4096 / (32 × 8) = 16 accum steps; 16 × 256 = 4096.
        let v = pick(4096, 32, 8);
        if let AccumVerdict::Ok {
            accum_steps,
            actual_effective_batch,
            ..
        } = v
        {
            assert_eq!(accum_steps, 16);
            assert_eq!(actual_effective_batch, 4096);
        }
    }

    #[test]
    fn target_equals_per_device_one_step() {
        let v = pick(256, 32, 8);
        if let AccumVerdict::Ok { accum_steps, .. } = v {
            assert_eq!(accum_steps, 1);
        }
    }

    #[test]
    fn target_below_per_device_unreachable() {
        let v = pick(128, 32, 8);
        assert!(matches!(v, AccumVerdict::UnreachableTarget { .. }));
    }

    #[test]
    fn accum_warning_above_32_steps() {
        // 1024 × 1 / (16 × 1) = 64 accum steps → warn.
        let v = pick(1024, 16, 1);
        if let AccumVerdict::Ok {
            accuracy_warning, ..
        } = v
        {
            assert!(accuracy_warning);
        }
    }

    #[test]
    fn no_warning_below_32() {
        let v = pick(256, 32, 1);
        if let AccumVerdict::Ok {
            accuracy_warning, ..
        } = v
        {
            assert!(!accuracy_warning);
        }
    }

    #[test]
    fn invalid_zero_target_rejected() {
        assert_eq!(pick(0, 32, 8), AccumVerdict::InvalidShape);
    }

    #[test]
    fn invalid_zero_micro_batch_rejected() {
        assert_eq!(pick(1024, 0, 8), AccumVerdict::InvalidShape);
    }

    #[test]
    fn invalid_zero_devices_rejected() {
        assert_eq!(pick(1024, 32, 0), AccumVerdict::InvalidShape);
    }

    #[test]
    fn rounds_up_to_meet_or_exceed() {
        // 100 / 32 = 3.125 → ceil 4 accum steps × 32 = 128.
        let v = pick(100, 32, 1);
        if let AccumVerdict::Ok {
            accum_steps,
            actual_effective_batch,
            ..
        } = v
        {
            assert_eq!(accum_steps, 4);
            assert_eq!(actual_effective_batch, 128);
        }
    }

    #[test]
    fn boundary_at_32_no_warning() {
        // 32 accum steps exactly → no warning (must be > 32).
        let v = pick(32 * 32, 32, 1);
        if let AccumVerdict::Ok {
            accum_steps,
            accuracy_warning,
            ..
        } = v
        {
            assert_eq!(accum_steps, 32);
            assert!(!accuracy_warning);
        }
    }
}
