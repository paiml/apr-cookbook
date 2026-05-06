//! # Distillation Gradient-Accumulation Picker
//!
//! Pick grad-accum steps so the student's *effective* batch size
//! matches the teacher's. effective = micro_batch × accum_steps.
//! Returns the smallest accum_steps such that effective ≥ target.
//!
//! Demonstrates the **DIST.37** recipe for PMAT-158 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Smith et al. (2018) "Don't Decay the Learning Rate, Increase the Batch Size."
//!
//! Run with: cargo run --example distill_grad_accum_picker
//!
//! Added by PMAT-158 (catalog 1045→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AccumVerdict {
    Ok {
        accum_steps: u32,
        effective_batch: u32,
    },
    InvalidConfig,
}

pub fn pick(target_effective_batch: u32, student_micro_batch: u32) -> AccumVerdict {
    if target_effective_batch == 0 || student_micro_batch == 0 {
        return AccumVerdict::InvalidConfig;
    }
    let accum_steps = target_effective_batch.div_ceil(student_micro_batch);
    let effective_batch = accum_steps * student_micro_batch;
    AccumVerdict::Ok {
        accum_steps,
        effective_batch,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_grad_accum_picker")?;

    println!("typical: {:?}", pick(512, 16));
    println!("aligned: {:?}", pick(256, 64));
    println!("tiny micro: {:?}", pick(1024, 1));
    println!("invalid: {:?}", pick(0, 16));
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
    fn aligned_returns_exact() {
        let v = pick(256, 64);
        if let AccumVerdict::Ok {
            accum_steps,
            effective_batch,
        } = v
        {
            assert_eq!(accum_steps, 4);
            assert_eq!(effective_batch, 256);
        }
    }

    #[test]
    fn unaligned_rounds_up() {
        let v = pick(512, 16);
        if let AccumVerdict::Ok { accum_steps, .. } = v {
            assert_eq!(accum_steps, 32);
        }
    }

    #[test]
    fn target_below_micro_one_step() {
        let v = pick(8, 16);
        if let AccumVerdict::Ok { accum_steps, .. } = v {
            assert_eq!(accum_steps, 1);
        }
    }

    #[test]
    fn zero_target_invalid() {
        assert_eq!(pick(0, 16), AccumVerdict::InvalidConfig);
    }

    #[test]
    fn zero_micro_invalid() {
        assert_eq!(pick(256, 0), AccumVerdict::InvalidConfig);
    }

    #[test]
    fn micro_one_steps_equal_target() {
        let v = pick(1024, 1);
        if let AccumVerdict::Ok { accum_steps, .. } = v {
            assert_eq!(accum_steps, 1024);
        }
    }

    #[test]
    fn effective_at_least_target() {
        for target in [100, 256, 333, 512, 1024] {
            for micro in [1, 16, 32, 64] {
                let v = pick(target, micro);
                if let AccumVerdict::Ok {
                    effective_batch, ..
                } = v
                {
                    assert!(effective_batch >= target);
                }
            }
        }
    }

    #[test]
    fn just_above_target_rounds_up() {
        let v = pick(257, 64);
        if let AccumVerdict::Ok { accum_steps, .. } = v {
            assert_eq!(accum_steps, 5);
        }
    }

    #[test]
    fn target_equal_micro() {
        let v = pick(64, 64);
        if let AccumVerdict::Ok { accum_steps, .. } = v {
            assert_eq!(accum_steps, 1);
        }
    }

    #[test]
    fn deterministic() {
        let a = pick(512, 16);
        let b = pick(512, 16);
        assert_eq!(a, b);
    }
}
