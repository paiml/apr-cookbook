//! # Distillation Loss Weight Schedule
//!
//! Loss = α × KD_loss + (1 - α) × CE_loss.
//! Schedule: anneal α from high (early — trust teacher) to low (late
//! — trust ground truth).
//!
//! Linear: α(s) = α_start + (α_end - α_start) × (s/total).
//!
//! Demonstrates the **DIST.24** recipe for PMAT-152 (milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hinton et al. (2015) annealed knowledge distillation.
//!
//! Run with: cargo run --example distill_loss_weight_schedule
//!
//! Added by PMAT-152 (catalog crosses 1000 recipes).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AlphaVerdict {
    Ok {
        alpha: f64,
        kd_weight: f64,
        ce_weight: f64,
    },
    InvalidStep,
    InvalidAlpha,
}

pub fn pick(current_step: u32, total_steps: u32, alpha_start: f64, alpha_end: f64) -> AlphaVerdict {
    if total_steps == 0 {
        return AlphaVerdict::InvalidStep;
    }
    if !alpha_start.is_finite()
        || !alpha_end.is_finite()
        || !(0.0..=1.0).contains(&alpha_start)
        || !(0.0..=1.0).contains(&alpha_end)
    {
        return AlphaVerdict::InvalidAlpha;
    }
    let progress = f64::from(current_step.min(total_steps)) / f64::from(total_steps);
    let alpha = alpha_start + (alpha_end - alpha_start) * progress;
    AlphaVerdict::Ok {
        alpha,
        kd_weight: alpha,
        ce_weight: 1.0 - alpha,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_loss_weight_schedule")?;

    println!("step 0: {:?}", pick(0, 100, 0.9, 0.1));
    println!("step 50: {:?}", pick(50, 100, 0.9, 0.1));
    println!("step 100: {:?}", pick(100, 100, 0.9, 0.1));
    println!("invalid: {:?}", pick(0, 0, 0.9, 0.1));
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
    fn step_zero_alpha_start() {
        let v = pick(0, 100, 0.9, 0.1);
        if let AlphaVerdict::Ok { alpha, .. } = v {
            assert!((alpha - 0.9).abs() < 1e-9);
        }
    }

    #[test]
    fn end_step_alpha_end() {
        let v = pick(100, 100, 0.9, 0.1);
        if let AlphaVerdict::Ok { alpha, .. } = v {
            assert!((alpha - 0.1).abs() < 1e-9);
        }
    }

    #[test]
    fn mid_step_average() {
        let v = pick(50, 100, 0.9, 0.1);
        if let AlphaVerdict::Ok { alpha, .. } = v {
            assert!((alpha - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn weights_sum_to_one() {
        let v = pick(50, 100, 0.9, 0.1);
        if let AlphaVerdict::Ok {
            kd_weight,
            ce_weight,
            ..
        } = v
        {
            assert!((kd_weight + ce_weight - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn invalid_total_zero() {
        assert_eq!(pick(0, 0, 0.5, 0.5), AlphaVerdict::InvalidStep);
    }

    #[test]
    fn invalid_alpha_above_one() {
        assert_eq!(pick(0, 100, 1.5, 0.1), AlphaVerdict::InvalidAlpha);
    }

    #[test]
    fn invalid_alpha_below_zero() {
        assert_eq!(pick(0, 100, -0.1, 0.5), AlphaVerdict::InvalidAlpha);
    }

    #[test]
    fn nan_alpha_invalid() {
        assert_eq!(pick(0, 100, f64::NAN, 0.5), AlphaVerdict::InvalidAlpha);
    }

    #[test]
    fn step_above_total_clamped() {
        let v = pick(200, 100, 0.9, 0.1);
        if let AlphaVerdict::Ok { alpha, .. } = v {
            assert!((alpha - 0.1).abs() < 1e-9);
        }
    }

    #[test]
    fn anneal_high_to_low_decreasing() {
        let v0 = pick(0, 100, 0.9, 0.1);
        let v100 = pick(100, 100, 0.9, 0.1);
        if let (AlphaVerdict::Ok { alpha: a0, .. }, AlphaVerdict::Ok { alpha: a100, .. }) =
            (v0, v100)
        {
            assert!(a0 > a100);
        }
    }

    #[test]
    fn constant_schedule_when_endpoints_equal() {
        let v0 = pick(0, 100, 0.5, 0.5);
        let v50 = pick(50, 100, 0.5, 0.5);
        if let (AlphaVerdict::Ok { alpha: a0, .. }, AlphaVerdict::Ok { alpha: a50, .. }) = (v0, v50)
        {
            assert!((a0 - a50).abs() < 1e-9);
        }
    }
}
