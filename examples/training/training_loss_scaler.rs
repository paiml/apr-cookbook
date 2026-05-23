//! # Training Mixed-Precision Loss Scaler
//!
//! Mixed-precision training uses fp16 forward/backward pass; gradients
//! often underflow at fp16 precision. Loss scaling: multiply loss by
//! S before backprop, divide gradients by S before optimizer step.
//!
//! Dynamic adjustment:
//!   gradient_inf_or_nan → halve scale, skip step
//!   N consecutive successes → double scale (try higher dynamic range)
//!
//! Demonstrates the **TRAIN.15** recipe for PMAT-144 (training round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NVIDIA Apex amp + PyTorch GradScaler.
//!
//! Run with: cargo run --example training_loss_scaler
//!
//! Added by PMAT-144 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const INITIAL_SCALE: f64 = 65536.0;
const GROWTH_INTERVAL: u32 = 2000;
const MIN_SCALE: f64 = 1.0;
const MAX_SCALE: f64 = 1_073_741_824.0;

#[derive(Debug, PartialEq)]
pub enum ScalerVerdict {
    OptimizerStepThenGrow { new_scale: f64 },
    OptimizerStepNoChange { scale: f64 },
    SkipStepHalveScale { new_scale: f64 },
    UnderflowScaleAtMinimum,
    InvalidScale,
}

pub fn step(current_scale: f64, consecutive_successes: u32, grad_overflow: bool) -> ScalerVerdict {
    if !current_scale.is_finite() || current_scale < MIN_SCALE {
        return ScalerVerdict::InvalidScale;
    }
    if grad_overflow {
        let new_scale = (current_scale / 2.0).max(MIN_SCALE);
        if (current_scale - MIN_SCALE).abs() < f64::EPSILON {
            return ScalerVerdict::UnderflowScaleAtMinimum;
        }
        return ScalerVerdict::SkipStepHalveScale { new_scale };
    }
    if consecutive_successes + 1 >= GROWTH_INTERVAL {
        let new_scale = (current_scale * 2.0).min(MAX_SCALE);
        return ScalerVerdict::OptimizerStepThenGrow { new_scale };
    }
    ScalerVerdict::OptimizerStepNoChange {
        scale: current_scale,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("training_loss_scaler")?;

    println!("normal step: {:?}", step(INITIAL_SCALE, 100, false));
    println!(
        "growth interval: {:?}",
        step(INITIAL_SCALE, GROWTH_INTERVAL - 1, false)
    );
    println!("overflow halves: {:?}", step(INITIAL_SCALE, 0, true));
    println!("at min, overflow: {:?}", step(MIN_SCALE, 0, true));
    println!("invalid: {:?}", step(0.5, 0, false));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaler_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn normal_step_no_change() {
        let v = step(INITIAL_SCALE, 100, false);
        if let ScalerVerdict::OptimizerStepNoChange { scale } = v {
            assert_eq!(scale, INITIAL_SCALE);
        }
    }

    #[test]
    fn at_growth_interval_doubles() {
        let v = step(INITIAL_SCALE, GROWTH_INTERVAL - 1, false);
        if let ScalerVerdict::OptimizerStepThenGrow { new_scale } = v {
            assert_eq!(new_scale, INITIAL_SCALE * 2.0);
        }
    }

    #[test]
    fn overflow_halves_scale() {
        let v = step(INITIAL_SCALE, 0, true);
        if let ScalerVerdict::SkipStepHalveScale { new_scale } = v {
            assert_eq!(new_scale, INITIAL_SCALE / 2.0);
        }
    }

    #[test]
    fn overflow_at_min_underflow() {
        let v = step(MIN_SCALE, 0, true);
        assert_eq!(v, ScalerVerdict::UnderflowScaleAtMinimum);
    }

    #[test]
    fn invalid_below_min_rejected() {
        assert_eq!(step(0.5, 0, false), ScalerVerdict::InvalidScale);
    }

    #[test]
    fn nan_scale_rejected() {
        assert_eq!(step(f64::NAN, 0, false), ScalerVerdict::InvalidScale);
    }

    #[test]
    fn growth_clamped_to_max() {
        // current at MAX_SCALE → grow returns MAX_SCALE.
        let v = step(MAX_SCALE, GROWTH_INTERVAL - 1, false);
        if let ScalerVerdict::OptimizerStepThenGrow { new_scale } = v {
            assert_eq!(new_scale, MAX_SCALE);
        }
    }

    #[test]
    fn halving_floored_at_min() {
        // current at MIN_SCALE × 2 → halve to MIN_SCALE.
        let v = step(MIN_SCALE * 2.0, 0, true);
        if let ScalerVerdict::SkipStepHalveScale { new_scale } = v {
            assert_eq!(new_scale, MIN_SCALE);
        }
    }

    #[test]
    fn overflow_takes_precedence_over_growth() {
        // Even at growth_interval, overflow halves and skips.
        let v = step(INITIAL_SCALE, GROWTH_INTERVAL - 1, true);
        assert!(matches!(v, ScalerVerdict::SkipStepHalveScale { .. }));
    }

    #[test]
    fn just_below_growth_interval_no_change() {
        let v = step(INITIAL_SCALE, GROWTH_INTERVAL - 2, false);
        assert!(matches!(v, ScalerVerdict::OptimizerStepNoChange { .. }));
    }
}
