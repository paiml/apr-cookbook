//! # Training Eval-Step Cadence Picker
//!
//! Evaluating too often = wasted GPU time (can be 5-15% of training).
//! Evaluating too rarely = miss the best checkpoint or loss-spike.
//! Heuristic: pick eval_every_steps so eval consumes ≤ 5% of training
//! wall time. eval_every = ceil(eval_duration_s / (training_step_s × 0.05)).
//! This recipe builds the picker.
//!
//! Demonstrates the **TRAIN.12** recipe for PMAT-135 (training coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: PyTorch Lightning eval-cadence guide.
//!
//! Run with: cargo run --example training_eval_cadence
//!
//! Added by PMAT-135 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const DEFAULT_OVERHEAD_BUDGET: f64 = 0.05;
const MIN_CADENCE_STEPS: u32 = 1;
const MAX_CADENCE_STEPS: u32 = 100_000;

#[derive(Debug, PartialEq)]
pub enum CadenceVerdict {
    Ok {
        eval_every_steps: u32,
        overhead_pct: f64,
    },
    InvalidStepDuration,
    InvalidEvalDuration,
    EvalLongerThanReasonable,
}

pub fn pick(
    training_step_secs: f64,
    eval_duration_secs: f64,
    overhead_budget: f64,
) -> CadenceVerdict {
    if !training_step_secs.is_finite() || training_step_secs <= 0.0 {
        return CadenceVerdict::InvalidStepDuration;
    }
    if !eval_duration_secs.is_finite() || eval_duration_secs <= 0.0 {
        return CadenceVerdict::InvalidEvalDuration;
    }
    if !overhead_budget.is_finite() || overhead_budget <= 0.0 || overhead_budget > 1.0 {
        return CadenceVerdict::InvalidEvalDuration;
    }
    if eval_duration_secs > training_step_secs * f64::from(MAX_CADENCE_STEPS) {
        return CadenceVerdict::EvalLongerThanReasonable;
    }
    let raw = (eval_duration_secs / (training_step_secs * overhead_budget)).ceil() as u32;
    let cadence = raw.clamp(MIN_CADENCE_STEPS, MAX_CADENCE_STEPS);
    let overhead_pct = (eval_duration_secs
        / (training_step_secs * f64::from(cadence) + eval_duration_secs))
        * 100.0;
    CadenceVerdict::Ok {
        eval_every_steps: cadence,
        overhead_pct,
    }
}

pub fn pick_default(training_step_secs: f64, eval_duration_secs: f64) -> CadenceVerdict {
    pick(
        training_step_secs,
        eval_duration_secs,
        DEFAULT_OVERHEAD_BUDGET,
    )
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("training_eval_cadence")?;

    for (step_s, eval_s) in [(0.5_f64, 30.0_f64), (0.1, 60.0), (2.0, 10.0)] {
        println!(
            "step={step_s}s eval={eval_s}s → {:?}",
            pick_default(step_s, eval_s)
        );
    }
    println!("zero step: {:?}", pick_default(0.0, 30.0));
    println!("custom 10% overhead: {:?}", pick(0.5, 30.0, 0.1));
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
    fn typical_cadence_balanced() {
        // step 0.5s, eval 30s, budget 5% → 30 / (0.5 × 0.05) = 1200 steps.
        let v = pick_default(0.5, 30.0);
        if let CadenceVerdict::Ok {
            eval_every_steps, ..
        } = v
        {
            assert_eq!(eval_every_steps, 1200);
        }
    }

    #[test]
    fn fast_step_long_eval_high_cadence() {
        // step 0.1s, eval 60s, budget 5% → 60 / (0.1 × 0.05) = 12000 steps.
        let v = pick_default(0.1, 60.0);
        if let CadenceVerdict::Ok {
            eval_every_steps, ..
        } = v
        {
            assert_eq!(eval_every_steps, 12000);
        }
    }

    #[test]
    fn slow_step_short_eval_low_cadence() {
        // step 2s, eval 10s, budget 5% → 10 / (2 × 0.05) = 100 steps.
        let v = pick_default(2.0, 10.0);
        if let CadenceVerdict::Ok {
            eval_every_steps, ..
        } = v
        {
            assert_eq!(eval_every_steps, 100);
        }
    }

    #[test]
    fn higher_budget_reduces_cadence() {
        // 10% budget → cadence half of 5% budget.
        let v_5 = pick(0.5, 30.0, 0.05);
        let v_10 = pick(0.5, 30.0, 0.10);
        if let (
            CadenceVerdict::Ok {
                eval_every_steps: c5,
                ..
            },
            CadenceVerdict::Ok {
                eval_every_steps: c10,
                ..
            },
        ) = (v_5, v_10)
        {
            assert!(c10 < c5);
        }
    }

    #[test]
    fn zero_step_invalid() {
        assert_eq!(pick_default(0.0, 30.0), CadenceVerdict::InvalidStepDuration);
    }

    #[test]
    fn zero_eval_invalid() {
        assert_eq!(pick_default(0.5, 0.0), CadenceVerdict::InvalidEvalDuration);
    }

    #[test]
    fn nan_step_invalid() {
        assert_eq!(
            pick_default(f64::NAN, 30.0),
            CadenceVerdict::InvalidStepDuration
        );
    }

    #[test]
    fn budget_above_one_invalid() {
        assert_eq!(pick(0.5, 30.0, 1.5), CadenceVerdict::InvalidEvalDuration);
    }

    #[test]
    fn overhead_pct_under_budget() {
        // Computed cadence should keep overhead at or under budget.
        if let CadenceVerdict::Ok { overhead_pct, .. } = pick_default(0.5, 30.0) {
            assert!(overhead_pct <= 5.0);
        }
    }

    #[test]
    fn cadence_clamped_to_minimum() {
        // Eval much shorter than step → cadence floor at 1.
        let v = pick(10.0, 0.1, 0.5);
        if let CadenceVerdict::Ok {
            eval_every_steps, ..
        } = v
        {
            assert!(eval_every_steps >= MIN_CADENCE_STEPS);
        }
    }
}
