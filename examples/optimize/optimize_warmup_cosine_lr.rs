//! # Optimize Warmup + Cosine LR Scheduler
//!
//! Warmup ramps LR from 0 to peak over `warmup_steps`; cosine decay
//! reduces LR from peak to min_lr over remaining steps. Combined
//! schedule: lr(s) = peak × (s / warmup_steps) for s ≤ warmup;
//! min_lr + 0.5 × (peak − min_lr) × (1 + cos(π × t)) afterwards,
//! where t = (s − warmup_steps) / (total − warmup_steps). This recipe
//! builds the per-step LR + the warmup-fraction validator.
//!
//! Demonstrates the **OPT.25** recipe for PMAT-131 (optimize coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Loshchilov & Hutter (2017). SGDR: Stochastic Gradient Descent with Warm Restarts.
//!
//! Run with: cargo run --example optimize_warmup_cosine_lr
//!
//! Added by PMAT-131 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

use std::f64::consts::PI;

#[derive(Debug, PartialEq)]
pub enum LrVerdict {
    Ok { lr: f64 },
    StepBeyondTotal,
    InvalidTotal,
    InvalidWarmup,
    InvalidPeak,
}

pub fn schedule_lr(
    step: u32,
    total_steps: u32,
    warmup_steps: u32,
    peak_lr: f64,
    min_lr: f64,
) -> LrVerdict {
    if total_steps == 0 {
        return LrVerdict::InvalidTotal;
    }
    if warmup_steps >= total_steps {
        return LrVerdict::InvalidWarmup;
    }
    if !peak_lr.is_finite() || peak_lr <= 0.0 || min_lr < 0.0 || min_lr > peak_lr {
        return LrVerdict::InvalidPeak;
    }
    if step > total_steps {
        return LrVerdict::StepBeyondTotal;
    }
    let lr = if warmup_steps > 0 && step <= warmup_steps {
        peak_lr * f64::from(step) / f64::from(warmup_steps)
    } else {
        let progress = f64::from(step - warmup_steps) / f64::from(total_steps - warmup_steps);
        let t = progress.clamp(0.0, 1.0);
        min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + (PI * t).cos())
    };
    LrVerdict::Ok { lr }
}

pub fn warmup_fraction_pct(warmup_steps: u32, total_steps: u32) -> Option<f64> {
    if total_steps == 0 {
        return None;
    }
    Some(f64::from(warmup_steps) / f64::from(total_steps) * 100.0)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("optimize_warmup_cosine_lr")?;

    let total = 1000;
    let warmup = 100;
    let peak = 0.001;
    for step in [0u32, 50, 100, 500, 1000] {
        println!(
            "step={step}  →  {:?}",
            schedule_lr(step, total, warmup, peak, 0.0)
        );
    }
    println!("warmup pct: {:?}%", warmup_fraction_pct(warmup, total));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn lr_zero_at_step_zero_with_warmup() {
        if let LrVerdict::Ok { lr } = schedule_lr(0, 1000, 100, 0.001, 0.0) {
            assert!((lr - 0.0).abs() < 1e-12);
        }
    }

    #[test]
    fn lr_peaks_at_warmup_end() {
        if let LrVerdict::Ok { lr } = schedule_lr(100, 1000, 100, 0.001, 0.0) {
            assert!((lr - 0.001).abs() < 1e-9);
        }
    }

    #[test]
    fn lr_decays_after_warmup() {
        let early = schedule_lr(200, 1000, 100, 0.001, 0.0);
        let later = schedule_lr(500, 1000, 100, 0.001, 0.0);
        if let (LrVerdict::Ok { lr: e }, LrVerdict::Ok { lr: l }) = (early, later) {
            assert!(l < e);
        }
    }

    #[test]
    fn lr_at_min_at_total_steps() {
        if let LrVerdict::Ok { lr } = schedule_lr(1000, 1000, 100, 0.001, 0.0) {
            assert!(lr.abs() < 1e-9);
        }
    }

    #[test]
    fn cosine_midpoint_is_half_of_peak() {
        // At progress=0.5, LR = min + 0.5×(peak−min)×(1 + cos(π/2)) = min + 0.5×(peak−min) = midpoint.
        if let LrVerdict::Ok { lr } = schedule_lr(550, 1000, 100, 0.002, 0.0) {
            // (550-100)/(1000-100) = 0.5 → LR = 0.5×0.002 = 0.001.
            assert!((lr - 0.001).abs() < 1e-9);
        }
    }

    #[test]
    fn invalid_total_steps_rejected() {
        assert_eq!(schedule_lr(0, 0, 0, 0.001, 0.0), LrVerdict::InvalidTotal);
    }

    #[test]
    fn warmup_geq_total_rejected() {
        assert_eq!(
            schedule_lr(0, 100, 100, 0.001, 0.0),
            LrVerdict::InvalidWarmup
        );
    }

    #[test]
    fn invalid_peak_rejected() {
        assert_eq!(schedule_lr(0, 100, 10, -1.0, 0.0), LrVerdict::InvalidPeak);
        assert_eq!(schedule_lr(0, 100, 10, 0.001, 0.01), LrVerdict::InvalidPeak);
    }

    #[test]
    fn step_beyond_total_rejected() {
        assert_eq!(
            schedule_lr(2000, 1000, 100, 0.001, 0.0),
            LrVerdict::StepBeyondTotal
        );
    }

    #[test]
    fn warmup_pct_basic_math() {
        // 100 / 1000 = 10%.
        let pct = warmup_fraction_pct(100, 1000).unwrap();
        assert!((pct - 10.0).abs() < 1e-9);
    }
}
