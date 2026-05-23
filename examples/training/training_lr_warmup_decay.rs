//! # Training LR Warmup → Cosine Decay Schedule
//!
//! Standard transformer training schedule:
//! - Steps 0..warmup: lr = peak × (step / warmup)  (linear warmup)
//! - Steps warmup..total: lr = min + (peak - min) × 0.5 × (1 + cos(π × t))
//!   where t = (step - warmup) / (total - warmup)  (cosine decay)
//! - Steps ≥ total: lr = min
//!
//! This recipe builds the per-step picker.
//!
//! Demonstrates the **TRAIN.10** recipe for PMAT-135 (training coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Loshchilov & Hutter (2017). SGDR: Stochastic Gradient Descent with Warm Restarts. arXiv:1608.03983.
//!
//! Run with: cargo run --example training_lr_warmup_decay
//!
//! Added by PMAT-135 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::f64::consts::PI;

#[derive(Debug, PartialEq)]
pub enum LrVerdict {
    Ok { lr: f64 },
    InvalidPeakLr,
    InvalidMinLr,
    InvalidStepCount,
    WarmupExceedsTotal,
}

pub fn pick(
    step: u32,
    warmup_steps: u32,
    total_steps: u32,
    peak_lr: f64,
    min_lr: f64,
) -> LrVerdict {
    if !peak_lr.is_finite() || peak_lr <= 0.0 {
        return LrVerdict::InvalidPeakLr;
    }
    if !min_lr.is_finite() || min_lr < 0.0 || min_lr > peak_lr {
        return LrVerdict::InvalidMinLr;
    }
    if total_steps == 0 {
        return LrVerdict::InvalidStepCount;
    }
    if warmup_steps >= total_steps {
        return LrVerdict::WarmupExceedsTotal;
    }
    let lr = if step < warmup_steps {
        if warmup_steps == 0 {
            peak_lr
        } else {
            peak_lr * f64::from(step) / f64::from(warmup_steps)
        }
    } else if step >= total_steps {
        min_lr
    } else {
        let t = f64::from(step - warmup_steps) / f64::from(total_steps - warmup_steps);
        min_lr + (peak_lr - min_lr) * 0.5 * (1.0 + (PI * t).cos())
    };
    LrVerdict::Ok { lr }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("training_lr_warmup_decay")?;

    let warmup = 100u32;
    let total = 1000u32;
    let peak = 1e-3f64;
    let min = 1e-5f64;
    for step in [0u32, 50, 100, 500, 1000, 1500] {
        println!("step={step}: {:?}", pick(step, warmup, total, peak, min));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_runs() {
        main().expect("recipe execution failed");
    }

    fn lr_at(step: u32) -> f64 {
        if let LrVerdict::Ok { lr } = pick(step, 100, 1000, 1e-3, 1e-5) {
            lr
        } else {
            f64::NAN
        }
    }

    #[test]
    fn step_zero_lr_is_zero() {
        assert!(lr_at(0).abs() < 1e-12);
    }

    #[test]
    fn warmup_end_at_peak() {
        // step = warmup → lr just hits peak.
        let lr = if let LrVerdict::Ok { lr } = pick(100, 100, 1000, 1e-3, 1e-5) {
            lr
        } else {
            f64::NAN
        };
        assert!((lr - 1e-3).abs() < 1e-9);
    }

    #[test]
    fn mid_warmup_linear() {
        // step = 50 of warmup 100 → lr = 0.5 × peak.
        let lr = lr_at(50);
        assert!((lr - 0.5e-3).abs() < 1e-9);
    }

    #[test]
    fn end_of_total_decays_to_min() {
        // step >= total → min.
        let lr = if let LrVerdict::Ok { lr } = pick(1000, 100, 1000, 1e-3, 1e-5) {
            lr
        } else {
            f64::NAN
        };
        assert!((lr - 1e-5).abs() < 1e-9);
    }

    #[test]
    fn beyond_total_lr_clamped_to_min() {
        let lr = lr_at(1500);
        assert!((lr - 1e-5).abs() < 1e-9);
    }

    #[test]
    fn cosine_decay_monotone_decreasing_after_warmup() {
        let l1 = lr_at(200);
        let l2 = lr_at(500);
        let l3 = lr_at(800);
        assert!(l1 > l2);
        assert!(l2 > l3);
    }

    #[test]
    fn invalid_peak_lr_rejected() {
        assert_eq!(pick(0, 100, 1000, 0.0, 1e-5), LrVerdict::InvalidPeakLr);
        assert_eq!(pick(0, 100, 1000, -1.0, 1e-5), LrVerdict::InvalidPeakLr);
    }

    #[test]
    fn min_above_peak_rejected() {
        assert_eq!(pick(0, 100, 1000, 1e-5, 1e-3), LrVerdict::InvalidMinLr);
    }

    #[test]
    fn zero_total_invalid() {
        assert_eq!(pick(0, 100, 0, 1e-3, 1e-5), LrVerdict::InvalidStepCount);
    }

    #[test]
    fn warmup_above_total_rejected() {
        assert_eq!(
            pick(0, 1000, 100, 1e-3, 1e-5),
            LrVerdict::WarmupExceedsTotal
        );
    }

    #[test]
    fn cosine_at_quarter_decay_is_high() {
        // t = 0.25 → cos(0.25π) ≈ 0.707. lr ≈ min + (peak-min) × 0.854.
        let lr = lr_at(325); // (325-100)/(1000-100) = 0.25
        assert!(lr > 5e-4 && lr < 1e-3);
    }
}
