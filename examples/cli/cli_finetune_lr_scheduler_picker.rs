//! # apr finetune --lr-scheduler — Schedule Picker
//!
//! `apr finetune --lr-scheduler <S>` accepts {constant, linear, cosine,
//! cosine_warm_restarts, exponential}. Decision rules: cosine is the
//! modern default for ≥ 1K steps; linear suits short jobs (< 500 steps);
//! exponential degrades over long training. Warmup ratio defaults to 3%.
//!
//! Demonstrates the **FT.5** recipe for PMAT-113 (apr finetune coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender FT-001 + Loshchilov & Hutter 2017 (SGDR cosine)
//!
//! Run with: cargo run --example cli_finetune_lr_scheduler_picker
//!
//! Added by PMAT-113 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LrScheduler {
    Constant,
    Linear,
    Cosine,
    CosineWarmRestarts,
    Exponential,
}

impl LrScheduler {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "constant" => Some(LrScheduler::Constant),
            "linear" => Some(LrScheduler::Linear),
            "cosine" => Some(LrScheduler::Cosine),
            "cosine_warm_restarts" => Some(LrScheduler::CosineWarmRestarts),
            "exponential" => Some(LrScheduler::Exponential),
            _ => None,
        }
    }
}

const SHORT_JOB_THRESHOLD: u32 = 500;
const LONG_JOB_THRESHOLD: u32 = 1000;
const DEFAULT_WARMUP_RATIO: f64 = 0.03;

pub fn auto_pick_scheduler(num_steps: u32) -> LrScheduler {
    if num_steps == 0 {
        return LrScheduler::Constant;
    }
    if num_steps < SHORT_JOB_THRESHOLD {
        LrScheduler::Linear
    } else if num_steps < LONG_JOB_THRESHOLD {
        LrScheduler::Cosine
    } else {
        LrScheduler::CosineWarmRestarts
    }
}

pub fn warmup_steps(num_steps: u32, ratio: f64) -> u32 {
    if !ratio.is_finite() || !(0.0..=1.0).contains(&ratio) {
        return 0;
    }
    (num_steps as f64 * ratio).round() as u32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_finetune_lr_scheduler_picker")?;

    for n in [50u32, 400, 800, 5000] {
        let s = auto_pick_scheduler(n);
        let w = warmup_steps(n, DEFAULT_WARMUP_RATIO);
        println!("steps={n:>5} → {s:?}  warmup={w}");
    }
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
    fn short_job_picks_linear() {
        assert_eq!(auto_pick_scheduler(100), LrScheduler::Linear);
        assert_eq!(auto_pick_scheduler(499), LrScheduler::Linear);
    }

    #[test]
    fn medium_job_picks_cosine() {
        assert_eq!(auto_pick_scheduler(500), LrScheduler::Cosine);
        assert_eq!(auto_pick_scheduler(999), LrScheduler::Cosine);
    }

    #[test]
    fn long_job_picks_cosine_warm_restarts() {
        assert_eq!(auto_pick_scheduler(1000), LrScheduler::CosineWarmRestarts);
        assert_eq!(auto_pick_scheduler(50000), LrScheduler::CosineWarmRestarts);
    }

    #[test]
    fn zero_steps_constant() {
        assert_eq!(auto_pick_scheduler(0), LrScheduler::Constant);
    }

    #[test]
    fn warmup_3pct_of_1000_is_30() {
        assert_eq!(warmup_steps(1000, 0.03), 30);
    }

    #[test]
    fn warmup_invalid_ratio_yields_zero() {
        assert_eq!(warmup_steps(1000, -0.1), 0);
        assert_eq!(warmup_steps(1000, 1.5), 0);
        assert_eq!(warmup_steps(1000, f64::NAN), 0);
    }

    #[test]
    fn warmup_zero_steps_yields_zero() {
        assert_eq!(warmup_steps(0, 0.03), 0);
    }

    #[test]
    fn known_schedulers_round_trip() {
        for s in [
            "constant",
            "linear",
            "cosine",
            "cosine_warm_restarts",
            "exponential",
        ] {
            assert!(LrScheduler::from_str_strict(s).is_some());
        }
        assert!(LrScheduler::from_str_strict("triangular").is_none());
    }
}
