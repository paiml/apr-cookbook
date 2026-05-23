//! # Dataset Distillation Synthetic-Sample Ratio
//!
//! Dataset distillation compresses N real samples into K << N
//! synthetic samples that train a comparable model. Practical ratios:
//! K/N ∈ [0.001, 0.1]; below 0.001 collapses; above 0.1 = "small
//! dataset" not "distilled". This recipe builds the validator + budget
//! calculator (synth samples × IPC where IPC = images-per-class).
//!
//! Demonstrates the **DISTILL.6** recipe for PMAT-124 (distillation coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Wang et al. (2018). Dataset Distillation. arXiv:1811.10959.
//!
//! Run with: cargo run --example distill_dataset_synth_ratio
//!
//! Added by PMAT-124 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RatioVerdict {
    Ok { ratio: f64 },
    BelowFloor,
    AboveCeiling,
    InvalidCounts,
}

const MIN_RATIO: f64 = 0.001;
const MAX_RATIO: f64 = 0.1;

pub fn validate(num_synth: u64, num_real: u64) -> RatioVerdict {
    if num_real == 0 || num_synth == 0 {
        return RatioVerdict::InvalidCounts;
    }
    let ratio = num_synth as f64 / num_real as f64;
    if ratio < MIN_RATIO {
        RatioVerdict::BelowFloor
    } else if ratio > MAX_RATIO {
        RatioVerdict::AboveCeiling
    } else {
        RatioVerdict::Ok { ratio }
    }
}

#[derive(Debug, PartialEq)]
pub enum BudgetVerdict {
    Ok { total_samples: u64 },
    InvalidIpc,
    InvalidClassCount,
}

pub fn budget_total(images_per_class: u32, num_classes: u32) -> BudgetVerdict {
    if images_per_class == 0 {
        return BudgetVerdict::InvalidIpc;
    }
    if num_classes == 0 {
        return BudgetVerdict::InvalidClassCount;
    }
    BudgetVerdict::Ok {
        total_samples: u64::from(images_per_class) * u64::from(num_classes),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_dataset_synth_ratio")?;

    let cases = [
        (10u64, 60_000u64), // ratio ≈ 0.000167 — below floor
        (100, 60_000),      // 0.00167 — ok
        (1000, 60_000),     // 0.0167 — ok
        (10_000, 60_000),   // 0.167 — above ceiling
        (0, 60_000),        // invalid
    ];
    for (s, r) in cases {
        println!("synth={s} real={r}  →  {:?}", validate(s, r));
    }
    println!("budget(10 IPC × 100 classes) = {:?}", budget_total(10, 100));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_ratio_passes() {
        let v = validate(100, 60_000);
        assert!(matches!(v, RatioVerdict::Ok { .. }));
    }

    #[test]
    fn below_floor_rejected() {
        // 10 / 60K = 0.000167 < 0.001 → reject.
        assert_eq!(validate(10, 60_000), RatioVerdict::BelowFloor);
    }

    #[test]
    fn above_ceiling_rejected() {
        // 10K / 60K = 0.167 > 0.1 → reject.
        assert_eq!(validate(10_000, 60_000), RatioVerdict::AboveCeiling);
    }

    #[test]
    fn at_floor_passes() {
        // 60 / 60_000 = 0.001 (boundary inclusive).
        let v = validate(60, 60_000);
        assert!(matches!(v, RatioVerdict::Ok { .. }));
    }

    #[test]
    fn at_ceiling_passes() {
        // 6000 / 60_000 = 0.1 (boundary inclusive).
        let v = validate(6_000, 60_000);
        assert!(matches!(v, RatioVerdict::Ok { .. }));
    }

    #[test]
    fn zero_real_rejected() {
        assert_eq!(validate(100, 0), RatioVerdict::InvalidCounts);
    }

    #[test]
    fn zero_synth_rejected() {
        assert_eq!(validate(0, 60_000), RatioVerdict::InvalidCounts);
    }

    #[test]
    fn budget_typical_passes() {
        // 10 IPC × 100 classes = 1000 synth samples.
        assert_eq!(
            budget_total(10, 100),
            BudgetVerdict::Ok {
                total_samples: 1000
            }
        );
    }

    #[test]
    fn budget_zero_ipc_rejected() {
        assert_eq!(budget_total(0, 100), BudgetVerdict::InvalidIpc);
    }

    #[test]
    fn budget_zero_classes_rejected() {
        assert_eq!(budget_total(10, 0), BudgetVerdict::InvalidClassCount);
    }
}
