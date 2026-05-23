//! # apr finetune --grad-accum — Effective Batch Budget
//!
//! `apr finetune --batch-size <B> --grad-accum <K>` yields effective
//! batch B×K. Rules: effective batch must equal a target (e.g., 256 on
//! 7B for stability); per-device B must fit GPU memory; K ≥ 1. This
//! recipe builds the budget validator + auto K-picker.
//!
//! Demonstrates the **FT.6** recipe for PMAT-113 (apr finetune coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender FT-001 + Smith et al. 2018 (linear scaling)
//!
//! Run with: cargo run --example cli_finetune_grad_accum_budget
//!
//! Added by PMAT-113 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BudgetVerdict {
    Ok { effective_batch: u32 },
    InvalidPerDevice,
    InvalidGradAccum,
    EffectiveExceedsTarget { effective: u32, target: u32 },
    EffectiveBelowTarget { effective: u32, target: u32 },
}

pub fn validate(per_device: u32, grad_accum: u32, target_effective: u32) -> BudgetVerdict {
    if per_device == 0 {
        return BudgetVerdict::InvalidPerDevice;
    }
    if grad_accum == 0 {
        return BudgetVerdict::InvalidGradAccum;
    }
    let effective = per_device * grad_accum;
    if target_effective == 0 {
        return BudgetVerdict::Ok {
            effective_batch: effective,
        };
    }
    match effective.cmp(&target_effective) {
        std::cmp::Ordering::Equal => BudgetVerdict::Ok {
            effective_batch: effective,
        },
        std::cmp::Ordering::Greater => BudgetVerdict::EffectiveExceedsTarget {
            effective,
            target: target_effective,
        },
        std::cmp::Ordering::Less => BudgetVerdict::EffectiveBelowTarget {
            effective,
            target: target_effective,
        },
    }
}

pub fn auto_pick_grad_accum(per_device: u32, target_effective: u32) -> Option<u32> {
    if per_device == 0 || target_effective == 0 {
        return None;
    }
    if target_effective % per_device != 0 {
        return None;
    }
    Some(target_effective / per_device)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_finetune_grad_accum_budget")?;

    let cases = [(4u32, 64, 256), (4, 32, 256), (4, 128, 256), (0, 1, 256)];
    for (b, k, target) in cases {
        println!("B={b} K={k} target={target} → {:?}", validate(b, k, target));
    }
    println!(
        "auto_pick(B=8, target=256) = {:?}",
        auto_pick_grad_accum(8, 256)
    );
    println!(
        "auto_pick(B=7, target=256) = {:?}",
        auto_pick_grad_accum(7, 256)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn exact_target_match_ok() {
        // 4 × 64 = 256.
        let v = validate(4, 64, 256);
        assert!(matches!(
            v,
            BudgetVerdict::Ok {
                effective_batch: 256
            }
        ));
    }

    #[test]
    fn target_zero_means_no_check() {
        let v = validate(4, 8, 0);
        assert!(matches!(
            v,
            BudgetVerdict::Ok {
                effective_batch: 32
            }
        ));
    }

    #[test]
    fn effective_below_target_rejected() {
        let v = validate(4, 32, 256);
        assert!(matches!(v, BudgetVerdict::EffectiveBelowTarget { .. }));
    }

    #[test]
    fn effective_exceeds_target_rejected() {
        let v = validate(4, 128, 256);
        assert!(matches!(v, BudgetVerdict::EffectiveExceedsTarget { .. }));
    }

    #[test]
    fn zero_per_device_rejected() {
        assert_eq!(validate(0, 1, 256), BudgetVerdict::InvalidPerDevice);
    }

    #[test]
    fn zero_grad_accum_rejected() {
        assert_eq!(validate(4, 0, 256), BudgetVerdict::InvalidGradAccum);
    }

    #[test]
    fn auto_pick_divisible_yields_quotient() {
        assert_eq!(auto_pick_grad_accum(8, 256), Some(32));
    }

    #[test]
    fn auto_pick_non_divisible_yields_none() {
        // 256 not divisible by 7.
        assert!(auto_pick_grad_accum(7, 256).is_none());
    }

    #[test]
    fn auto_pick_zero_inputs_yield_none() {
        assert!(auto_pick_grad_accum(0, 256).is_none());
        assert!(auto_pick_grad_accum(8, 0).is_none());
    }
}
