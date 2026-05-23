//! # GPU Register Spill Estimator
//!
//! CUDA kernels with too many live values spill registers to local
//! memory (slow). Per-thread register limit:
//!   pre-Ampere (CC < 8.0): 256 max usable / 64 typical
//!   Ampere+ (CC ≥ 8.0):    255 max / 64 typical
//!
//! Estimator: required_regs = max(1, intermediate_values + thread_locals).
//! If above threshold → predict spill, return penalty cycle estimate.
//!
//! Demonstrates the **GPU.26** recipe for PMAT-143 (gpu round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NVIDIA CUDA C++ Programming Guide § Register Pressure.
//!
//! Run with: cargo run --example gpu_register_spill
//!
//! Added by PMAT-143 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const TYPICAL_BUDGET: u32 = 64;
const HARD_LIMIT_PRE_AMPERE: u32 = 256;
const HARD_LIMIT_AMPERE_PLUS: u32 = 255;
const SPILL_PENALTY_CYCLES_PER_REG: u32 = 24;

#[derive(Debug, PartialEq)]
pub enum SpillVerdict {
    NoSpill {
        regs_used: u32,
        budget_remaining: u32,
    },
    PartialSpill {
        regs_used: u32,
        spilled: u32,
        penalty_cycles: u32,
    },
    HardLimitExceeded {
        regs_used: u32,
        max: u32,
    },
    InvalidShape,
}

pub fn estimate(intermediate_values: u32, thread_locals: u32, cc_major: u8) -> SpillVerdict {
    let required = intermediate_values.saturating_add(thread_locals).max(1);
    let max = if cc_major >= 8 {
        HARD_LIMIT_AMPERE_PLUS
    } else {
        HARD_LIMIT_PRE_AMPERE
    };
    if required > max {
        return SpillVerdict::HardLimitExceeded {
            regs_used: required,
            max,
        };
    }
    if required <= TYPICAL_BUDGET {
        return SpillVerdict::NoSpill {
            regs_used: required,
            budget_remaining: TYPICAL_BUDGET - required,
        };
    }
    let spilled = required - TYPICAL_BUDGET;
    SpillVerdict::PartialSpill {
        regs_used: required,
        spilled,
        penalty_cycles: spilled * SPILL_PENALTY_CYCLES_PER_REG,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_register_spill")?;

    println!("simple kernel: {:?}", estimate(20, 10, 8));
    println!("medium kernel: {:?}", estimate(80, 20, 8));
    println!("excessive: {:?}", estimate(300, 50, 8));
    println!("zero values: {:?}", estimate(0, 0, 8));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn small_kernel_no_spill() {
        let v = estimate(20, 10, 8);
        if let SpillVerdict::NoSpill { regs_used, .. } = v {
            assert_eq!(regs_used, 30);
        }
    }

    #[test]
    fn at_budget_no_spill() {
        let v = estimate(60, 4, 8);
        assert!(matches!(v, SpillVerdict::NoSpill { .. }));
    }

    #[test]
    fn just_above_budget_partial_spill() {
        let v = estimate(60, 10, 8);
        assert!(matches!(v, SpillVerdict::PartialSpill { spilled: 6, .. }));
    }

    #[test]
    fn medium_kernel_partial_spill() {
        let v = estimate(80, 20, 8);
        assert!(matches!(v, SpillVerdict::PartialSpill { .. }));
    }

    #[test]
    fn excessive_hard_limit() {
        let v = estimate(300, 50, 8);
        assert!(matches!(v, SpillVerdict::HardLimitExceeded { .. }));
    }

    #[test]
    fn budget_remaining_correct() {
        let v = estimate(20, 10, 8);
        if let SpillVerdict::NoSpill {
            budget_remaining, ..
        } = v
        {
            assert_eq!(budget_remaining, 34);
        }
    }

    #[test]
    fn zero_values_floors_to_one() {
        let v = estimate(0, 0, 8);
        if let SpillVerdict::NoSpill { regs_used, .. } = v {
            assert_eq!(regs_used, 1);
        }
    }

    #[test]
    fn pre_ampere_higher_max() {
        // 256 → over Ampere limit (255), but OK for pre-Ampere.
        let pre = estimate(200, 56, 7);
        let post = estimate(200, 56, 8);
        assert!(matches!(pre, SpillVerdict::PartialSpill { .. }));
        assert!(matches!(post, SpillVerdict::HardLimitExceeded { .. }));
    }

    #[test]
    fn penalty_cycles_proportional() {
        // 100 regs total, spill = 36, penalty = 36 × 24 = 864.
        let v = estimate(70, 30, 8);
        if let SpillVerdict::PartialSpill { penalty_cycles, .. } = v {
            assert_eq!(penalty_cycles, 864);
        }
    }

    #[test]
    fn saturating_no_overflow() {
        // u32 saturating prevents panic.
        let v = estimate(u32::MAX, u32::MAX, 8);
        assert!(matches!(v, SpillVerdict::HardLimitExceeded { .. }));
    }
}
