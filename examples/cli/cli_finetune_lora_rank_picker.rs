//! # apr finetune --method lora --rank — Rank Picker
//!
//! `apr finetune --method lora --rank <R>` controls the LoRA bottleneck.
//! Rules: r ≥ 4 for usable expressivity; r ≤ 256 (beyond is dense
//! finetune in disguise); r should be a power of 2 for SIMD-friendly
//! kernels. Auto-pick by base-model size: 7B → r=8, 13B → r=16, 70B → r=32.
//!
//! Demonstrates the **FT.4** recipe for PMAT-113 (apr finetune coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender FT-001 + Hu et al. 2021 (LoRA)
//!
//! Run with: cargo run --example cli_finetune_lora_rank_picker
//!
//! Added by PMAT-113 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RankVerdict {
    Ok,
    BelowMinimum,
    AboveMaximum,
    NotPowerOfTwo,
}

const MIN_RANK: u32 = 4;
const MAX_RANK: u32 = 256;

pub fn validate_rank(rank: u32) -> RankVerdict {
    if rank < MIN_RANK {
        return RankVerdict::BelowMinimum;
    }
    if rank > MAX_RANK {
        return RankVerdict::AboveMaximum;
    }
    if !rank.is_power_of_two() {
        return RankVerdict::NotPowerOfTwo;
    }
    RankVerdict::Ok
}

pub fn auto_pick_rank(base_params: u64) -> u32 {
    if base_params < 10_000_000_000 {
        // < 10B → r=8
        8
    } else if base_params < 30_000_000_000 {
        // 10B-30B → r=16
        16
    } else {
        // >= 30B → r=32
        32
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_finetune_lora_rank_picker")?;

    for r in [2u32, 4, 6, 8, 16, 32, 256, 512] {
        println!("r={r:>3}  →  {:?}", validate_rank(r));
    }
    for size in [7_000_000_000u64, 13_000_000_000, 70_000_000_000] {
        println!("auto({size:>13}B)  →  r={}", auto_pick_rank(size));
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
    fn rank_below_minimum_rejected() {
        assert_eq!(validate_rank(2), RankVerdict::BelowMinimum);
        assert_eq!(validate_rank(0), RankVerdict::BelowMinimum);
    }

    #[test]
    fn rank_at_minimum_passes() {
        assert_eq!(validate_rank(MIN_RANK), RankVerdict::Ok);
    }

    #[test]
    fn rank_above_maximum_rejected() {
        assert_eq!(validate_rank(512), RankVerdict::AboveMaximum);
    }

    #[test]
    fn non_power_of_two_rejected() {
        // 6 is in [4, 256] but not 2^k.
        assert_eq!(validate_rank(6), RankVerdict::NotPowerOfTwo);
        assert_eq!(validate_rank(100), RankVerdict::NotPowerOfTwo);
    }

    #[test]
    fn typical_lora_ranks_pass() {
        for r in [4u32, 8, 16, 32, 64, 128, 256] {
            assert_eq!(validate_rank(r), RankVerdict::Ok, "r={r}");
        }
    }

    #[test]
    fn auto_pick_7b_yields_8() {
        assert_eq!(auto_pick_rank(7_000_000_000), 8);
    }

    #[test]
    fn auto_pick_13b_yields_16() {
        assert_eq!(auto_pick_rank(13_000_000_000), 16);
    }

    #[test]
    fn auto_pick_70b_yields_32() {
        assert_eq!(auto_pick_rank(70_000_000_000), 32);
    }

    #[test]
    fn auto_pick_results_always_valid() {
        for size in [1u64, 5_000_000_000, 30_000_000_000, 175_000_000_000] {
            let r = auto_pick_rank(size);
            assert_eq!(validate_rank(r), RankVerdict::Ok);
        }
    }
}
