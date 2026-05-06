//! # SIMD Strided-Load vs Gather Picker
//!
//! Strided loads (every Nth element):
//!   stride 1 → contiguous load, fastest
//!   stride 2-8 → use shuffle/permute after contiguous load
//!   stride > 8 → use AVX-512 vpgatherdd (slow but only option)
//!
//! Demonstrates the **SIMD.15** recipe for PMAT-147 (simd round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Intel optimization manual § gather/scatter cost model.
//!
//! Run with: cargo run --example simd_strided_load
//!
//! Added by PMAT-147 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadStrategy {
    Contiguous,
    ShufflePack,
    Gather,
}

#[derive(Debug, PartialEq)]
pub enum LoadVerdict {
    Ok {
        strategy: LoadStrategy,
        cycles_estimate: u32,
    },
    InvalidStride,
}

pub fn pick(stride: u32, has_avx512: bool) -> LoadVerdict {
    if stride == 0 {
        return LoadVerdict::InvalidStride;
    }
    let strategy = if stride == 1 {
        LoadStrategy::Contiguous
    } else if stride <= 8 {
        LoadStrategy::ShufflePack
    } else if has_avx512 {
        LoadStrategy::Gather
    } else {
        // No AVX-512 + huge stride → fall back to gather but warn.
        LoadStrategy::Gather
    };
    let cycles = match strategy {
        LoadStrategy::Contiguous => 4,
        LoadStrategy::ShufflePack => 12,
        LoadStrategy::Gather => 50,
    };
    LoadVerdict::Ok {
        strategy,
        cycles_estimate: cycles,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("simd_strided_load")?;

    println!("stride 1: {:?}", pick(1, true));
    println!("stride 4: {:?}", pick(4, true));
    println!("stride 16: {:?}", pick(16, true));
    println!("stride 100: {:?}", pick(100, false));
    println!("invalid: {:?}", pick(0, true));
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
    fn stride_1_contiguous() {
        let v = pick(1, true);
        if let LoadVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, LoadStrategy::Contiguous);
        }
    }

    #[test]
    fn stride_4_shuffle_pack() {
        let v = pick(4, true);
        if let LoadVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, LoadStrategy::ShufflePack);
        }
    }

    #[test]
    fn stride_16_gather() {
        let v = pick(16, true);
        if let LoadVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, LoadStrategy::Gather);
        }
    }

    #[test]
    fn no_avx512_still_gathers_for_large() {
        let v = pick(100, false);
        if let LoadVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, LoadStrategy::Gather);
        }
    }

    #[test]
    fn invalid_zero_stride() {
        assert_eq!(pick(0, true), LoadVerdict::InvalidStride);
    }

    #[test]
    fn cycles_increase_with_complexity() {
        let v_1 = pick(1, true);
        let v_4 = pick(4, true);
        let v_16 = pick(16, true);
        if let (
            LoadVerdict::Ok {
                cycles_estimate: c1,
                ..
            },
            LoadVerdict::Ok {
                cycles_estimate: c4,
                ..
            },
            LoadVerdict::Ok {
                cycles_estimate: c16,
                ..
            },
        ) = (v_1, v_4, v_16)
        {
            assert!(c1 < c4);
            assert!(c4 < c16);
        }
    }

    #[test]
    fn boundary_at_stride_8_still_shuffle() {
        let v = pick(8, true);
        if let LoadVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, LoadStrategy::ShufflePack);
        }
    }

    #[test]
    fn just_above_8_gathers() {
        let v = pick(9, true);
        if let LoadVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, LoadStrategy::Gather);
        }
    }

    #[test]
    fn stride_2_shuffle() {
        let v = pick(2, true);
        if let LoadVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, LoadStrategy::ShufflePack);
        }
    }

    #[test]
    fn contiguous_cheapest() {
        let v = pick(1, true);
        if let LoadVerdict::Ok {
            cycles_estimate, ..
        } = v
        {
            assert!(cycles_estimate <= 10);
        }
    }
}
