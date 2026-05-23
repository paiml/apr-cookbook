//! # GPU Dynamic-Shape Bin Picker (CUDA Graph reuse)
//:
//! Variable batch sizes prevent CUDA graph reuse. Bin sizes round up
//! to the next supported shape:
//!   bins = [1, 2, 4, 8, 16, 32, 64, 128]
//! batch=10 → bin 16, batch=33 → bin 64.
//!
//! Picker reports actual_bin + waste_pct.
//!
//! Demonstrates the **GPU.33** recipe for PMAT-150 (gpu round 6).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: TensorRT dynamic-shape optimization profile.
//!
//! Run with: cargo run --example gpu_dynamic_shape_handler
//!
//! Added by PMAT-150 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BinVerdict {
    Ok { bin_size: u32, waste_pct: u32 },
    InvalidBatchSize,
    BatchTooLarge { max_bin: u32 },
}

const SUPPORTED_BINS: &[u32] = &[1, 2, 4, 8, 16, 32, 64, 128];

pub fn pick(batch_size: u32) -> BinVerdict {
    if batch_size == 0 {
        return BinVerdict::InvalidBatchSize;
    }
    let max = *SUPPORTED_BINS.last().unwrap();
    if batch_size > max {
        return BinVerdict::BatchTooLarge { max_bin: max };
    }
    let bin = SUPPORTED_BINS
        .iter()
        .find(|&&b| b >= batch_size)
        .copied()
        .unwrap();
    let waste = ((bin - batch_size) * 100) / bin;
    BinVerdict::Ok {
        bin_size: bin,
        waste_pct: waste,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_dynamic_shape_handler")?;

    println!("batch 1: {:?}", pick(1));
    println!("batch 10: {:?}", pick(10));
    println!("batch 33: {:?}", pick(33));
    println!("batch 128: {:?}", pick(128));
    println!("batch 200: {:?}", pick(200));
    println!("invalid: {:?}", pick(0));
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
    fn one_picks_one() {
        let v = pick(1);
        if let BinVerdict::Ok { bin_size, .. } = v {
            assert_eq!(bin_size, 1);
        }
    }

    #[test]
    fn ten_picks_sixteen() {
        let v = pick(10);
        if let BinVerdict::Ok { bin_size, .. } = v {
            assert_eq!(bin_size, 16);
        }
    }

    #[test]
    fn thirty_three_picks_sixty_four() {
        let v = pick(33);
        if let BinVerdict::Ok { bin_size, .. } = v {
            assert_eq!(bin_size, 64);
        }
    }

    #[test]
    fn at_max_bin() {
        let v = pick(128);
        if let BinVerdict::Ok {
            bin_size,
            waste_pct,
            ..
        } = v
        {
            assert_eq!(bin_size, 128);
            assert_eq!(waste_pct, 0);
        }
    }

    #[test]
    fn above_max_rejected() {
        let v = pick(200);
        assert!(matches!(v, BinVerdict::BatchTooLarge { .. }));
    }

    #[test]
    fn invalid_zero_batch() {
        assert_eq!(pick(0), BinVerdict::InvalidBatchSize);
    }

    #[test]
    fn waste_proportional_to_padding() {
        // batch 9, bin 16 → waste = 7/16 ≈ 43%.
        let v = pick(9);
        if let BinVerdict::Ok { waste_pct, .. } = v {
            assert!(waste_pct >= 40);
            assert!(waste_pct <= 50);
        }
    }

    #[test]
    fn exact_bin_zero_waste() {
        let v = pick(16);
        if let BinVerdict::Ok { waste_pct, .. } = v {
            assert_eq!(waste_pct, 0);
        }
    }

    #[test]
    fn powers_of_two_no_waste() {
        for bin in &[1u32, 2, 4, 8, 16, 32, 64, 128] {
            if let BinVerdict::Ok { waste_pct, .. } = pick(*bin) {
                assert_eq!(waste_pct, 0);
            }
        }
    }

    #[test]
    fn small_off_pick_next_bin() {
        // batch 3 → bin 4.
        let v = pick(3);
        if let BinVerdict::Ok { bin_size, .. } = v {
            assert_eq!(bin_size, 4);
        }
    }
}
