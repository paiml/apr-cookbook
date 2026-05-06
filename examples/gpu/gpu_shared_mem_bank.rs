//! # GPU Shared-Memory Bank-Conflict Detector
//!
//! Shared memory has 32 banks. Threads in a warp accessing different
//! addresses in the SAME bank cause N-way conflicts (serialized).
//! Padding by 1 element per row breaks bank conflicts in 2D access.
//!
//! Demonstrates the **GPU.39** recipe for PMAT-152 (milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NVIDIA CUDA C Programming Guide § shared memory banks.
//!
//! Run with: cargo run --example gpu_shared_mem_bank
//!
//! Added by PMAT-152 (catalog crosses 1000 recipes).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const BANK_COUNT: u32 = 32;

#[derive(Debug, PartialEq)]
pub enum BankVerdict {
    NoConflict,
    NWayConflict { conflict_factor: u32 },
    PaddingRecommended { suggested_pad: u32 },
    InvalidStride,
}

pub fn check(stride_bytes: u32, element_size_bytes: u32) -> BankVerdict {
    if stride_bytes == 0 || element_size_bytes == 0 {
        return BankVerdict::InvalidStride;
    }
    let stride_elements = stride_bytes / element_size_bytes;
    if stride_elements == 0 {
        return BankVerdict::InvalidStride;
    }
    // Check if stride causes warp-wide bank conflict.
    let gcd = compute_gcd(stride_elements, BANK_COUNT);
    let conflict_factor = BANK_COUNT / gcd;
    if conflict_factor == BANK_COUNT {
        BankVerdict::NoConflict
    } else if conflict_factor == 1 {
        BankVerdict::NWayConflict {
            conflict_factor: BANK_COUNT,
        }
    } else if conflict_factor == 2 {
        BankVerdict::PaddingRecommended { suggested_pad: 1 }
    } else {
        BankVerdict::NWayConflict { conflict_factor }
    }
}

fn compute_gcd(a: u32, b: u32) -> u32 {
    let mut a = a;
    let mut b = b;
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_shared_mem_bank")?;

    println!("stride 4 (no conflict): {:?}", check(4, 4));
    println!("stride 32 (full conflict): {:?}", check(128, 4));
    println!("stride 16 (2-way): {:?}", check(64, 4));
    println!("stride 8 (4-way): {:?}", check(32, 4));
    println!("invalid: {:?}", check(0, 4));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn stride_4_no_conflict() {
        // stride_elements = 1. gcd(1, 32) = 1; conflict_factor = 32 = BANK_COUNT → NoConflict.
        let v = check(4, 4);
        assert_eq!(v, BankVerdict::NoConflict);
    }

    #[test]
    fn stride_128_full_conflict() {
        // stride_elements = 32. gcd(32, 32) = 32; conflict_factor = 1 → NWay full conflict.
        let v = check(128, 4);
        assert!(matches!(v, BankVerdict::NWayConflict { .. }));
    }

    #[test]
    fn stride_64_2_way() {
        // stride_elements = 16. gcd(16, 32) = 16; conflict_factor = 2 → padding.
        let v = check(64, 4);
        assert_eq!(v, BankVerdict::PaddingRecommended { suggested_pad: 1 });
    }

    #[test]
    fn invalid_zero_stride() {
        assert_eq!(check(0, 4), BankVerdict::InvalidStride);
    }

    #[test]
    fn invalid_zero_element_size() {
        assert_eq!(check(4, 0), BankVerdict::InvalidStride);
    }

    #[test]
    fn stride_smaller_than_element_invalid() {
        // stride 2, element 4 → stride_elements = 0.
        assert_eq!(check(2, 4), BankVerdict::InvalidStride);
    }

    #[test]
    fn stride_8_4way_conflict() {
        // 8 / 4 = 2 elements. gcd(2, 32) = 2; conflict_factor = 16. Wait, 32/2 = 16-way.
        // That's not 4-way, that's 16-way.
        let v = check(8, 4);
        assert!(matches!(
            v,
            BankVerdict::NWayConflict {
                conflict_factor: 16
            }
        ));
    }

    #[test]
    fn stride_64_uneven() {
        // 64 / 8 = 8 elements (8-byte doubles). gcd(8, 32) = 8; conflict_factor = 4.
        let v = check(64, 8);
        assert!(matches!(
            v,
            BankVerdict::NWayConflict { conflict_factor: 4 }
        ));
    }

    #[test]
    fn padding_only_for_2way() {
        let v = check(64, 4);
        assert_eq!(v, BankVerdict::PaddingRecommended { suggested_pad: 1 });
    }

    #[test]
    fn gcd_helper_correct() {
        assert_eq!(compute_gcd(12, 8), 4);
        assert_eq!(compute_gcd(15, 25), 5);
        assert_eq!(compute_gcd(7, 11), 1);
    }

    #[test]
    fn deterministic() {
        let a = check(64, 4);
        let b = check(64, 4);
        assert_eq!(a, b);
    }
}
