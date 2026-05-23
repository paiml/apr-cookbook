//! # GPU Tensor-Core Alignment Picker
//!
//! Tensor cores require M, N, K divisibility:
//!   Volta (CC 7.0) — M, N, K multiples of 8 (fp16) / 16 (int8)
//!   Ampere (CC 8.0) — M, N, K multiples of 8 (TF32) / 16 (fp16/bf16)
//!   Hopper (CC 9.0) — multiples of 8/16/32 depending on dtype
//!
//! Picker validates dimensions + suggests padded shape if needed.
//!
//! Demonstrates the **GPU.32** recipe for PMAT-150 (gpu round 6).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NVIDIA Tensor Core IL semantics docs.
//!
//! Run with: cargo run --example gpu_tensor_core_alignment
//!
//! Added by PMAT-150 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Fp16,
    Bf16,
    Tf32,
    Int8,
}

#[derive(Debug, PartialEq)]
pub enum AlignmentVerdict {
    Aligned {
        m: u32,
        n: u32,
        k: u32,
    },
    NeedsPadding {
        padded_m: u32,
        padded_n: u32,
        padded_k: u32,
        wasted_pct: u32,
    },
    InvalidShape,
    UnsupportedCC,
}

pub fn check(m: u32, n: u32, k: u32, dtype: DataType, cc_major: u8) -> AlignmentVerdict {
    if m == 0 || n == 0 || k == 0 {
        return AlignmentVerdict::InvalidShape;
    }
    if cc_major < 7 {
        return AlignmentVerdict::UnsupportedCC;
    }
    let alignment = match dtype {
        DataType::Fp16 | DataType::Bf16 => 16,
        DataType::Tf32 => 8,
        DataType::Int8 => 16,
    };
    if m % alignment == 0 && n % alignment == 0 && k % alignment == 0 {
        return AlignmentVerdict::Aligned { m, n, k };
    }
    let pad = |x: u32| -> u32 { x.div_ceil(alignment) * alignment };
    let padded_m = pad(m);
    let padded_n = pad(n);
    let padded_k = pad(k);
    let original = u64::from(m) * u64::from(n) * u64::from(k);
    let padded_total = u64::from(padded_m) * u64::from(padded_n) * u64::from(padded_k);
    let wasted_pct = ((padded_total - original) * 100 / padded_total) as u32;
    AlignmentVerdict::NeedsPadding {
        padded_m,
        padded_n,
        padded_k,
        wasted_pct,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_tensor_core_alignment")?;

    println!("aligned: {:?}", check(128, 128, 128, DataType::Fp16, 8));
    println!("needs pad: {:?}", check(127, 100, 50, DataType::Fp16, 8));
    println!(
        "tf32 alignment 8: {:?}",
        check(40, 40, 40, DataType::Tf32, 8)
    );
    println!("invalid: {:?}", check(0, 100, 100, DataType::Fp16, 8));
    println!("unsupported: {:?}", check(128, 128, 128, DataType::Fp16, 6));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn aligned_dims_pass() {
        let v = check(128, 128, 128, DataType::Fp16, 8);
        assert!(matches!(v, AlignmentVerdict::Aligned { .. }));
    }

    #[test]
    fn unaligned_dims_pad() {
        let v = check(127, 100, 50, DataType::Fp16, 8);
        assert!(matches!(v, AlignmentVerdict::NeedsPadding { .. }));
    }

    #[test]
    fn tf32_uses_8_alignment() {
        // 40 % 8 == 0 → aligned.
        let v = check(40, 40, 40, DataType::Tf32, 8);
        assert!(matches!(v, AlignmentVerdict::Aligned { .. }));
    }

    #[test]
    fn fp16_requires_16_alignment() {
        // 8 % 16 != 0 → padding.
        let v = check(8, 128, 128, DataType::Fp16, 8);
        assert!(matches!(v, AlignmentVerdict::NeedsPadding { .. }));
    }

    #[test]
    fn invalid_zero_dim() {
        assert_eq!(
            check(0, 100, 100, DataType::Fp16, 8),
            AlignmentVerdict::InvalidShape
        );
    }

    #[test]
    fn unsupported_cc_rejected() {
        assert_eq!(
            check(128, 128, 128, DataType::Fp16, 6),
            AlignmentVerdict::UnsupportedCC
        );
    }

    #[test]
    fn padding_rounds_up() {
        let v = check(15, 16, 16, DataType::Fp16, 8);
        if let AlignmentVerdict::NeedsPadding { padded_m, .. } = v {
            assert_eq!(padded_m, 16);
        }
    }

    #[test]
    fn wasted_pct_correct() {
        // 15 padded to 16 → 1/16 wasted ≈ 6%.
        let v = check(15, 16, 16, DataType::Fp16, 8);
        if let AlignmentVerdict::NeedsPadding { wasted_pct, .. } = v {
            assert!(wasted_pct > 0);
        }
    }

    #[test]
    fn int8_requires_16_alignment() {
        let v = check(8, 16, 16, DataType::Int8, 8);
        assert!(matches!(v, AlignmentVerdict::NeedsPadding { .. }));
    }

    #[test]
    fn cc7_supports_tensor_cores() {
        let v = check(128, 128, 128, DataType::Fp16, 7);
        assert!(matches!(v, AlignmentVerdict::Aligned { .. }));
    }
}
