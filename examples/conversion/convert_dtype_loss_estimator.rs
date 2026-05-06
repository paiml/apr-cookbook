//! # Conversion Dtype-Conversion Loss Estimator
//!
//! Estimate expected RMSE / max-abs-error when converting between
//! dtypes. FP32 → FP16: ε ≈ 1e-3 (mantissa truncation). FP32 → BF16:
//! ε ≈ 1e-2 (smaller mantissa than FP16). FP16 → Int8: ε ≈ scale_step
//! / 2. This recipe builds the per-pair estimator.
//!
//! Demonstrates the **CONV.7** recipe for PMAT-127 (conversion coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Goldberg (1991). What Every Computer Scientist Should Know About FP.
//!
//! Run with: cargo run --example convert_dtype_loss_estimator
//!
//! Added by PMAT-127 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Fp32,
    Fp16,
    Bf16,
    Int8,
    Int4,
}

#[derive(Debug, PartialEq)]
pub enum LossVerdict {
    Lossless,
    BoundedRelativeError { rms_estimate: f64 },
    LossyQuantization { rms_estimate: f64 },
    Unsupported,
}

pub fn estimate(source: Dtype, target: Dtype, value_range: f64) -> LossVerdict {
    if source == target {
        return LossVerdict::Lossless;
    }
    if !value_range.is_finite() || value_range <= 0.0 {
        return LossVerdict::Unsupported;
    }
    match (source, target) {
        (Dtype::Fp16 | Dtype::Bf16, Dtype::Fp32)
        | (Dtype::Int4, Dtype::Int8 | Dtype::Bf16 | Dtype::Fp16 | Dtype::Fp32)
        | (Dtype::Int8, Dtype::Bf16 | Dtype::Fp16 | Dtype::Fp32) => LossVerdict::Lossless,
        (Dtype::Fp32, Dtype::Fp16) => LossVerdict::BoundedRelativeError {
            rms_estimate: value_range * 1e-3,
        },
        (Dtype::Fp32 | Dtype::Fp16, Dtype::Bf16) => LossVerdict::BoundedRelativeError {
            rms_estimate: value_range * 1e-2,
        },
        (Dtype::Bf16, Dtype::Fp16) => LossVerdict::BoundedRelativeError {
            rms_estimate: value_range * 5e-3,
        },
        (_, Dtype::Int8) => LossVerdict::LossyQuantization {
            rms_estimate: value_range / 256.0 / 2.0,
        },
        (_, Dtype::Int4) => LossVerdict::LossyQuantization {
            rms_estimate: value_range / 16.0 / 2.0,
        },
        // All same-type pairs already handled by the early return above.
        _ => LossVerdict::Lossless,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("convert_dtype_loss_estimator")?;

    let cases = [
        (Dtype::Fp32, Dtype::Fp16),
        (Dtype::Fp32, Dtype::Bf16),
        (Dtype::Fp16, Dtype::Int8),
        (Dtype::Fp32, Dtype::Int4),
        (Dtype::Fp16, Dtype::Fp32),
        (Dtype::Bf16, Dtype::Bf16),
    ];
    for (s, t) in cases {
        println!("{s:?} → {t:?}  =  {:?}", estimate(s, t, 1.0));
    }
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
    fn identity_lossless() {
        assert_eq!(
            estimate(Dtype::Fp32, Dtype::Fp32, 1.0),
            LossVerdict::Lossless
        );
    }

    #[test]
    fn fp16_to_fp32_lossless_widening() {
        assert_eq!(
            estimate(Dtype::Fp16, Dtype::Fp32, 1.0),
            LossVerdict::Lossless
        );
    }

    #[test]
    fn int_widening_lossless() {
        assert_eq!(
            estimate(Dtype::Int4, Dtype::Int8, 1.0),
            LossVerdict::Lossless
        );
        assert_eq!(
            estimate(Dtype::Int8, Dtype::Fp32, 1.0),
            LossVerdict::Lossless
        );
    }

    #[test]
    fn fp32_to_fp16_bounded_error() {
        let v = estimate(Dtype::Fp32, Dtype::Fp16, 1.0);
        if let LossVerdict::BoundedRelativeError { rms_estimate } = v {
            assert!((rms_estimate - 1e-3).abs() < 1e-12);
        }
    }

    #[test]
    fn fp32_to_bf16_larger_error_than_fp16() {
        let to_fp16 = estimate(Dtype::Fp32, Dtype::Fp16, 1.0);
        let to_bf16 = estimate(Dtype::Fp32, Dtype::Bf16, 1.0);
        if let (
            LossVerdict::BoundedRelativeError { rms_estimate: a },
            LossVerdict::BoundedRelativeError { rms_estimate: b },
        ) = (to_fp16, to_bf16)
        {
            assert!(b > a);
        }
    }

    #[test]
    fn fp_to_int8_lossy_quantization() {
        let v = estimate(Dtype::Fp16, Dtype::Int8, 1.0);
        assert!(matches!(v, LossVerdict::LossyQuantization { .. }));
    }

    #[test]
    fn fp_to_int4_more_lossy_than_int8() {
        let int8 = estimate(Dtype::Fp32, Dtype::Int8, 1.0);
        let int4 = estimate(Dtype::Fp32, Dtype::Int4, 1.0);
        if let (
            LossVerdict::LossyQuantization { rms_estimate: a },
            LossVerdict::LossyQuantization { rms_estimate: b },
        ) = (int8, int4)
        {
            assert!(b > a);
        }
    }

    #[test]
    fn invalid_range_unsupported() {
        assert_eq!(
            estimate(Dtype::Fp32, Dtype::Fp16, 0.0),
            LossVerdict::Unsupported
        );
        assert_eq!(
            estimate(Dtype::Fp32, Dtype::Fp16, -1.0),
            LossVerdict::Unsupported
        );
    }

    #[test]
    fn nan_range_unsupported() {
        assert_eq!(
            estimate(Dtype::Fp32, Dtype::Fp16, f64::NAN),
            LossVerdict::Unsupported
        );
    }

    #[test]
    fn loss_scales_with_range() {
        let small = estimate(Dtype::Fp32, Dtype::Fp16, 1.0);
        let big = estimate(Dtype::Fp32, Dtype::Fp16, 100.0);
        if let (
            LossVerdict::BoundedRelativeError { rms_estimate: s },
            LossVerdict::BoundedRelativeError { rms_estimate: b },
        ) = (small, big)
        {
            assert!((b / s - 100.0).abs() < 1e-9);
        }
    }
}
