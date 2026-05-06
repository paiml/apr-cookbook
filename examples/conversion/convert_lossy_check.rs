//! # Conversion Lossy Check
//!
//! Pre-conversion check: which target precision will preserve data?
//!   f32 → f16: lossy if any value > 65504 (max fp16) or precision-bit drift
//!   f32 → bf16: lossy if value mantissa bits > 7
//!   f32 → int8: lossy unless quantizable (bounded range)
//!
//! Returns Lossless / LossyButOk / TooLossy.
//!
//! Demonstrates the **CONV.16** recipe for PMAT-148 (conversion round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: IEEE 754 binary16 (fp16) range and precision spec.
//!
//! Run with: cargo run --example convert_lossy_check
//!
//! Added by PMAT-148 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    F32,
    Bf16,
    F16,
    Int8,
}

#[derive(Debug, PartialEq)]
pub enum LossyVerdict {
    Lossless,
    LossyButOk { reason: &'static str },
    TooLossy { reason: &'static str },
    EmptyTensor,
    Unsupported,
}

const FP16_MAX: f32 = 65504.0;

pub fn check(values: &[f32], target: Precision) -> LossyVerdict {
    if values.is_empty() {
        return LossyVerdict::EmptyTensor;
    }
    if target == Precision::F32 {
        return LossyVerdict::Lossless;
    }
    let max_abs = values.iter().fold(0.0_f32, |acc, &v| acc.max(v.abs()));
    match target {
        Precision::F32 => LossyVerdict::Lossless,
        Precision::F16 => {
            if !max_abs.is_finite() {
                return LossyVerdict::TooLossy {
                    reason: "non-finite values not representable in fp16",
                };
            }
            if max_abs > FP16_MAX {
                return LossyVerdict::TooLossy {
                    reason: "max value exceeds fp16 range (65504)",
                };
            }
            LossyVerdict::LossyButOk {
                reason: "fp16 has 10 mantissa bits vs fp32's 23",
            }
        }
        Precision::Bf16 => {
            // bf16 has same exp range as fp32 (no overflow risk) but only 7
            // mantissa bits.
            LossyVerdict::LossyButOk {
                reason: "bf16 keeps fp32 range; loses precision in mantissa",
            }
        }
        Precision::Int8 => {
            // Bounded range required; otherwise too lossy.
            if max_abs > 127.0 {
                return LossyVerdict::TooLossy {
                    reason: "max value out of int8 range; needs scale+zero_point",
                };
            }
            LossyVerdict::LossyButOk {
                reason: "8-bit int discretization rounds floats",
            }
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("convert_lossy_check")?;

    println!("f32 → f32: {:?}", check(&[1.0, 2.0], Precision::F32));
    println!(
        "f32 → f16 ok: {:?}",
        check(&[1.0, 2.0, -100.0], Precision::F16)
    );
    println!(
        "f32 → f16 too lossy: {:?}",
        check(&[1.0, 1e6], Precision::F16)
    );
    println!("f32 → bf16: {:?}", check(&[1.0, 1e6], Precision::Bf16));
    println!(
        "f32 → int8 ok: {:?}",
        check(&[-100.0, 50.0], Precision::Int8)
    );
    println!(
        "f32 → int8 too lossy: {:?}",
        check(&[1000.0], Precision::Int8)
    );
    println!("empty: {:?}", check(&[], Precision::F16));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn same_precision_lossless() {
        assert_eq!(check(&[1.0], Precision::F32), LossyVerdict::Lossless);
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(check(&[], Precision::F16), LossyVerdict::EmptyTensor);
    }

    #[test]
    fn fp16_within_range_lossy_but_ok() {
        let v = check(&[1.0, 100.0, -50.0], Precision::F16);
        assert!(matches!(v, LossyVerdict::LossyButOk { .. }));
    }

    #[test]
    fn fp16_overflow_too_lossy() {
        let v = check(&[1e6], Precision::F16);
        assert!(matches!(v, LossyVerdict::TooLossy { .. }));
    }

    #[test]
    fn fp16_infinity_too_lossy() {
        let v = check(&[f32::INFINITY], Precision::F16);
        assert!(matches!(v, LossyVerdict::TooLossy { .. }));
    }

    #[test]
    fn bf16_range_unchanged_lossy_but_ok() {
        let v = check(&[1e6], Precision::Bf16);
        assert!(matches!(v, LossyVerdict::LossyButOk { .. }));
    }

    #[test]
    fn int8_within_range_lossy_but_ok() {
        let v = check(&[-100.0, 50.0, 0.0], Precision::Int8);
        assert!(matches!(v, LossyVerdict::LossyButOk { .. }));
    }

    #[test]
    fn int8_out_of_range_too_lossy() {
        let v = check(&[1000.0], Precision::Int8);
        assert!(matches!(v, LossyVerdict::TooLossy { .. }));
    }

    #[test]
    fn fp16_max_boundary_value_ok() {
        let v = check(&[FP16_MAX], Precision::F16);
        assert!(matches!(v, LossyVerdict::LossyButOk { .. }));
    }

    #[test]
    fn just_above_fp16_max_too_lossy() {
        let v = check(&[FP16_MAX + 1.0], Precision::F16);
        assert!(matches!(v, LossyVerdict::TooLossy { .. }));
    }
}
