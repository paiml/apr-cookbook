//! # Conversion Dtype Widener (lossless promotion)
//!
//! Lossless dtype promotion paths:
//! - i8 → i16 → i32 → i64 (sign-extend)
//! - u8 → u16 → u32 → u64 (zero-extend)
//! - f16 → f32 → f64 (mantissa expand)
//! - bf16 → f32 (mantissa expand)
//!
//! Lossy or undefined paths must be rejected. This recipe builds the
//! widener policy.
//!
//! Demonstrates the **CONV.14** recipe for PMAT-136 (conversion round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: IEEE 754-2008 (binary16 → binary32 promotion).
//!
//! Run with: cargo run --example convert_dtype_widener
//!
//! Added by PMAT-136 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F16,
    Bf16,
    F32,
    F64,
}

#[derive(Debug, PartialEq)]
pub enum WidenVerdict {
    Lossless,
    NoOp,
    LossyOrUndefined { reason: &'static str },
}

pub fn check(from: Dtype, to: Dtype) -> WidenVerdict {
    if from == to {
        return WidenVerdict::NoOp;
    }
    let signed_int_path = matches!(
        (from, to),
        (Dtype::I8, Dtype::I16 | Dtype::I32 | Dtype::I64)
            | (Dtype::I16, Dtype::I32 | Dtype::I64)
            | (Dtype::I32, Dtype::I64)
    );
    let unsigned_int_path = matches!(
        (from, to),
        (Dtype::U8, Dtype::U16 | Dtype::U32 | Dtype::U64)
            | (Dtype::U16, Dtype::U32 | Dtype::U64)
            | (Dtype::U32, Dtype::U64)
    );
    let float_path = matches!(
        (from, to),
        (Dtype::F16 | Dtype::Bf16, Dtype::F32 | Dtype::F64) | (Dtype::F32, Dtype::F64)
    );
    if signed_int_path || unsigned_int_path || float_path {
        return WidenVerdict::Lossless;
    }
    let signed_to_unsigned = matches!(from, Dtype::I8 | Dtype::I16 | Dtype::I32 | Dtype::I64)
        && matches!(to, Dtype::U8 | Dtype::U16 | Dtype::U32 | Dtype::U64);
    if signed_to_unsigned {
        return WidenVerdict::LossyOrUndefined {
            reason: "negative values become MSB-encoded under unsigned",
        };
    }
    let int_to_float_lossy = matches!(
        (from, to),
        (
            Dtype::I32 | Dtype::U32 | Dtype::I64 | Dtype::U64,
            Dtype::F32
        ) | (Dtype::I64 | Dtype::U64, Dtype::F64)
    );
    if int_to_float_lossy {
        return WidenVerdict::LossyOrUndefined {
            reason: "mantissa cannot represent all integer magnitudes",
        };
    }
    WidenVerdict::LossyOrUndefined {
        reason: "narrowing or unsupported pair",
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("convert_dtype_widener")?;

    let pairs = [
        (Dtype::F16, Dtype::F32),
        (Dtype::I8, Dtype::I32),
        (Dtype::U8, Dtype::U64),
        (Dtype::F32, Dtype::F32),
        (Dtype::I32, Dtype::U32),
        (Dtype::F32, Dtype::F16),
        (Dtype::I64, Dtype::F32),
    ];
    for (from, to) in pairs {
        println!("{from:?} → {to:?}: {:?}", check(from, to));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widener_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn same_dtype_is_noop() {
        assert_eq!(check(Dtype::F32, Dtype::F32), WidenVerdict::NoOp);
        assert_eq!(check(Dtype::I64, Dtype::I64), WidenVerdict::NoOp);
    }

    #[test]
    fn f16_to_f32_lossless() {
        assert_eq!(check(Dtype::F16, Dtype::F32), WidenVerdict::Lossless);
    }

    #[test]
    fn bf16_to_f32_lossless() {
        assert_eq!(check(Dtype::Bf16, Dtype::F32), WidenVerdict::Lossless);
    }

    #[test]
    fn i8_to_i64_lossless() {
        assert_eq!(check(Dtype::I8, Dtype::I64), WidenVerdict::Lossless);
    }

    #[test]
    fn u8_to_u32_lossless() {
        assert_eq!(check(Dtype::U8, Dtype::U32), WidenVerdict::Lossless);
    }

    #[test]
    fn signed_to_unsigned_lossy() {
        let v = check(Dtype::I32, Dtype::U32);
        assert!(matches!(v, WidenVerdict::LossyOrUndefined { .. }));
    }

    #[test]
    fn narrowing_lossy() {
        let v = check(Dtype::F32, Dtype::F16);
        assert!(matches!(v, WidenVerdict::LossyOrUndefined { .. }));
        let v2 = check(Dtype::I64, Dtype::I32);
        assert!(matches!(v2, WidenVerdict::LossyOrUndefined { .. }));
    }

    #[test]
    fn i64_to_f32_lossy_due_to_mantissa() {
        let v = check(Dtype::I64, Dtype::F32);
        assert!(matches!(v, WidenVerdict::LossyOrUndefined { .. }));
    }

    #[test]
    fn i32_to_f32_lossy() {
        // i32 max = 2^31; f32 mantissa is 24 bits.
        let v = check(Dtype::I32, Dtype::F32);
        assert!(matches!(v, WidenVerdict::LossyOrUndefined { .. }));
    }

    #[test]
    fn f32_to_f64_lossless() {
        assert_eq!(check(Dtype::F32, Dtype::F64), WidenVerdict::Lossless);
    }

    #[test]
    fn unsigned_to_signed_not_a_widen_path() {
        // u32 → i32 is not lossless even at same width.
        let v = check(Dtype::U32, Dtype::I32);
        assert!(matches!(v, WidenVerdict::LossyOrUndefined { .. }));
    }
}
