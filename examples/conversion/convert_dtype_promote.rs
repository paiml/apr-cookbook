//! # Conversion Dtype Promotion Table
//!
//! Result dtype of an arithmetic op given two input dtypes:
//!   f16 + bf16 → f32 (need wider exponent + mantissa)
//!   f32 + i32 → f32 (float wins)
//!   i8 + i32 → i32 (wider int wins)
//!
//! Demonstrates the **CONV.20** recipe for PMAT-153 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NumPy promote_types rules.
//!
//! Run with: cargo run --example convert_dtype_promote
//!
//! Added by PMAT-153 (catalog 1000→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    I8,
    I16,
    I32,
    I64,
    U8,
    F16,
    Bf16,
    F32,
    F64,
}

#[derive(Debug, PartialEq)]
pub enum PromoteVerdict {
    Ok(Dtype),
}

pub fn promote(a: Dtype, b: Dtype) -> PromoteVerdict {
    if a == b {
        return PromoteVerdict::Ok(a);
    }
    let result = match (a, b) {
        // Float promotion (anything with float → float).
        (Dtype::F64, _) | (_, Dtype::F64) => Dtype::F64,
        (Dtype::F32, Dtype::F16 | Dtype::Bf16) | (Dtype::F16 | Dtype::Bf16, Dtype::F32) => {
            Dtype::F32
        }
        (Dtype::F16, Dtype::Bf16) | (Dtype::Bf16, Dtype::F16) => Dtype::F32,
        (Dtype::F32, _) | (_, Dtype::F32) => Dtype::F32,
        (Dtype::F16, _) | (_, Dtype::F16) => Dtype::F16,
        (Dtype::Bf16, _) | (_, Dtype::Bf16) => Dtype::Bf16,
        // Int promotion (wider wins).
        (Dtype::I64, _) | (_, Dtype::I64) => Dtype::I64,
        (Dtype::I32, _) | (_, Dtype::I32) => Dtype::I32,
        (Dtype::I16, _) | (_, Dtype::I16) => Dtype::I16,
        (Dtype::I8, Dtype::U8) | (Dtype::U8, Dtype::I8) => Dtype::I16,
        (Dtype::I8, _) | (_, Dtype::I8) => Dtype::I8,
        (Dtype::U8, _) | (_, Dtype::U8) => Dtype::U8,
        (Dtype::Bool, Dtype::Bool) => Dtype::Bool,
    };
    PromoteVerdict::Ok(result)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("convert_dtype_promote")?;

    println!("f16 + bf16: {:?}", promote(Dtype::F16, Dtype::Bf16));
    println!("f32 + i32: {:?}", promote(Dtype::F32, Dtype::I32));
    println!("i8 + i32: {:?}", promote(Dtype::I8, Dtype::I32));
    println!("i8 + u8: {:?}", promote(Dtype::I8, Dtype::U8));
    println!("bool + bool: {:?}", promote(Dtype::Bool, Dtype::Bool));
    println!("f64 + i8: {:?}", promote(Dtype::F64, Dtype::I8));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promoter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn same_dtype_unchanged() {
        for t in [Dtype::I32, Dtype::F32, Dtype::Bool] {
            assert_eq!(promote(t, t), PromoteVerdict::Ok(t));
        }
    }

    #[test]
    fn f16_bf16_promote_to_f32() {
        assert_eq!(
            promote(Dtype::F16, Dtype::Bf16),
            PromoteVerdict::Ok(Dtype::F32)
        );
        assert_eq!(
            promote(Dtype::Bf16, Dtype::F16),
            PromoteVerdict::Ok(Dtype::F32)
        );
    }

    #[test]
    fn f32_int_promote_to_f32() {
        assert_eq!(
            promote(Dtype::F32, Dtype::I32),
            PromoteVerdict::Ok(Dtype::F32)
        );
    }

    #[test]
    fn f64_anything_f64() {
        assert_eq!(
            promote(Dtype::F64, Dtype::I8),
            PromoteVerdict::Ok(Dtype::F64)
        );
        assert_eq!(
            promote(Dtype::F64, Dtype::F32),
            PromoteVerdict::Ok(Dtype::F64)
        );
    }

    #[test]
    fn wider_int_wins() {
        assert_eq!(
            promote(Dtype::I8, Dtype::I32),
            PromoteVerdict::Ok(Dtype::I32)
        );
        assert_eq!(
            promote(Dtype::I64, Dtype::I8),
            PromoteVerdict::Ok(Dtype::I64)
        );
    }

    #[test]
    fn signed_unsigned_promote_to_wider() {
        // i8 + u8 → i16 (need to fit signed -128 and unsigned 255).
        assert_eq!(
            promote(Dtype::I8, Dtype::U8),
            PromoteVerdict::Ok(Dtype::I16)
        );
    }

    #[test]
    fn bool_with_int_promotes() {
        // Bool + int → int.
        assert_eq!(
            promote(Dtype::Bool, Dtype::I32),
            PromoteVerdict::Ok(Dtype::I32)
        );
    }

    #[test]
    fn symmetric() {
        // promote(a, b) == promote(b, a).
        let pairs = [
            (Dtype::F16, Dtype::F32),
            (Dtype::I8, Dtype::I32),
            (Dtype::Bool, Dtype::F64),
        ];
        for (a, b) in pairs {
            assert_eq!(promote(a, b), promote(b, a));
        }
    }

    #[test]
    fn f32_bf16_stays_f32() {
        assert_eq!(
            promote(Dtype::F32, Dtype::Bf16),
            PromoteVerdict::Ok(Dtype::F32)
        );
    }

    #[test]
    fn f16_int_promotes_to_f16() {
        // f16 + i8 → f16 (no wider needed).
        assert_eq!(
            promote(Dtype::F16, Dtype::I8),
            PromoteVerdict::Ok(Dtype::F16)
        );
    }

    #[test]
    fn deterministic() {
        let a = promote(Dtype::F16, Dtype::I32);
        let b = promote(Dtype::F16, Dtype::I32);
        assert_eq!(a, b);
    }
}
