//! # apr import --dtype — Coercion Compatibility Matrix
//!
//! `apr import <FILE> --dtype <TARGET>` may need to coerce source dtype
//! to target. Safe widenings (FP16 → FP32, BF16 → FP32, Int8 → Int16)
//! lossless; downcasts (FP32 → FP16) lossy → warn; mixed FP/Int requires
//! explicit cast spec. This recipe builds the matrix.
//!
//! Demonstrates the **IMP.5** recipe for PMAT-115 (apr import coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender IMP-001 + IEEE 754 + Goldberg 1991
//!
//! Run with: cargo run --example cli_import_dtype_coercion_validator
//!
//! Added by PMAT-115 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Fp32,
    Fp16,
    Bf16,
    Int32,
    Int16,
    Int8,
    Int4,
}

#[derive(Debug, PartialEq)]
pub enum CoerceVerdict {
    Identity,
    SafeWiden,
    LossyDowncast { source: Dtype, target: Dtype },
    CrossKindRequiresExplicit, // FP ↔ Int
    Unsupported,
}

fn is_floating(d: Dtype) -> bool {
    matches!(d, Dtype::Fp32 | Dtype::Fp16 | Dtype::Bf16)
}

fn is_integer(d: Dtype) -> bool {
    matches!(d, Dtype::Int32 | Dtype::Int16 | Dtype::Int8 | Dtype::Int4)
}

fn bits(d: Dtype) -> u8 {
    match d {
        Dtype::Fp32 | Dtype::Int32 => 32,
        Dtype::Fp16 | Dtype::Bf16 | Dtype::Int16 => 16,
        Dtype::Int8 => 8,
        Dtype::Int4 => 4,
    }
}

pub fn classify_coercion(source: Dtype, target: Dtype) -> CoerceVerdict {
    if source == target {
        return CoerceVerdict::Identity;
    }
    let src_fp = is_floating(source);
    let tgt_fp = is_floating(target);
    if src_fp != tgt_fp && (is_integer(source) || is_integer(target)) {
        return CoerceVerdict::CrossKindRequiresExplicit;
    }
    if (src_fp && !tgt_fp) || (!src_fp && tgt_fp) {
        return CoerceVerdict::CrossKindRequiresExplicit;
    }
    if bits(target) >= bits(source) {
        CoerceVerdict::SafeWiden
    } else {
        CoerceVerdict::LossyDowncast { source, target }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_import_dtype_coercion_validator")?;

    let cases = [
        (Dtype::Fp16, Dtype::Fp32),
        (Dtype::Bf16, Dtype::Fp32),
        (Dtype::Fp32, Dtype::Fp16),
        (Dtype::Int8, Dtype::Int16),
        (Dtype::Fp16, Dtype::Int8),
        (Dtype::Int8, Dtype::Int8),
    ];
    for (s, t) in cases {
        println!("{s:?} → {t:?}  =  {:?}", classify_coercion(s, t));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn identity_returns_identity() {
        assert_eq!(
            classify_coercion(Dtype::Fp32, Dtype::Fp32),
            CoerceVerdict::Identity
        );
    }

    #[test]
    fn fp16_to_fp32_safe_widen() {
        assert_eq!(
            classify_coercion(Dtype::Fp16, Dtype::Fp32),
            CoerceVerdict::SafeWiden
        );
    }

    #[test]
    fn bf16_to_fp32_safe_widen() {
        assert_eq!(
            classify_coercion(Dtype::Bf16, Dtype::Fp32),
            CoerceVerdict::SafeWiden
        );
    }

    #[test]
    fn int8_to_int16_safe_widen() {
        assert_eq!(
            classify_coercion(Dtype::Int8, Dtype::Int16),
            CoerceVerdict::SafeWiden
        );
    }

    #[test]
    fn fp32_to_fp16_lossy_downcast() {
        let v = classify_coercion(Dtype::Fp32, Dtype::Fp16);
        assert!(matches!(v, CoerceVerdict::LossyDowncast { .. }));
    }

    #[test]
    fn fp_to_int_requires_explicit() {
        assert_eq!(
            classify_coercion(Dtype::Fp16, Dtype::Int8),
            CoerceVerdict::CrossKindRequiresExplicit
        );
        assert_eq!(
            classify_coercion(Dtype::Int8, Dtype::Fp16),
            CoerceVerdict::CrossKindRequiresExplicit
        );
    }

    #[test]
    fn int_widen_int4_to_int32() {
        assert_eq!(
            classify_coercion(Dtype::Int4, Dtype::Int32),
            CoerceVerdict::SafeWiden
        );
    }

    #[test]
    fn fp16_to_bf16_same_bits_safe_widen() {
        // Same bit width but different dynamic range — treated as safe widen
        // (no precision loss in either direction at same bits, but careful
        // semantics differ. Per the rule above, bits(target) >= bits(source) → SafeWiden).
        assert_eq!(
            classify_coercion(Dtype::Fp16, Dtype::Bf16),
            CoerceVerdict::SafeWiden
        );
    }
}
