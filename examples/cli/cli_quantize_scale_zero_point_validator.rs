//! # apr quantize — Scale × Zero-Point Validator
//!
//! Asymmetric per-tensor quant: q = clamp(round(x / scale) + zero_point, qmin, qmax).
//! Constraints: `scale > 0`, `qmin ≤ zero_point ≤ qmax`. For Int8 symmetric,
//! zero_point = 0 enforced. This recipe codifies the validator.
//!
//! Demonstrates the **QUANT.6** recipe for PMAT-112 (apr quantize coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender QUANT-001 + Jacob et al. 2018 (Quantization)
//!
//! Run with: cargo run --example cli_quantize_scale_zero_point_validator
//!
//! Added by PMAT-112 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum QuantParams {
    Int8Symmetric,
    Int8Asymmetric,
    Int4Asymmetric,
}

impl QuantParams {
    pub fn qmin(self) -> i32 {
        match self {
            QuantParams::Int8Symmetric => -127,
            QuantParams::Int8Asymmetric => -128,
            QuantParams::Int4Asymmetric => -8,
        }
    }
    pub fn qmax(self) -> i32 {
        match self {
            QuantParams::Int8Symmetric => 127,
            QuantParams::Int8Asymmetric => 127,
            QuantParams::Int4Asymmetric => 7,
        }
    }
    pub fn requires_zero_point_zero(self) -> bool {
        matches!(self, QuantParams::Int8Symmetric)
    }
}

#[derive(Debug, PartialEq)]
pub enum ValidationVerdict {
    Ok,
    NonPositiveScale,
    ZeroPointOutOfRange { qmin: i32, qmax: i32 },
    SymmetricRequiresZeroPointZero,
    NonFiniteScale,
}

pub fn validate(scale: f64, zero_point: i32, params: QuantParams) -> ValidationVerdict {
    if !scale.is_finite() {
        return ValidationVerdict::NonFiniteScale;
    }
    if scale <= 0.0 {
        return ValidationVerdict::NonPositiveScale;
    }
    if params.requires_zero_point_zero() && zero_point != 0 {
        return ValidationVerdict::SymmetricRequiresZeroPointZero;
    }
    if zero_point < params.qmin() || zero_point > params.qmax() {
        return ValidationVerdict::ZeroPointOutOfRange {
            qmin: params.qmin(),
            qmax: params.qmax(),
        };
    }
    ValidationVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_quantize_scale_zero_point_validator")?;

    let cases = [
        ("ok int8 sym", 0.012, 0, QuantParams::Int8Symmetric),
        ("ok int8 asym", 0.05, 64, QuantParams::Int8Asymmetric),
        ("scale 0", 0.0, 0, QuantParams::Int8Symmetric),
        ("zp out of range", 0.05, 200, QuantParams::Int8Asymmetric),
        ("sym non-zero zp", 0.05, 5, QuantParams::Int8Symmetric),
        ("scale NaN", f64::NAN, 0, QuantParams::Int8Symmetric),
    ];
    for (label, s, zp, p) in cases {
        println!("{label:>20}  →  {:?}", validate(s, zp, p));
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
    fn typical_int8_sym_passes() {
        assert_eq!(
            validate(0.05, 0, QuantParams::Int8Symmetric),
            ValidationVerdict::Ok
        );
    }

    #[test]
    fn typical_int8_asym_passes() {
        assert_eq!(
            validate(0.05, 64, QuantParams::Int8Asymmetric),
            ValidationVerdict::Ok
        );
    }

    #[test]
    fn zero_scale_rejected() {
        assert_eq!(
            validate(0.0, 0, QuantParams::Int8Symmetric),
            ValidationVerdict::NonPositiveScale
        );
    }

    #[test]
    fn negative_scale_rejected() {
        assert_eq!(
            validate(-0.01, 0, QuantParams::Int8Symmetric),
            ValidationVerdict::NonPositiveScale
        );
    }

    #[test]
    fn nan_scale_rejected() {
        assert_eq!(
            validate(f64::NAN, 0, QuantParams::Int8Symmetric),
            ValidationVerdict::NonFiniteScale
        );
    }

    #[test]
    fn symmetric_non_zero_zp_rejected() {
        assert_eq!(
            validate(0.05, 5, QuantParams::Int8Symmetric),
            ValidationVerdict::SymmetricRequiresZeroPointZero
        );
    }

    #[test]
    fn asym_zp_out_of_range_rejected() {
        let v = validate(0.05, 200, QuantParams::Int8Asymmetric);
        assert!(matches!(v, ValidationVerdict::ZeroPointOutOfRange { .. }));
    }

    #[test]
    fn int4_qmin_qmax_correct() {
        // Int4 asymmetric: [-8, 7].
        assert_eq!(QuantParams::Int4Asymmetric.qmin(), -8);
        assert_eq!(QuantParams::Int4Asymmetric.qmax(), 7);
        assert_eq!(
            validate(0.5, 0, QuantParams::Int4Asymmetric),
            ValidationVerdict::Ok
        );
        assert!(matches!(
            validate(0.5, 8, QuantParams::Int4Asymmetric),
            ValidationVerdict::ZeroPointOutOfRange { .. }
        ));
    }
}
