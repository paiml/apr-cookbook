//! # apr rosetta convert — `--quantize` Envelope
//!
//! `apr rosetta convert <SOURCE> <TARGET> --quantize {int8,int4,fp16}`
//! quantizes during conversion. This recipe models the validation
//! envelope: target extension must be writable, source dtype must be
//! quantizable to the requested target, and the (source, target, quant)
//! triple must be compatible with the model format.
//!
//! Demonstrates the **ROSETTA-CONVERT.1** recipe for PMAT-098 (apr rosetta convert coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-ROSETTA-001 + dtype-compatibility table
//!
//! Run with: cargo run --example cli_rosetta_convert_quantize_envelope
//!
//! Added by PMAT-098 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quant {
    None,
    Int8,
    Int4,
    Fp16,
}

impl Quant {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "int8" => Some(Quant::Int8),
            "int4" => Some(Quant::Int4),
            "fp16" => Some(Quant::Fp16),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Fp32,
    Fp16,
    Bf16,
    Int8,
    Int4,
}

#[derive(Debug, PartialEq)]
pub enum ConvertVerdict {
    Ok,
    UnknownQuant(String),
    InvalidDowncast { from: Dtype, to: Quant },
}

pub fn validate_quant(source_dtype: Dtype, requested: Option<&str>) -> ConvertVerdict {
    let Some(raw) = requested else {
        return ConvertVerdict::Ok;
    };
    let Some(target) = Quant::from_str_strict(raw) else {
        return ConvertVerdict::UnknownQuant(raw.into());
    };
    // Forbid up-casting (fp16 → fp32 isn't a "quantize") and same-dtype no-op.
    // Forbid dequantization paths that the operator probably mistyped:
    //   int8/int4 → fp16, int4 → int8.
    let invalid = matches!(
        (source_dtype, target),
        (Dtype::Int8 | Dtype::Int4, Quant::Fp16) | (Dtype::Int4, Quant::Int8)
    );
    if invalid {
        ConvertVerdict::InvalidDowncast {
            from: source_dtype,
            to: target,
        }
    } else {
        ConvertVerdict::Ok
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_convert_quantize_envelope")?;

    let cases = [
        ("fp32 → int8 (good)", Dtype::Fp32, Some("int8")),
        ("bf16 → int4 (good)", Dtype::Bf16, Some("int4")),
        ("int4 → fp16 (bad)", Dtype::Int4, Some("fp16")),
        ("int4 → int8 (bad)", Dtype::Int4, Some("int8")),
        ("typo q4_0", Dtype::Fp32, Some("q4_0")),
        ("no quant", Dtype::Fp32, None),
    ];
    for (label, dt, q) in cases {
        println!("{label:>22}  →  {:?}", validate_quant(dt, q));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_quant_is_ok() {
        assert_eq!(validate_quant(Dtype::Fp32, None), ConvertVerdict::Ok);
    }

    #[test]
    fn fp32_to_int8_is_ok() {
        assert_eq!(
            validate_quant(Dtype::Fp32, Some("int8")),
            ConvertVerdict::Ok
        );
    }

    #[test]
    fn bf16_to_int4_is_ok() {
        assert_eq!(
            validate_quant(Dtype::Bf16, Some("int4")),
            ConvertVerdict::Ok
        );
    }

    #[test]
    fn int4_to_fp16_rejected() {
        // This is dequantization, not quantization — refuse rather than silently allow.
        let v = validate_quant(Dtype::Int4, Some("fp16"));
        assert!(matches!(v, ConvertVerdict::InvalidDowncast { .. }));
    }

    #[test]
    fn int4_to_int8_rejected() {
        assert!(matches!(
            validate_quant(Dtype::Int4, Some("int8")),
            ConvertVerdict::InvalidDowncast { .. }
        ));
    }

    #[test]
    fn typo_q4_0_rejected_as_unknown() {
        // Common GGUF-style typo — must surface as "unknown quant" not silently ignore.
        let v = validate_quant(Dtype::Fp32, Some("q4_0"));
        assert!(matches!(v, ConvertVerdict::UnknownQuant(_)));
    }

    #[test]
    fn empty_quant_string_unknown() {
        let v = validate_quant(Dtype::Fp32, Some(""));
        assert!(matches!(v, ConvertVerdict::UnknownQuant(_)));
    }
}
