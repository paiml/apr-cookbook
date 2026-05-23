//! # apr export --target-dtype — Per-Format Compatibility Validator
//!
//! Each export target supports a different dtype set: ONNX accepts
//! FP32/FP16/BF16/Int8 but not Int4; GGUF accepts Int4/Int8/FP16; HF
//! SafeTensors accepts FP32/FP16/BF16 but not quantized integers.
//! This recipe builds the format × dtype compatibility matrix.
//!
//! Demonstrates the **EXP.4** recipe for PMAT-117 (apr export coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender EXP-001 + ONNX/GGUF/SafeTensors specs
//!
//! Run with: cargo run --example cli_export_target_dtype_validator
//!
//! Added by PMAT-117 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    Onnx,
    Gguf,
    SafeTensors,
    Apr,
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
pub enum CompatVerdict {
    Ok,
    Unsupported { format: ExportFormat, dtype: Dtype },
}

pub fn check(format: ExportFormat, dtype: Dtype) -> CompatVerdict {
    let supported = matches!(
        (format, dtype),
        (
            ExportFormat::Onnx,
            Dtype::Fp32 | Dtype::Fp16 | Dtype::Bf16 | Dtype::Int8
        ) | (ExportFormat::Gguf, Dtype::Fp16 | Dtype::Int8 | Dtype::Int4)
            | (
                ExportFormat::SafeTensors,
                Dtype::Fp32 | Dtype::Fp16 | Dtype::Bf16
            )
            | (ExportFormat::Apr, _)
    );
    if supported {
        CompatVerdict::Ok
    } else {
        CompatVerdict::Unsupported { format, dtype }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_export_target_dtype_validator")?;

    let cases = [
        (ExportFormat::Onnx, Dtype::Fp16),
        (ExportFormat::Onnx, Dtype::Int4),
        (ExportFormat::Gguf, Dtype::Int4),
        (ExportFormat::Gguf, Dtype::Bf16),
        (ExportFormat::SafeTensors, Dtype::Int8),
        (ExportFormat::Apr, Dtype::Int4),
    ];
    for (f, d) in cases {
        println!("{f:?} + {d:?}  →  {:?}", check(f, d));
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
    fn onnx_supports_fp_and_int8() {
        for d in [Dtype::Fp32, Dtype::Fp16, Dtype::Bf16, Dtype::Int8] {
            assert_eq!(check(ExportFormat::Onnx, d), CompatVerdict::Ok);
        }
    }

    #[test]
    fn onnx_rejects_int4() {
        let v = check(ExportFormat::Onnx, Dtype::Int4);
        assert!(matches!(v, CompatVerdict::Unsupported { .. }));
    }

    #[test]
    fn gguf_supports_quantized_and_fp16() {
        for d in [Dtype::Fp16, Dtype::Int8, Dtype::Int4] {
            assert_eq!(check(ExportFormat::Gguf, d), CompatVerdict::Ok);
        }
    }

    #[test]
    fn gguf_rejects_fp32_and_bf16() {
        // GGUF doesn't ship full FP32 weights; BF16 not in standard.
        for d in [Dtype::Fp32, Dtype::Bf16] {
            assert!(matches!(
                check(ExportFormat::Gguf, d),
                CompatVerdict::Unsupported { .. }
            ));
        }
    }

    #[test]
    fn safetensors_supports_fp_only() {
        for d in [Dtype::Fp32, Dtype::Fp16, Dtype::Bf16] {
            assert_eq!(check(ExportFormat::SafeTensors, d), CompatVerdict::Ok);
        }
    }

    #[test]
    fn safetensors_rejects_int4_and_int8() {
        for d in [Dtype::Int8, Dtype::Int4] {
            assert!(matches!(
                check(ExportFormat::SafeTensors, d),
                CompatVerdict::Unsupported { .. }
            ));
        }
    }

    #[test]
    fn apr_accepts_all_dtypes() {
        // APR is the union format — accepts everything.
        for d in [
            Dtype::Fp32,
            Dtype::Fp16,
            Dtype::Bf16,
            Dtype::Int8,
            Dtype::Int4,
        ] {
            assert_eq!(check(ExportFormat::Apr, d), CompatVerdict::Ok);
        }
    }

    #[test]
    fn unsupported_carries_format_and_dtype() {
        let v = check(ExportFormat::Onnx, Dtype::Int4);
        if let CompatVerdict::Unsupported { format, dtype } = v {
            assert_eq!(format, ExportFormat::Onnx);
            assert_eq!(dtype, Dtype::Int4);
        }
    }
}
