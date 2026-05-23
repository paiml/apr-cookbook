//! # apr export — Format Allowlist + Capability Matrix
//!
//! `apr export <FILE> --format <FMT>` accepts {safetensors, gguf, mlx,
//! onnx, openvino, coreml}. Each format has different capabilities (e.g.,
//! quantization support, lm_head sharing). This recipe vendors the
//! capability matrix and asserts the contract: unknown formats reject,
//! requested features must intersect with format capabilities.
//!
//! Demonstrates the **EXPORT.5** recipe for PMAT-099 (apr export coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-EXPORT-001
//!
//! Run with: cargo run --example cli_export_format_allowlist
//!
//! Added by PMAT-099 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    SafeTensors,
    Gguf,
    Mlx,
    Onnx,
    OpenVino,
    CoreMl,
}

impl Format {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "safetensors" => Some(Format::SafeTensors),
            "gguf" => Some(Format::Gguf),
            "mlx" => Some(Format::Mlx),
            "onnx" => Some(Format::Onnx),
            "openvino" => Some(Format::OpenVino),
            "coreml" => Some(Format::CoreMl),
            _ => None,
        }
    }

    pub fn supports_int4(self) -> bool {
        matches!(self, Format::Gguf | Format::OpenVino)
    }

    pub fn supports_lora_adapters(self) -> bool {
        matches!(self, Format::SafeTensors | Format::Gguf | Format::Mlx)
    }

    pub fn supports_dynamic_shape(self) -> bool {
        matches!(self, Format::Onnx | Format::OpenVino | Format::CoreMl)
    }
}

#[derive(Debug, PartialEq)]
pub enum ExportVerdict {
    Ok,
    UnknownFormat(String),
    Int4NotSupported(Format),
}

pub fn validate_export(format: &str, requested_int4: bool) -> ExportVerdict {
    let Some(f) = Format::from_str_strict(format) else {
        return ExportVerdict::UnknownFormat(format.into());
    };
    if requested_int4 && !f.supports_int4() {
        return ExportVerdict::Int4NotSupported(f);
    }
    ExportVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_export_format_allowlist")?;

    println!("Format            int4   lora   dyn-shape");
    for f in [
        Format::SafeTensors,
        Format::Gguf,
        Format::Mlx,
        Format::Onnx,
        Format::OpenVino,
        Format::CoreMl,
    ] {
        println!(
            "{:<14?}  {:>4}   {:>4}   {:>5}",
            f,
            f.supports_int4(),
            f.supports_lora_adapters(),
            f.supports_dynamic_shape()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowlist_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn known_formats_parse() {
        for f in ["safetensors", "gguf", "mlx", "onnx", "openvino", "coreml"] {
            assert!(Format::from_str_strict(f).is_some(), "missing format: {f}");
        }
    }

    #[test]
    fn unknown_format_returns_none() {
        assert!(Format::from_str_strict("torchscript").is_none());
        assert!(Format::from_str_strict("").is_none());
    }

    #[test]
    fn int4_supported_only_by_gguf_and_openvino() {
        assert!(Format::Gguf.supports_int4());
        assert!(Format::OpenVino.supports_int4());
        assert!(!Format::SafeTensors.supports_int4());
        assert!(!Format::Onnx.supports_int4());
    }

    #[test]
    fn int4_request_to_unsupported_format_rejected() {
        let v = validate_export("safetensors", true);
        assert!(matches!(v, ExportVerdict::Int4NotSupported(_)));
    }

    #[test]
    fn int4_request_to_supported_format_passes() {
        assert_eq!(validate_export("gguf", true), ExportVerdict::Ok);
    }

    #[test]
    fn no_int4_request_passes_for_any_format() {
        for f in ["safetensors", "gguf", "mlx", "onnx", "openvino", "coreml"] {
            assert_eq!(validate_export(f, false), ExportVerdict::Ok);
        }
    }

    #[test]
    fn lora_supported_only_by_native_formats() {
        // SafeTensors / GGUF / MLX understand LoRA adapters; runtime formats
        // (ONNX, OpenVINO, CoreML) require pre-merged weights.
        assert!(Format::SafeTensors.supports_lora_adapters());
        assert!(Format::Gguf.supports_lora_adapters());
        assert!(Format::Mlx.supports_lora_adapters());
        assert!(!Format::Onnx.supports_lora_adapters());
        assert!(!Format::CoreMl.supports_lora_adapters());
    }

    #[test]
    fn dynamic_shape_supported_only_by_runtime_formats() {
        assert!(Format::Onnx.supports_dynamic_shape());
        assert!(Format::OpenVino.supports_dynamic_shape());
        assert!(Format::CoreMl.supports_dynamic_shape());
        assert!(!Format::SafeTensors.supports_dynamic_shape());
        assert!(!Format::Gguf.supports_dynamic_shape());
    }
}
