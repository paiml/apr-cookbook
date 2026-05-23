#![allow(unused_imports)]
//! # Cross-Format Conversion via Rosetta
//!
//! **CLI equivalent:** `apr rosetta convert --from safetensors --to apr`
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/apr-format-roundtrip-v1.yaml
//!
//! Demonstrates cross-format conversion using an intermediate representation.
//! The Rosetta module finds the optimal conversion path between any two
//! supported formats and executes the transformation step by step.
//!
//! ## Sections
//! 1. Format registry — supported formats and their capabilities
//! 2. Path finding — discover direct and transitive conversion paths
//! 3. Conversion execution — apply each step in the conversion path
//! 4. Verification — validate the output matches expectations
//!
//!
//! ## Format Variants
//! ```bash
//! apr convert model.apr          # APR native format
//! apr convert model.gguf         # GGUF (llama.cpp compatible)
//! apr convert model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language Processing*. EMNLP. DOI: 10.18653/v1/2020.emnlp-demos.6

use apr_cookbook::prelude::*;
use std::collections::HashMap;
use std::fmt;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    let ctx = RecipeContext::new("format_rosetta_convert")?;

    let registry = FormatRegistry::new();

    print_format_registry(&registry);
    print_conversion_paths(&registry);
    run_direct_conversion(&registry);
    run_transitive_conversion(&registry);

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_format_is_identity() {
        let reg = FormatRegistry::new();
        let path = reg.find_path(Format::Apr, Format::Apr).unwrap();
        assert!(path.is_identity());
        assert_eq!(path.steps.len(), 0);
    }

    #[test]
    fn test_direct_conversion_safetensors_to_apr() {
        let reg = FormatRegistry::new();
        let path = reg.find_path(Format::SafeTensors, Format::Apr).unwrap();
        assert!(path.is_direct());
        assert_eq!(path.steps.len(), 1);
    }

    #[test]
    fn test_transitive_conversion_safetensors_to_gguf() {
        let reg = FormatRegistry::new();
        let path = reg.find_path(Format::SafeTensors, Format::Gguf).unwrap();
        assert_eq!(path.steps.len(), 2); // ST → APR → GGUF
        assert_eq!(path.steps[0].from, Format::SafeTensors);
        assert_eq!(path.steps[0].to, Format::Apr);
        assert_eq!(path.steps[1].from, Format::Apr);
        assert_eq!(path.steps[1].to, Format::Gguf);
    }

    #[test]
    fn test_transitive_conversion_gguf_to_onnx() {
        let reg = FormatRegistry::new();
        let path = reg.find_path(Format::Gguf, Format::Onnx).unwrap();
        assert_eq!(path.steps.len(), 2); // GGUF → APR → ONNX
    }

    #[test]
    fn test_lossy_path_detected() {
        let reg = FormatRegistry::new();
        let path = reg.find_path(Format::Apr, Format::Onnx).unwrap();
        assert!(path.is_lossy());
    }

    #[test]
    fn test_lossless_path() {
        let reg = FormatRegistry::new();
        let path = reg.find_path(Format::SafeTensors, Format::Apr).unwrap();
        assert!(!path.is_lossy());
    }

    #[test]
    fn test_execute_conversion() {
        let reg = FormatRegistry::new();
        let data = ModelData {
            format: Format::SafeTensors,
            payload: generate_model_payload(42, 1024),
            tensor_count: 2,
        };
        let path = reg.find_path(Format::SafeTensors, Format::Apr).unwrap();
        let result = convert(data, &path);
        assert_eq!(result.format, Format::Apr);
        assert_eq!(result.tensor_count, 2);
        assert!(&result.payload[0..4] == b"APR2");
    }

    #[test]
    fn test_identity_conversion_preserves_data() {
        let reg = FormatRegistry::new();
        let payload = generate_model_payload(42, 512);
        let data = ModelData {
            format: Format::Apr,
            payload: payload.clone(),
            tensor_count: 1,
        };
        let path = reg.find_path(Format::Apr, Format::Apr).unwrap();
        let result = convert(data, &path);
        assert_eq!(result.payload, payload);
    }

    #[test]
    fn test_all_formats_reachable_from_apr() {
        let reg = FormatRegistry::new();
        for fmt in [Format::SafeTensors, Format::Gguf, Format::Onnx] {
            assert!(
                reg.find_path(Format::Apr, fmt).is_some(),
                "APR → {fmt} should have a path"
            );
        }
    }

    #[test]
    fn test_all_formats_can_reach_apr() {
        let reg = FormatRegistry::new();
        for fmt in [Format::SafeTensors, Format::Gguf, Format::Onnx] {
            assert!(
                reg.find_path(fmt, Format::Apr).is_some(),
                "{fmt} → APR should have a path"
            );
        }
    }

    #[test]
    fn test_format_display() {
        assert_eq!(format!("{}", Format::Apr), "APR");
        assert_eq!(format!("{}", Format::SafeTensors), "SafeTensors");
        assert_eq!(format!("{}", Format::Gguf), "GGUF");
        assert_eq!(format!("{}", Format::Onnx), "ONNX");
    }
}
