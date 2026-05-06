//! # apr rosetta convert — Extension-Based Format Inference
//!
//! `apr rosetta convert <SOURCE> <TARGET>` infers the target format from
//! the file extension: `.apr` → APR v2, `.gguf` → GGUF, `.safetensors`
//! (or `.st`) → SafeTensors, `.onnx` → ONNX. This recipe tests the
//! resolver and asserts the contract: unknown extensions must NOT pick
//! a default — operator must specify explicitly.
//!
//! Demonstrates the **ROSETTA-CONVERT.2** recipe for PMAT-098 (apr rosetta convert coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-ROSETTA-001
//!
//! Run with: cargo run --example cli_rosetta_convert_extension_inference
//!
//! Added by PMAT-098 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetFormat {
    AprV2,
    Gguf,
    SafeTensors,
    Onnx,
}

pub fn infer_target_format(path: &str) -> Option<TargetFormat> {
    let lower = path.to_ascii_lowercase();
    let extension = lower.rsplit('.').next()?;
    match extension {
        "apr" => Some(TargetFormat::AprV2),
        "gguf" => Some(TargetFormat::Gguf),
        "safetensors" | "st" => Some(TargetFormat::SafeTensors),
        "onnx" => Some(TargetFormat::Onnx),
        _ => None,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_convert_extension_inference")?;

    let cases = [
        "model.apr",
        "model.gguf",
        "model.safetensors",
        "model.st",
        "model.onnx",
        "MODEL.APR",
        "model.bin",
        "model",
        "/tmp/qwen-coder-7b.apr",
    ];
    for c in cases {
        println!("{c:>30}  →  {:?}", infer_target_format(c));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn apr_extension_recognized() {
        assert_eq!(infer_target_format("model.apr"), Some(TargetFormat::AprV2));
    }

    #[test]
    fn gguf_extension_recognized() {
        assert_eq!(infer_target_format("model.gguf"), Some(TargetFormat::Gguf));
    }

    #[test]
    fn safetensors_canonical_and_short_alias() {
        assert_eq!(
            infer_target_format("model.safetensors"),
            Some(TargetFormat::SafeTensors)
        );
        assert_eq!(
            infer_target_format("model.st"),
            Some(TargetFormat::SafeTensors)
        );
    }

    #[test]
    fn case_insensitive() {
        // "MODEL.APR" must resolve the same as "model.apr".
        assert_eq!(infer_target_format("MODEL.APR"), Some(TargetFormat::AprV2));
    }

    #[test]
    fn unknown_extension_returns_none_not_default() {
        // Critical: don't pick a default for ".bin" (could be GGUF or pkl or
        // anything). Operator must specify.
        assert!(infer_target_format("model.bin").is_none());
        assert!(infer_target_format("model.pt").is_none());
        assert!(infer_target_format("model.h5").is_none());
    }

    #[test]
    fn no_extension_returns_none() {
        // Pure filename with no dot — also must return None.
        // Note: rsplit on a string with no '.' returns the whole string,
        // which won't match any known extension.
        assert!(infer_target_format("model").is_none());
    }

    #[test]
    fn full_path_with_directory_resolves() {
        // Path like /tmp/x/y/model.apr must still resolve based on extension.
        assert_eq!(
            infer_target_format("/tmp/qwen/model.apr"),
            Some(TargetFormat::AprV2)
        );
    }

    #[test]
    fn double_extension_uses_last() {
        // "model.tar.gguf" → GGUF (last extension wins).
        assert_eq!(
            infer_target_format("model.tar.gguf"),
            Some(TargetFormat::Gguf)
        );
    }
}
