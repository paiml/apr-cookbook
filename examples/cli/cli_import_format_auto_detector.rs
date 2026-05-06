//! # apr import — Format Auto-Detector
//!
//! `apr import <FILE>` infers format by magic bytes + extension. Magic
//! beats extension on conflict. Recognised: APR2 (apr-cookbook), GGUF
//! (llama.cpp), PT (PyTorch), SafeTensors (HF), ONNX (Microsoft).
//! Unknown returns Unknown rather than panicking.
//!
//! Demonstrates the **IMP.4** recipe for PMAT-115 (apr import coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender IMP-001 + format spec catalog
//!
//! Run with: cargo run --example cli_import_format_auto_detector
//!
//! Added by PMAT-115 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Format {
    Apr,
    Gguf,
    SafeTensors,
    Pytorch,
    Onnx,
    Unknown,
}

pub fn detect_by_magic(bytes: &[u8]) -> Format {
    if bytes.len() < 4 {
        return Format::Unknown;
    }
    match &bytes[..4] {
        b"APR2" => Format::Apr,
        b"GGUF" => Format::Gguf,
        // SafeTensors header is a JSON-prefixed length frame — no fixed magic;
        // detect via leading u64 (small) + opening `{`.
        b => {
            // PyTorch zip: PK\x03\x04
            if b == b"PK\x03\x04" {
                Format::Pytorch
            // ONNX (protobuf): often starts with 0x08 or 0x12 (varint tags)
            } else if bytes[0] == 0x08 || bytes[0] == 0x12 {
                Format::Onnx
            } else if bytes.len() >= 9 && bytes[8] == b'{' {
                Format::SafeTensors
            } else {
                Format::Unknown
            }
        }
    }
}

pub fn detect_by_ext(path: &str) -> Format {
    if let Some(ext) = path.rsplit('.').next() {
        match ext.to_ascii_lowercase().as_str() {
            "apr" => Format::Apr,
            "gguf" => Format::Gguf,
            "safetensors" => Format::SafeTensors,
            "pt" | "pth" | "bin" => Format::Pytorch,
            "onnx" => Format::Onnx,
            _ => Format::Unknown,
        }
    } else {
        Format::Unknown
    }
}

pub fn detect(bytes: &[u8], path: &str) -> Format {
    let m = detect_by_magic(bytes);
    if m != Format::Unknown {
        return m;
    }
    detect_by_ext(path)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_import_format_auto_detector")?;

    println!("APR2 magic: {:?}", detect_by_magic(b"APR2_more"));
    println!("GGUF magic: {:?}", detect_by_magic(b"GGUF\0"));
    println!("PK ZIP: {:?}", detect_by_magic(b"PK\x03\x04..."));
    println!(".apr ext: {:?}", detect_by_ext("model.apr"));
    println!(".pt ext: {:?}", detect_by_ext("model.pt"));
    println!("unknown: {:?}", detect(b"random", "noext"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn apr_magic_detected() {
        assert_eq!(detect_by_magic(b"APR2_more_bytes"), Format::Apr);
    }

    #[test]
    fn gguf_magic_detected() {
        assert_eq!(detect_by_magic(b"GGUF\0\0\0\0"), Format::Gguf);
    }

    #[test]
    fn pytorch_zip_magic_detected() {
        assert_eq!(detect_by_magic(b"PK\x03\x04rest"), Format::Pytorch);
    }

    #[test]
    fn ext_detection_apr() {
        assert_eq!(detect_by_ext("model.apr"), Format::Apr);
        assert_eq!(detect_by_ext("model.APR"), Format::Apr);
    }

    #[test]
    fn ext_detection_pytorch_variants() {
        assert_eq!(detect_by_ext("model.pt"), Format::Pytorch);
        assert_eq!(detect_by_ext("model.pth"), Format::Pytorch);
        assert_eq!(detect_by_ext("model.bin"), Format::Pytorch);
    }

    #[test]
    fn unknown_format_returns_unknown() {
        assert_eq!(detect_by_magic(b"random"), Format::Unknown);
        assert_eq!(detect_by_ext("noext"), Format::Unknown);
    }

    #[test]
    fn magic_beats_extension_on_conflict() {
        // Bytes start with APR2 but extension says .pt — magic wins.
        let b = b"APR2_misnamed";
        assert_eq!(detect(b, "misnamed.pt"), Format::Apr);
    }

    #[test]
    fn ext_used_when_magic_unknown() {
        let b = b"unrecognized_random_bytes_here";
        assert_eq!(detect(b, "model.apr"), Format::Apr);
    }

    #[test]
    fn tiny_bytes_yield_unknown() {
        assert_eq!(detect_by_magic(b""), Format::Unknown);
        assert_eq!(detect_by_magic(b"AP"), Format::Unknown);
    }
}
