//! # apr rosetta inspect — Format Detection
//!
//! `apr rosetta inspect <FILE>` auto-detects the model format (APR / GGUF /
//! SafeTensors / ONNX) by reading the first few bytes (magic number).
//! This recipe builds the magic-byte detector and asserts the contract:
//! ambiguous prefixes return Unknown rather than guessing.
//!
//! Demonstrates the **ROSETTA-INSPECT.1** recipe for PMAT-098 (apr rosetta inspect coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-ROSETTA-001 + APR/GGUF/SafeTensors/ONNX format magic bytes
//!
//! Run with: cargo run --example cli_rosetta_inspect_format_detector
//!
//! Added by PMAT-098 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    AprV2,
    Gguf,
    SafeTensors,
    Onnx,
    Unknown,
}

pub fn detect_format(prefix: &[u8]) -> Format {
    // APR v2: "APR2" magic.
    if prefix.starts_with(b"APR2") {
        return Format::AprV2;
    }
    // GGUF: 4-byte "GGUF" magic.
    if prefix.starts_with(b"GGUF") {
        return Format::Gguf;
    }
    // ONNX: starts with protobuf field tag 0x08 0x07 (model.ir_version=7).
    // Use a heuristic — leading 0x08 tag + small varint.
    if prefix.len() >= 2 && prefix[0] == 0x08 && (1..=10).contains(&prefix[1]) {
        return Format::Onnx;
    }
    // SafeTensors: starts with an 8-byte little-endian header length, then
    // a JSON object beginning with `{`. Detect by sniffing for the `{` at
    // offset 8.
    if prefix.len() >= 9 && prefix[8] == b'{' {
        return Format::SafeTensors;
    }
    Format::Unknown
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_inspect_format_detector")?;

    let apr = b"APR2\x00\x00\x00\x00";
    let gguf = b"GGUF\x03\x00\x00\x00";
    let safetensors_prefix = {
        let mut v = vec![0u8; 8];
        v.push(b'{');
        v.extend_from_slice(b"\"weight\"...");
        v
    };
    let onnx = &[0x08, 0x07, 0x12, 0x05][..];
    let unknown = b"random\x00\x00";

    for (label, bytes) in &[
        ("APR v2", &apr[..]),
        ("GGUF", &gguf[..]),
        ("SafeTensors", &safetensors_prefix[..]),
        ("ONNX", onnx),
        ("Unknown", &unknown[..]),
    ] {
        println!("{label:>12}  →  {:?}", detect_format(bytes));
    }
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
        assert_eq!(detect_format(b"APR2\x00"), Format::AprV2);
    }

    #[test]
    fn gguf_magic_detected() {
        assert_eq!(detect_format(b"GGUF\x03\x00\x00\x00"), Format::Gguf);
    }

    #[test]
    fn safetensors_detected_by_brace_after_8_bytes() {
        let mut v = vec![0xCDu8; 8]; // 8-byte header length
        v.push(b'{');
        v.extend_from_slice(b"\"foo\":1}");
        assert_eq!(detect_format(&v), Format::SafeTensors);
    }

    #[test]
    fn onnx_detected_by_protobuf_tag() {
        assert_eq!(detect_format(&[0x08, 0x07]), Format::Onnx);
    }

    #[test]
    fn empty_input_returns_unknown_not_apr() {
        // Empty input has no magic — must NOT return AprV2 (would happen if
        // the impl used `starts_with(b"")` as a fallback).
        assert_eq!(detect_format(&[]), Format::Unknown);
    }

    #[test]
    fn random_bytes_return_unknown() {
        assert_eq!(detect_format(b"random\x00\x00\x00"), Format::Unknown);
    }

    #[test]
    fn truncated_apr_does_not_match() {
        // Less than 4 bytes can't satisfy starts_with("APR2").
        assert_eq!(detect_format(b"APR"), Format::Unknown);
    }
}
