//! # Format Magic-Bytes Validator
//!
//! File-format identification by leading bytes:
//! - APR2: `b"APR2"` (4 bytes)
//! - GGUF: `b"GGUF"` (4 bytes)
//! - SafeTensors: 8-byte LE u64 header_len, no magic — detect by
//!   range-of-plausibility heuristic
//! - ONNX: protobuf, no magic — best-effort header sniff
//!
//! This recipe builds the classifier.
//!
//! Demonstrates the **FMT.24** recipe for PMAT-136 (format round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: APR file format spec; GGUF spec.
//!
//! Run with: cargo run --example format_magic_bytes_validator
//!
//! Added by PMAT-136 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    Apr2,
    Gguf,
    SafeTensorsLikely,
    Unknown,
}

#[derive(Debug, PartialEq)]
pub enum ClassifyVerdict {
    Ok(FileFormat),
    TooShort { len: usize, min: usize },
}

const MIN_BYTES: usize = 8;
const SAFETENSORS_MAX_HEADER_LEN: u64 = 100 * 1024 * 1024;

pub fn classify(prefix: &[u8]) -> ClassifyVerdict {
    if prefix.len() < MIN_BYTES {
        return ClassifyVerdict::TooShort {
            len: prefix.len(),
            min: MIN_BYTES,
        };
    }
    if &prefix[0..4] == b"APR2" {
        return ClassifyVerdict::Ok(FileFormat::Apr2);
    }
    if &prefix[0..4] == b"GGUF" {
        return ClassifyVerdict::Ok(FileFormat::Gguf);
    }
    let header_len = u64::from_le_bytes(prefix[0..8].try_into().unwrap_or([0; 8]));
    if header_len > 0 && header_len < SAFETENSORS_MAX_HEADER_LEN {
        return ClassifyVerdict::Ok(FileFormat::SafeTensorsLikely);
    }
    ClassifyVerdict::Ok(FileFormat::Unknown)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("format_magic_bytes_validator")?;

    let apr2 = b"APR2\x00\x00\x00\x00";
    println!("APR2: {:?}", classify(apr2));

    let gguf = b"GGUF\x00\x00\x00\x00";
    println!("GGUF: {:?}", classify(gguf));

    let st = 1024u64.to_le_bytes();
    println!("SafeTensors-like: {:?}", classify(&st));

    println!("short: {:?}", classify(b"abc"));
    println!("unknown: {:?}", classify(&[0u8; 8]));
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
    fn apr2_recognized() {
        let v = classify(b"APR2\x00\x00\x00\x00");
        assert_eq!(v, ClassifyVerdict::Ok(FileFormat::Apr2));
    }

    #[test]
    fn gguf_recognized() {
        let v = classify(b"GGUF\x00\x00\x00\x00");
        assert_eq!(v, ClassifyVerdict::Ok(FileFormat::Gguf));
    }

    #[test]
    fn safetensors_typical_recognized() {
        let header_len = 1024u64.to_le_bytes();
        let v = classify(&header_len);
        assert_eq!(v, ClassifyVerdict::Ok(FileFormat::SafeTensorsLikely));
    }

    #[test]
    fn too_short_rejected() {
        let v = classify(b"abc");
        assert!(matches!(v, ClassifyVerdict::TooShort { .. }));
    }

    #[test]
    fn zero_header_unknown() {
        let v = classify(&[0u8; 8]);
        assert_eq!(v, ClassifyVerdict::Ok(FileFormat::Unknown));
    }

    #[test]
    fn excessive_header_unknown() {
        // Header > 100 MiB → not plausible.
        let huge = (200u64 * 1024 * 1024).to_le_bytes();
        let v = classify(&huge);
        assert_eq!(v, ClassifyVerdict::Ok(FileFormat::Unknown));
    }

    #[test]
    fn apr2_with_extra_bytes_still_recognized() {
        let mut bytes = b"APR2".to_vec();
        bytes.extend_from_slice(&[0u8; 100]);
        assert_eq!(classify(&bytes), ClassifyVerdict::Ok(FileFormat::Apr2));
    }

    #[test]
    fn gguf_with_extra_bytes_recognized() {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&[0u8; 100]);
        assert_eq!(classify(&bytes), ClassifyVerdict::Ok(FileFormat::Gguf));
    }

    #[test]
    fn boundary_at_min_bytes_passes() {
        // exactly 8 bytes is the minimum.
        let v = classify(&[0u8; 8]);
        assert!(matches!(v, ClassifyVerdict::Ok(_)));
    }

    #[test]
    fn case_sensitive_magic() {
        // "apr2" lowercase → not recognized as APR2.
        let v = classify(b"apr2\x00\x00\x00\x00");
        assert_ne!(v, ClassifyVerdict::Ok(FileFormat::Apr2));
    }

    #[test]
    fn just_below_threshold_safetensors_recognized() {
        // 99 MiB is below cap → SafeTensorsLikely.
        let almost = (99u64 * 1024 * 1024).to_le_bytes();
        assert_eq!(
            classify(&almost),
            ClassifyVerdict::Ok(FileFormat::SafeTensorsLikely)
        );
    }
}
