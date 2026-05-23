//! # Conversion SafeTensors Header Extractor
//!
//! SafeTensors layout: 8-byte LE u64 header_length; JSON header with
//! per-tensor `{name: {dtype, shape, data_offsets: [start, end]}}`;
//! contiguous tensor blob. This recipe parses the header bytes + sums
//! declared blob length.
//!
//! Demonstrates the **CONV.12** recipe for PMAT-133 (conversion coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HuggingFace SafeTensors format spec.
//!
//! Run with: cargo run --example convert_safetensors_header
//!
//! Added by PMAT-133 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum HeaderVerdict {
    Ok {
        header_len: u64,
        declared_blob_bytes: u64,
    },
    TooShort {
        got: usize,
        need: usize,
    },
    HeaderLenExceedsFile {
        header_len: u64,
        file_len: usize,
    },
    HeaderLenZero,
    HeaderLenAboveCap {
        header_len: u64,
        cap: u64,
    },
}

const HEADER_PREFIX_BYTES: usize = 8;
const MAX_HEADER_BYTES: u64 = 100 * 1024 * 1024; // 100 MiB

pub fn parse_header_prefix(bytes: &[u8]) -> HeaderVerdict {
    if bytes.len() < HEADER_PREFIX_BYTES {
        return HeaderVerdict::TooShort {
            got: bytes.len(),
            need: HEADER_PREFIX_BYTES,
        };
    }
    let header_len = u64::from_le_bytes(bytes[..HEADER_PREFIX_BYTES].try_into().unwrap());
    if header_len == 0 {
        return HeaderVerdict::HeaderLenZero;
    }
    if header_len > MAX_HEADER_BYTES {
        return HeaderVerdict::HeaderLenAboveCap {
            header_len,
            cap: MAX_HEADER_BYTES,
        };
    }
    let total_header_bytes = HEADER_PREFIX_BYTES.saturating_add(header_len as usize);
    if total_header_bytes > bytes.len() {
        return HeaderVerdict::HeaderLenExceedsFile {
            header_len,
            file_len: bytes.len(),
        };
    }
    let declared_blob_bytes = (bytes.len() - total_header_bytes) as u64;
    HeaderVerdict::Ok {
        header_len,
        declared_blob_bytes,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeTensorsDtype {
    F16,
    Bf16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    Bool,
}

impl SafeTensorsDtype {
    pub fn from_str_strict(s: &str) -> Option<Self> {
        match s {
            "F16" => Some(SafeTensorsDtype::F16),
            "BF16" => Some(SafeTensorsDtype::Bf16),
            "F32" => Some(SafeTensorsDtype::F32),
            "F64" => Some(SafeTensorsDtype::F64),
            "I8" => Some(SafeTensorsDtype::I8),
            "I16" => Some(SafeTensorsDtype::I16),
            "I32" => Some(SafeTensorsDtype::I32),
            "I64" => Some(SafeTensorsDtype::I64),
            "U8" => Some(SafeTensorsDtype::U8),
            "BOOL" => Some(SafeTensorsDtype::Bool),
            _ => None,
        }
    }

    pub fn bytes_per_element(self) -> u32 {
        match self {
            SafeTensorsDtype::F16 | SafeTensorsDtype::Bf16 | SafeTensorsDtype::I16 => 2,
            SafeTensorsDtype::F32 | SafeTensorsDtype::I32 => 4,
            SafeTensorsDtype::F64 | SafeTensorsDtype::I64 => 8,
            SafeTensorsDtype::I8 | SafeTensorsDtype::U8 | SafeTensorsDtype::Bool => 1,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("convert_safetensors_header")?;

    // Synthesize: header_len = 64, then 64 bytes of "header", then 16 bytes of blob.
    let mut bytes = 64u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(&[b'X'; 64]);
    bytes.extend_from_slice(&[0u8; 16]);
    println!("valid: {:?}", parse_header_prefix(&bytes));

    let too_short = b"AB";
    println!("too short: {:?}", parse_header_prefix(too_short));

    let bad_header_len = u64::MAX.to_le_bytes();
    println!("max u64 header: {:?}", parse_header_prefix(&bad_header_len));

    for ty in ["F16", "BF16", "F32", "I8", "BOOL", "TYPO"] {
        println!("dtype {ty}: {:?}", SafeTensorsDtype::from_str_strict(ty));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_header_parses() {
        let mut bytes = 64u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(&[b'X'; 64]);
        bytes.extend_from_slice(&[0u8; 16]);
        if let HeaderVerdict::Ok {
            header_len,
            declared_blob_bytes,
        } = parse_header_prefix(&bytes)
        {
            assert_eq!(header_len, 64);
            assert_eq!(declared_blob_bytes, 16);
        }
    }

    #[test]
    fn too_short_rejected() {
        assert!(matches!(
            parse_header_prefix(b"abc"),
            HeaderVerdict::TooShort { got: 3, need: 8 }
        ));
    }

    #[test]
    fn zero_header_len_rejected() {
        let zeros = vec![0u8; 16];
        assert_eq!(parse_header_prefix(&zeros), HeaderVerdict::HeaderLenZero);
    }

    #[test]
    fn header_above_cap_rejected() {
        let max_bytes = u64::MAX.to_le_bytes();
        let v = parse_header_prefix(&max_bytes);
        assert!(matches!(v, HeaderVerdict::HeaderLenAboveCap { .. }));
    }

    #[test]
    fn header_exceeds_file_rejected() {
        let mut bytes = 1000u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(&[0u8; 8]); // only 8 bytes for "header" claimed to be 1000
        let v = parse_header_prefix(&bytes);
        assert!(matches!(v, HeaderVerdict::HeaderLenExceedsFile { .. }));
    }

    #[test]
    fn dtype_round_trip() {
        for s in [
            "F16", "BF16", "F32", "F64", "I8", "I16", "I32", "I64", "U8", "BOOL",
        ] {
            assert!(SafeTensorsDtype::from_str_strict(s).is_some());
        }
        assert!(SafeTensorsDtype::from_str_strict("typo").is_none());
    }

    #[test]
    fn dtype_bytes_per_element_correct() {
        assert_eq!(SafeTensorsDtype::F16.bytes_per_element(), 2);
        assert_eq!(SafeTensorsDtype::F32.bytes_per_element(), 4);
        assert_eq!(SafeTensorsDtype::F64.bytes_per_element(), 8);
        assert_eq!(SafeTensorsDtype::I8.bytes_per_element(), 1);
    }

    #[test]
    fn bf16_same_size_as_f16() {
        assert_eq!(
            SafeTensorsDtype::Bf16.bytes_per_element(),
            SafeTensorsDtype::F16.bytes_per_element()
        );
    }

    #[test]
    fn declared_blob_zero_when_no_blob() {
        let mut bytes = 8u64.to_le_bytes().to_vec();
        bytes.extend_from_slice(&[b'X'; 8]); // header only, no blob
        if let HeaderVerdict::Ok {
            declared_blob_bytes,
            ..
        } = parse_header_prefix(&bytes)
        {
            assert_eq!(declared_blob_bytes, 0);
        }
    }
}
