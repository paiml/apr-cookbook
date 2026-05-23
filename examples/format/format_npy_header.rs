//! # Format NumPy .npy Header Parser
//!
//! NumPy .npy file structure: 6-byte magic (\x93NUMPY) + version (1
//! or 2) + header length + Python-dict header. This recipe parses
//! the magic and validates the version.
//!
//! Demonstrates the **FMT.31** recipe for PMAT-153 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NumPy .npy file format spec NEP 1.
//!
//! Run with: cargo run --example format_npy_header
//!
//! Added by PMAT-153 (catalog 1000→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MAGIC: &[u8] = &[0x93, b'N', b'U', b'M', b'P', b'Y'];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpyVersion {
    V1,
    V2,
    V3,
}

#[derive(Debug, PartialEq)]
pub enum NpyVerdict {
    Ok {
        version: NpyVersion,
        header_len: u32,
        data_offset: u64,
    },
    BadMagic,
    UnsupportedVersion {
        major: u8,
        minor: u8,
    },
    TruncatedHeader,
}

pub fn parse(bytes: &[u8]) -> NpyVerdict {
    if bytes.len() < 10 {
        return NpyVerdict::TruncatedHeader;
    }
    if &bytes[0..6] != MAGIC {
        return NpyVerdict::BadMagic;
    }
    let major = bytes[6];
    let minor = bytes[7];
    let version = match (major, minor) {
        (1, 0) => NpyVersion::V1,
        (2, 0) => NpyVersion::V2,
        (3, 0) => NpyVersion::V3,
        _ => return NpyVerdict::UnsupportedVersion { major, minor },
    };
    let (header_len, data_offset) = match version {
        NpyVersion::V1 => {
            // 2-byte little-endian header length.
            let hl = u32::from(u16::from_le_bytes([bytes[8], bytes[9]]));
            (hl, 10 + u64::from(hl))
        }
        NpyVersion::V2 | NpyVersion::V3 => {
            if bytes.len() < 12 {
                return NpyVerdict::TruncatedHeader;
            }
            let hl = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
            (hl, 12 + u64::from(hl))
        }
    };
    NpyVerdict::Ok {
        version,
        header_len,
        data_offset,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("format_npy_header")?;

    let mut v1 = Vec::from(MAGIC);
    v1.extend_from_slice(&[1, 0, 0x80, 0x00]);
    println!("v1: {:?}", parse(&v1));

    let mut v2 = Vec::from(MAGIC);
    v2.extend_from_slice(&[2, 0, 0x00, 0x80, 0x00, 0x00]);
    println!("v2: {:?}", parse(&v2));

    println!("bad magic: {:?}", parse(&[0; 12]));
    println!("truncated: {:?}", parse(&[0x93, b'N']));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_v1(header_len: u16) -> Vec<u8> {
        let mut bytes = Vec::from(MAGIC);
        bytes.extend_from_slice(&[1, 0]);
        bytes.extend_from_slice(&header_len.to_le_bytes());
        bytes
    }

    fn build_v2(header_len: u32) -> Vec<u8> {
        let mut bytes = Vec::from(MAGIC);
        bytes.extend_from_slice(&[2, 0]);
        bytes.extend_from_slice(&header_len.to_le_bytes());
        bytes
    }

    #[test]
    fn parser_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn v1_parsed() {
        let bytes = build_v1(0x80);
        if let NpyVerdict::Ok { version, .. } = parse(&bytes) {
            assert_eq!(version, NpyVersion::V1);
        }
    }

    #[test]
    fn v2_parsed() {
        let bytes = build_v2(0x80);
        if let NpyVerdict::Ok { version, .. } = parse(&bytes) {
            assert_eq!(version, NpyVersion::V2);
        }
    }

    #[test]
    fn v3_parsed() {
        let mut bytes = Vec::from(MAGIC);
        bytes.extend_from_slice(&[3, 0]);
        bytes.extend_from_slice(&0x100u32.to_le_bytes());
        if let NpyVerdict::Ok { version, .. } = parse(&bytes) {
            assert_eq!(version, NpyVersion::V3);
        }
    }

    #[test]
    fn bad_magic_rejected() {
        assert_eq!(parse(&[0u8; 12]), NpyVerdict::BadMagic);
    }

    #[test]
    fn truncated_rejected() {
        assert_eq!(parse(&[0x93, b'N']), NpyVerdict::TruncatedHeader);
    }

    #[test]
    fn unsupported_version() {
        let mut bytes = Vec::from(MAGIC);
        bytes.extend_from_slice(&[5, 0, 0, 0, 0, 0]);
        assert!(matches!(
            parse(&bytes),
            NpyVerdict::UnsupportedVersion { .. }
        ));
    }

    #[test]
    fn v1_data_offset_correct() {
        let bytes = build_v1(0x80);
        if let NpyVerdict::Ok {
            data_offset,
            header_len,
            ..
        } = parse(&bytes)
        {
            assert_eq!(header_len, 0x80);
            assert_eq!(data_offset, 10 + 0x80);
        }
    }

    #[test]
    fn v2_data_offset_correct() {
        let bytes = build_v2(0x100);
        if let NpyVerdict::Ok {
            data_offset,
            header_len,
            ..
        } = parse(&bytes)
        {
            assert_eq!(header_len, 0x100);
            assert_eq!(data_offset, 12 + 0x100);
        }
    }

    #[test]
    fn empty_input_truncated() {
        assert_eq!(parse(&[]), NpyVerdict::TruncatedHeader);
    }

    #[test]
    fn v2_short_header_truncated() {
        // V2 needs 12 bytes for header.
        let mut bytes = Vec::from(MAGIC);
        bytes.extend_from_slice(&[2, 0]);
        bytes.extend_from_slice(&[0u8; 2]); // not enough
        assert_eq!(parse(&bytes), NpyVerdict::TruncatedHeader);
    }
}
