//! # Bundle Manifest Header Validator
//!
//! APR2 bundles open with: 4-byte magic "APR2", 1-byte major version,
//! 1-byte minor version, 2-byte feature flag bitmap, 8-byte manifest
//! length (LE u64). This recipe parses the header and validates each
//! field.
//!
//! Demonstrates the **BUNDLE.11** recipe for PMAT-127 (bundling coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender APR-FORMAT spec §3.1.
//!
//! Run with: cargo run --example bundle_manifest_header_validator
//!
//! Added by PMAT-127 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MAGIC: &[u8; 4] = b"APR2";
const HEADER_LEN: usize = 16;
const MAX_MANIFEST_BYTES: u64 = 1 << 30; // 1 GiB

#[derive(Debug, PartialEq)]
pub struct Header {
    pub major: u8,
    pub minor: u8,
    pub flags: u16,
    pub manifest_len: u64,
}

#[derive(Debug, PartialEq)]
pub enum HeaderVerdict {
    Ok(Header),
    TooShort { len: usize },
    BadMagic { got: [u8; 4] },
    UnsupportedMajor { got: u8 },
    ManifestTooLarge { len: u64, max: u64 },
}

const SUPPORTED_MAJORS: &[u8] = &[1, 2];

pub fn parse(bytes: &[u8]) -> HeaderVerdict {
    if bytes.len() < HEADER_LEN {
        return HeaderVerdict::TooShort { len: bytes.len() };
    }
    let magic: [u8; 4] = bytes[..4].try_into().unwrap();
    if &magic != MAGIC {
        return HeaderVerdict::BadMagic { got: magic };
    }
    let major = bytes[4];
    let minor = bytes[5];
    if !SUPPORTED_MAJORS.contains(&major) {
        return HeaderVerdict::UnsupportedMajor { got: major };
    }
    let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
    let manifest_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    if manifest_len > MAX_MANIFEST_BYTES {
        return HeaderVerdict::ManifestTooLarge {
            len: manifest_len,
            max: MAX_MANIFEST_BYTES,
        };
    }
    HeaderVerdict::Ok(Header {
        major,
        minor,
        flags,
        manifest_len,
    })
}

pub fn flag_set(flags: u16, bit: u8) -> bool {
    bit < 16 && (flags >> bit) & 1 == 1
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("bundle_manifest_header_validator")?;

    let mut header = Vec::new();
    header.extend_from_slice(b"APR2");
    header.extend_from_slice(&[2u8, 1u8]); // 2.1
    header.extend_from_slice(&3u16.to_le_bytes()); // flags = 0b11
    header.extend_from_slice(&1024u64.to_le_bytes()); // manifest_len
    println!("valid: {:?}", parse(&header));

    let mut bad_magic = header.clone();
    bad_magic[0] = b'X';
    println!("bad magic: {:?}", parse(&bad_magic));

    println!("short: {:?}", parse(b"APR2"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_header(major: u8, minor: u8, flags: u16, manifest_len: u64) -> Vec<u8> {
        let mut h = Vec::new();
        h.extend_from_slice(b"APR2");
        h.extend_from_slice(&[major, minor]);
        h.extend_from_slice(&flags.to_le_bytes());
        h.extend_from_slice(&manifest_len.to_le_bytes());
        h
    }

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_v2_header_parses() {
        let h = make_header(2, 1, 3, 1024);
        if let HeaderVerdict::Ok(parsed) = parse(&h) {
            assert_eq!(parsed.major, 2);
            assert_eq!(parsed.minor, 1);
            assert_eq!(parsed.flags, 3);
            assert_eq!(parsed.manifest_len, 1024);
        }
    }

    #[test]
    fn v1_header_also_supported() {
        let h = make_header(1, 0, 0, 100);
        assert!(matches!(parse(&h), HeaderVerdict::Ok(_)));
    }

    #[test]
    fn future_major_rejected() {
        let h = make_header(99, 0, 0, 100);
        assert!(matches!(
            parse(&h),
            HeaderVerdict::UnsupportedMajor { got: 99 }
        ));
    }

    #[test]
    fn bad_magic_rejected() {
        let mut h = make_header(2, 0, 0, 100);
        h[0] = b'Z';
        assert!(matches!(parse(&h), HeaderVerdict::BadMagic { .. }));
    }

    #[test]
    fn too_short_rejected() {
        assert!(matches!(parse(b"APR2"), HeaderVerdict::TooShort { len: 4 }));
        assert!(matches!(parse(&[]), HeaderVerdict::TooShort { len: 0 }));
    }

    #[test]
    fn oversized_manifest_rejected() {
        let h = make_header(2, 0, 0, MAX_MANIFEST_BYTES + 1);
        assert!(matches!(parse(&h), HeaderVerdict::ManifestTooLarge { .. }));
    }

    #[test]
    fn at_max_manifest_passes() {
        let h = make_header(2, 0, 0, MAX_MANIFEST_BYTES);
        assert!(matches!(parse(&h), HeaderVerdict::Ok(_)));
    }

    #[test]
    fn flag_set_helper_checks_bits() {
        assert!(flag_set(0b0001, 0));
        assert!(!flag_set(0b0001, 1));
        assert!(flag_set(0b1000, 3));
        // Out-of-range bit returns false.
        assert!(!flag_set(0xFFFF, 16));
    }

    #[test]
    fn flags_round_trip_through_header() {
        let h = make_header(2, 0, 0xFEED, 100);
        if let HeaderVerdict::Ok(p) = parse(&h) {
            assert_eq!(p.flags, 0xFEED);
        }
    }
}
