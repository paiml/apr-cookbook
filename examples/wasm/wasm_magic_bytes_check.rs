//! # WASM Magic Bytes Check
//!
//! Validate WASM module header: 4-byte magic `\0asm` + 4-byte version
//! (currently 1, little-endian). Returns categorical verdict.
//!
//! Demonstrates the **WASM.X** recipe for PMAT-224 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly Core §5.5.1 module preamble; binary format
//!  v1 specification.
//!
//! Run with: cargo run --example wasm_magic_bytes_check
//!
//! Added by PMAT-224 (catalog 1639→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MagicVerdict {
    Valid,
    BadMagic,
    UnsupportedVersion { version: u32 },
    TruncatedHeader,
    InvalidConfig,
}

pub fn validate(bytes: &[u8]) -> MagicVerdict {
    if bytes.is_empty() {
        return MagicVerdict::InvalidConfig;
    }
    if bytes.len() < 8 {
        return MagicVerdict::TruncatedHeader;
    }
    if &bytes[0..4] != b"\0asm" {
        return MagicVerdict::BadMagic;
    }
    let version = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    if version != 1 {
        return MagicVerdict::UnsupportedVersion { version };
    }
    MagicVerdict::Valid
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("wasm_magic_bytes_check")?;

    println!("valid: {:?}", validate(&[0, b'a', b's', b'm', 1, 0, 0, 0]));
    println!("bad magic: {:?}", validate(&[1, 2, 3, 4, 1, 0, 0, 0]));
    println!("v2: {:?}", validate(&[0, b'a', b's', b'm', 2, 0, 0, 0]));
    println!("truncated: {:?}", validate(&[0, b'a', b's']));
    println!("invalid: {:?}", validate(&[]));
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
    fn empty_input_rejected() {
        assert_eq!(validate(&[]), MagicVerdict::InvalidConfig);
    }

    #[test]
    fn truncated_header_detected() {
        assert_eq!(validate(&[0, b'a', b's']), MagicVerdict::TruncatedHeader);
    }

    #[test]
    fn valid_v1_accepted() {
        let bytes = [0, b'a', b's', b'm', 1, 0, 0, 0];
        assert_eq!(validate(&bytes), MagicVerdict::Valid);
    }

    #[test]
    fn bad_magic_rejected() {
        let bytes = [1, 2, 3, 4, 1, 0, 0, 0];
        assert_eq!(validate(&bytes), MagicVerdict::BadMagic);
    }

    #[test]
    fn unsupported_version_returns_version() {
        let bytes = [0, b'a', b's', b'm', 2, 0, 0, 0];
        if let MagicVerdict::UnsupportedVersion { version } = validate(&bytes) {
            assert_eq!(version, 2);
        }
    }

    #[test]
    fn deterministic() {
        let bytes = [0, b'a', b's', b'm', 1, 0, 0, 0];
        let r1 = validate(&bytes);
        let r2 = validate(&bytes);
        assert_eq!(r1, r2);
    }

    #[test]
    fn extra_bytes_after_header_ok() {
        let mut bytes = vec![0, b'a', b's', b'm', 1, 0, 0, 0];
        bytes.extend(vec![0; 100]);
        assert_eq!(validate(&bytes), MagicVerdict::Valid);
    }

    #[test]
    fn boundary_8_bytes_validates() {
        let bytes = [0, b'a', b's', b'm', 1, 0, 0, 0];
        assert_eq!(bytes.len(), 8);
        assert_eq!(validate(&bytes), MagicVerdict::Valid);
    }

    #[test]
    fn version_zero_unsupported() {
        let bytes = [0, b'a', b's', b'm', 0, 0, 0, 0];
        assert!(matches!(
            validate(&bytes),
            MagicVerdict::UnsupportedVersion { version: 0 }
        ));
    }

    #[test]
    fn high_version_unsupported() {
        let bytes = [0, b'a', b's', b'm', 99, 0, 0, 0];
        if let MagicVerdict::UnsupportedVersion { version } = validate(&bytes) {
            assert_eq!(version, 99);
        }
    }

    #[test]
    fn one_byte_truncated() {
        assert_eq!(validate(&[0]), MagicVerdict::TruncatedHeader);
    }

    #[test]
    fn seven_bytes_truncated() {
        assert_eq!(
            validate(&[0, b'a', b's', b'm', 1, 0, 0]),
            MagicVerdict::TruncatedHeader
        );
    }
}
