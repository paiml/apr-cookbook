//! # apr encrypt --aad — Additional Authenticated Data Envelope
//!
//! AES-GCM accepts AAD (Additional Authenticated Data) — bytes that
//! must match on decrypt but are NOT encrypted. Used for context
//! binding (model name, version, owner). Constraints: max 2^61 bytes
//! per RFC 5116; in practice cap at 64 KiB. UTF-8 only when from CLI.
//!
//! Demonstrates the **ENC.6** recipe for PMAT-115 (apr encrypt coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender ENC-001 + RFC 5116 (AEAD interface)
//!
//! Run with: cargo run --example cli_encrypt_aad_envelope
//!
//! Added by PMAT-115 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MAX_AAD_BYTES: usize = 64 * 1024;

#[derive(Debug, PartialEq)]
pub enum AadVerdict {
    Ok { byte_count: usize },
    TooLarge { byte_count: usize, max: usize },
    InvalidUtf8,
}

pub fn validate_string_aad(s: &str) -> AadVerdict {
    let len = s.len();
    if len > MAX_AAD_BYTES {
        return AadVerdict::TooLarge {
            byte_count: len,
            max: MAX_AAD_BYTES,
        };
    }
    AadVerdict::Ok { byte_count: len }
}

pub fn validate_bytes_aad(bytes: &[u8], require_utf8: bool) -> AadVerdict {
    if bytes.len() > MAX_AAD_BYTES {
        return AadVerdict::TooLarge {
            byte_count: bytes.len(),
            max: MAX_AAD_BYTES,
        };
    }
    if require_utf8 && std::str::from_utf8(bytes).is_err() {
        return AadVerdict::InvalidUtf8;
    }
    AadVerdict::Ok {
        byte_count: bytes.len(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_encrypt_aad_envelope")?;

    println!("empty → {:?}", validate_string_aad(""));
    println!("normal → {:?}", validate_string_aad("model=llama-3:v1.0"));
    let big = "x".repeat(70_000);
    println!("oversize → {:?}", validate_string_aad(&big));
    let invalid = vec![0xff, 0xfe, 0x00];
    println!("non-utf8 → {:?}", validate_bytes_aad(&invalid, true));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_aad_passes() {
        let v = validate_string_aad("");
        assert!(matches!(v, AadVerdict::Ok { byte_count: 0 }));
    }

    #[test]
    fn typical_metadata_aad_passes() {
        let v = validate_string_aad("model=llama-3:v1.0;owner=meta");
        assert!(matches!(v, AadVerdict::Ok { .. }));
    }

    #[test]
    fn oversize_string_rejected() {
        let big = "x".repeat(MAX_AAD_BYTES + 1);
        let v = validate_string_aad(&big);
        assert!(matches!(v, AadVerdict::TooLarge { .. }));
    }

    #[test]
    fn at_max_size_passes() {
        let exactly_max = "x".repeat(MAX_AAD_BYTES);
        assert!(matches!(
            validate_string_aad(&exactly_max),
            AadVerdict::Ok { .. }
        ));
    }

    #[test]
    fn utf8_bytes_pass_when_required() {
        let bytes = "model=llama".as_bytes();
        assert!(matches!(
            validate_bytes_aad(bytes, true),
            AadVerdict::Ok { .. }
        ));
    }

    #[test]
    fn non_utf8_bytes_rejected_when_required() {
        let bytes = [0xff, 0xfe, 0x00];
        assert_eq!(validate_bytes_aad(&bytes, true), AadVerdict::InvalidUtf8);
    }

    #[test]
    fn non_utf8_bytes_pass_when_not_required() {
        // Binary AAD allowed when caller opts out of UTF-8.
        let bytes = [0xff, 0xfe, 0x00];
        assert!(matches!(
            validate_bytes_aad(&bytes, false),
            AadVerdict::Ok { .. }
        ));
    }

    #[test]
    fn oversize_bytes_rejected_before_utf8_check() {
        // Length check fires first.
        let big = vec![0xff; MAX_AAD_BYTES + 1];
        assert!(matches!(
            validate_bytes_aad(&big, true),
            AadVerdict::TooLarge { .. }
        ));
    }
}
