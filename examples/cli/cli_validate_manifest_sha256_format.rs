//! # apr validate-manifest — sha256 Format Check
//!
//! `apr validate-manifest` verifies the manifest's `sha256` field is
//! a valid hex digest: 64 hex characters, lowercase, no `0x` prefix
//! (per RFC 4648 base16). This recipe builds the validator and asserts
//! the contract.
//!
//! Demonstrates the **VAL-MANIFEST.7** recipe for PMAT-110 (apr validate-manifest coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender contracts/publish-manifest-v1.yaml + RFC 4648 base16
//!
//! Run with: cargo run --example cli_validate_manifest_sha256_format
//!
//! Added by PMAT-110 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum Sha256Verdict {
    Ok,
    WrongLength { observed: usize },
    NonHex { ch: char, position: usize },
    UppercaseHex { position: usize },
    LeadingZeroX,
}

const SHA256_HEX_LEN: usize = 64;

pub fn validate_sha256(s: &str) -> Sha256Verdict {
    if let Some(stripped) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        // Even if the hex length is correct, leading 0x is not RFC 4648.
        let _ = stripped;
        return Sha256Verdict::LeadingZeroX;
    }
    if s.len() != SHA256_HEX_LEN {
        return Sha256Verdict::WrongLength { observed: s.len() };
    }
    for (i, c) in s.chars().enumerate() {
        if !c.is_ascii_hexdigit() {
            return Sha256Verdict::NonHex { ch: c, position: i };
        }
        if c.is_ascii_uppercase() {
            return Sha256Verdict::UppercaseHex { position: i };
        }
    }
    Sha256Verdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_validate_manifest_sha256_format")?;

    let cases = [
        "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        "ABCDEF0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        "0xabcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
        "abc",
        "abcdefghijklmnopabcdefghijklmnopabcdefghijklmnopabcdefghijklmnop",
    ];
    for s in cases {
        let truncated = if s.len() > 30 {
            format!("{}…", &s[..30])
        } else {
            s.to_string()
        };
        println!("{truncated:>32}  →  {:?}", validate_sha256(s));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good() -> String {
        "abcdef0123456789".repeat(4)
    }

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn good_sha256_passes() {
        assert_eq!(validate_sha256(&good()), Sha256Verdict::Ok);
    }

    #[test]
    fn uppercase_hex_rejected() {
        let upper = good().to_ascii_uppercase();
        let v = validate_sha256(&upper);
        assert!(matches!(v, Sha256Verdict::UppercaseHex { .. }));
    }

    #[test]
    fn leading_0x_rejected() {
        let v = validate_sha256(&format!("0x{}", good()));
        assert_eq!(v, Sha256Verdict::LeadingZeroX);
    }

    #[test]
    fn wrong_length_rejected() {
        assert!(matches!(
            validate_sha256("abc"),
            Sha256Verdict::WrongLength { observed: 3 }
        ));
        assert!(matches!(
            validate_sha256("a".repeat(63).as_str()),
            Sha256Verdict::WrongLength { .. }
        ));
        assert!(matches!(
            validate_sha256("a".repeat(65).as_str()),
            Sha256Verdict::WrongLength { .. }
        ));
    }

    #[test]
    fn non_hex_char_rejected() {
        let bad = "g".to_string() + &good()[1..];
        let v = validate_sha256(&bad);
        assert!(matches!(
            v,
            Sha256Verdict::NonHex {
                ch: 'g',
                position: 0
            }
        ));
    }

    #[test]
    fn empty_string_rejected_as_wrong_length() {
        assert!(matches!(
            validate_sha256(""),
            Sha256Verdict::WrongLength { observed: 0 }
        ));
    }

    #[test]
    fn boundary_at_64_chars_passes() {
        // Exactly 64 lowercase hex chars.
        assert_eq!(validate_sha256(&good()), Sha256Verdict::Ok);
    }
}
