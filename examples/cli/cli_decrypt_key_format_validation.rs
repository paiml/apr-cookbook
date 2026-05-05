//! # apr decrypt — Key Format Validation
//!
//! `apr decrypt --key-file <FILE>` reads a 32-byte (256-bit) raw key from
//! disk. The key file must be EXACTLY 32 bytes — shorter is too weak,
//! longer indicates the operator passed the wrong file (PEM, hex string,
//! base64 with newlines). This recipe documents and tests the format
//! invariants, including the world-readable mode-bits warning.
//!
//! Demonstrates the **DECRYPT.4** recipe for PMAT-095 (apr decrypt coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SHIP-009 + AES-256-GCM (NIST SP 800-38D)
//!
//! Run with: cargo run --example cli_decrypt_key_format_validation
//!
//! Added by PMAT-095 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum KeyVerdict {
    Ok,
    WrongLength { observed: usize, expected: usize },
    LooksLikePem,
    LooksLikeHex,
    LooksLikeBase64,
    AllZeros,
    WorldReadable,
}

const KEY_LEN: usize = 32;

pub fn validate_key_bytes(bytes: &[u8], mode_octal: u32) -> Vec<KeyVerdict> {
    let mut out = Vec::new();
    if bytes.len() != KEY_LEN {
        out.push(KeyVerdict::WrongLength {
            observed: bytes.len(),
            expected: KEY_LEN,
        });
    }
    if bytes.starts_with(b"-----BEGIN") {
        out.push(KeyVerdict::LooksLikePem);
    }
    if bytes.iter().all(u8::is_ascii_hexdigit) && bytes.len() > KEY_LEN {
        out.push(KeyVerdict::LooksLikeHex);
    }
    if bytes
        .iter()
        .all(|b| b.is_ascii_alphanumeric() || *b == b'+' || *b == b'/' || *b == b'=' || *b == b'\n')
        && bytes.len() > KEY_LEN
    {
        out.push(KeyVerdict::LooksLikeBase64);
    }
    if bytes.iter().all(|b| *b == 0) {
        out.push(KeyVerdict::AllZeros);
    }
    if (mode_octal & 0o044) != 0 {
        out.push(KeyVerdict::WorldReadable);
    }
    if out.is_empty() {
        out.push(KeyVerdict::Ok);
    }
    out
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_decrypt_key_format_validation")?;

    let happy = [0xab; 32];
    let too_short = [0xab; 16];
    let too_long_hex = b"deadbeefcafebabe1122334455667788deadbeefcafebabe1122334455667788";
    let pem = b"-----BEGIN PRIVATE KEY-----\nMIIBOwIBA...";
    let zeros = [0u8; 32];

    println!("happy:    {:?}", validate_key_bytes(&happy, 0o600));
    println!("short:    {:?}", validate_key_bytes(&too_short, 0o600));
    println!("hex 64:   {:?}", validate_key_bytes(too_long_hex, 0o600));
    println!("pem:      {:?}", validate_key_bytes(pem, 0o600));
    println!("zeros:    {:?}", validate_key_bytes(&zeros, 0o600));
    println!("world ro: {:?}", validate_key_bytes(&happy, 0o644));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_32_byte_passes() {
        let v = validate_key_bytes(&[0xab; 32], 0o600);
        assert_eq!(v, vec![KeyVerdict::Ok]);
    }

    #[test]
    fn short_key_flagged() {
        let v = validate_key_bytes(&[0xab; 16], 0o600);
        assert!(v.iter().any(|x| matches!(
            x,
            KeyVerdict::WrongLength {
                observed: 16,
                expected: 32
            }
        )));
    }

    #[test]
    fn pem_pasted_as_key_detected() {
        // Common operator footgun: passing a PEM private key file.
        let v = validate_key_bytes(b"-----BEGIN PRIVATE KEY-----\n", 0o600);
        assert!(v.iter().any(|x| matches!(x, KeyVerdict::LooksLikePem)));
    }

    #[test]
    fn hex_string_paste_detected() {
        // 64 hex chars = 32 bytes encoded — operator probably forgot to decode.
        let hex = b"deadbeefcafebabe1122334455667788deadbeefcafebabe1122334455667788";
        let v = validate_key_bytes(hex, 0o600);
        assert!(v.iter().any(|x| matches!(x, KeyVerdict::LooksLikeHex)));
    }

    #[test]
    fn all_zeros_key_flagged() {
        // The all-zeros key works mathematically but is catastrophically weak.
        let v = validate_key_bytes(&[0u8; 32], 0o600);
        assert!(v.iter().any(|x| matches!(x, KeyVerdict::AllZeros)));
    }

    #[test]
    fn world_readable_mode_flagged() {
        // Mode 0o644 leaks the key to anyone with login access.
        let v = validate_key_bytes(&[0xab; 32], 0o644);
        assert!(v.iter().any(|x| matches!(x, KeyVerdict::WorldReadable)));
    }

    #[test]
    fn mode_600_does_not_trigger_warning() {
        // Owner-only read (mode 0o600) is the correct setting.
        let v = validate_key_bytes(&[0xab; 32], 0o600);
        assert!(!v.iter().any(|x| matches!(x, KeyVerdict::WorldReadable)));
    }
}
