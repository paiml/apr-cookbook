//! # apr encrypt — Keystream + MAC Envelope
//!
//! `apr encrypt` produces an `.enc` envelope: 8-byte magic "APRENC01"
//! prefix, 12-byte nonce, 32-byte BLAKE3-MAC tag, then the ciphertext.
//! Total overhead is constant 52 bytes regardless of plaintext size.
//! This recipe builds the envelope synthesizer + parser as pure
//! functions and asserts the byte layout.
//!
//! Demonstrates the **ENCRYPT.5** recipe for PMAT-103 (apr encrypt coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SHIP-009 + BLAKE3 keyed hash MAC
//!
//! Run with: cargo run --example cli_encrypt_keystream_envelope
//!
//! Added by PMAT-103 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MAGIC: &[u8] = b"APRENC01";
const NONCE_LEN: usize = 12;
const MAC_LEN: usize = 32;
pub const HEADER_OVERHEAD: usize = MAGIC.len() + NONCE_LEN + MAC_LEN;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncEnvelope<'a> {
    pub magic: &'a [u8; 8],
    pub nonce: &'a [u8; 12],
    pub mac: &'a [u8; 32],
    pub ciphertext: &'a [u8],
}

#[derive(Debug, PartialEq)]
pub enum ParseVerdict {
    Ok { ciphertext_len: usize },
    BadMagic,
    Truncated,
}

pub fn synthesize_envelope(nonce: &[u8; 12], mac: &[u8; 32], ciphertext: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_OVERHEAD + ciphertext.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(nonce);
    out.extend_from_slice(mac);
    out.extend_from_slice(ciphertext);
    out
}

pub fn parse_envelope(bytes: &[u8]) -> ParseVerdict {
    if bytes.len() < HEADER_OVERHEAD {
        return ParseVerdict::Truncated;
    }
    if &bytes[..MAGIC.len()] != MAGIC {
        return ParseVerdict::BadMagic;
    }
    ParseVerdict::Ok {
        ciphertext_len: bytes.len() - HEADER_OVERHEAD,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_encrypt_keystream_envelope")?;

    let nonce = [0x42u8; 12];
    let mac = [0xcdu8; 32];
    let ct: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    let env = synthesize_envelope(&nonce, &mac, &ct);
    println!(
        "envelope bytes: {} ({} overhead + {} ciphertext)",
        env.len(),
        HEADER_OVERHEAD,
        ct.len()
    );
    println!("parse: {:?}", parse_envelope(&env));
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
    fn header_overhead_is_52_bytes() {
        // 8 magic + 12 nonce + 32 mac = 52.
        assert_eq!(HEADER_OVERHEAD, 52);
    }

    #[test]
    fn synthesize_then_parse_round_trips() {
        let nonce = [1u8; 12];
        let mac = [2u8; 32];
        let ct = b"hello world";
        let env = synthesize_envelope(&nonce, &mac, ct);
        match parse_envelope(&env) {
            ParseVerdict::Ok { ciphertext_len } => assert_eq!(ciphertext_len, ct.len()),
            v => panic!("expected Ok, got {v:?}"),
        }
    }

    #[test]
    fn empty_ciphertext_round_trips() {
        let nonce = [0u8; 12];
        let mac = [0u8; 32];
        let env = synthesize_envelope(&nonce, &mac, &[]);
        assert_eq!(env.len(), HEADER_OVERHEAD);
        match parse_envelope(&env) {
            ParseVerdict::Ok { ciphertext_len } => assert_eq!(ciphertext_len, 0),
            v => panic!("expected Ok with len 0, got {v:?}"),
        }
    }

    #[test]
    fn bad_magic_rejected() {
        let mut env = synthesize_envelope(&[0u8; 12], &[0u8; 32], b"x");
        env[0] = b'X';
        assert_eq!(parse_envelope(&env), ParseVerdict::BadMagic);
    }

    #[test]
    fn truncated_below_header_rejected() {
        // < 52 bytes can't possibly be a valid envelope.
        assert_eq!(parse_envelope(&[0u8; 10]), ParseVerdict::Truncated);
        assert_eq!(parse_envelope(&[0u8; 51]), ParseVerdict::Truncated);
    }

    #[test]
    fn boundary_at_exactly_52_bytes_parses() {
        // 52 bytes = header only, 0 ciphertext.
        let nonce = [0u8; 12];
        let mac = [0u8; 32];
        let env = synthesize_envelope(&nonce, &mac, &[]);
        assert_eq!(env.len(), HEADER_OVERHEAD);
        assert!(matches!(
            parse_envelope(&env),
            ParseVerdict::Ok { ciphertext_len: 0 }
        ));
    }

    #[test]
    fn synthesize_emits_magic_at_offset_0() {
        let env = synthesize_envelope(&[0u8; 12], &[0u8; 32], b"x");
        assert_eq!(&env[..MAGIC.len()], MAGIC);
    }
}
