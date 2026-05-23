//! # API Request-ID Correlation
//!
//! Distributed APIs propagate a Request-ID across logs, traces, and
//! downstream services. UUIDv4 (8-4-4-4-12 hex) is the convention.
//! When a header arrives, validate format; when missing, generate.
//! This recipe builds the validator + a deterministic test generator.
//!
//! Demonstrates the **API.6** recipe for PMAT-125 (api coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: RFC 4122 (UUIDv4); W3C Trace Context.
//!
//! Run with: cargo run --example api_request_id_correlation
//!
//! Added by PMAT-125 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum IdVerdict {
    Ok,
    WrongLength { len: usize },
    BadHyphenPositions,
    NonHexCharacter { at: usize, ch: char },
    BadVersion { got: char },
    BadVariant { got: char },
}

pub fn validate(id: &str) -> IdVerdict {
    if id.len() != 36 {
        return IdVerdict::WrongLength { len: id.len() };
    }
    let bytes = id.as_bytes();
    for (idx, expect) in [(8usize, b'-'), (13, b'-'), (18, b'-'), (23, b'-')] {
        if bytes[idx] != expect {
            return IdVerdict::BadHyphenPositions;
        }
    }
    for (i, c) in id.char_indices() {
        if matches!(i, 8 | 13 | 18 | 23) {
            continue;
        }
        if !c.is_ascii_hexdigit() {
            return IdVerdict::NonHexCharacter { at: i, ch: c };
        }
    }
    let version_char = bytes[14] as char;
    if version_char != '4' {
        return IdVerdict::BadVersion { got: version_char };
    }
    let variant_char = bytes[19] as char;
    if !matches!(variant_char, '8' | '9' | 'a' | 'b' | 'A' | 'B') {
        return IdVerdict::BadVariant { got: variant_char };
    }
    IdVerdict::Ok
}

pub fn generate_deterministic(seed: u64) -> String {
    // Build a UUIDv4 from a seed (test-only — NOT cryptographically secure).
    // Use SplitMix64 to derive 16 bytes.
    let mut state = seed;
    let mut bytes = [0u8; 16];
    for chunk in bytes.chunks_mut(8) {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        chunk.copy_from_slice(&z.to_le_bytes()[..chunk.len()]);
    }
    bytes[6] = (bytes[6] & 0x0F) | 0x40; // version = 4
    bytes[8] = (bytes[8] & 0x3F) | 0x80; // variant = 10
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6], bytes[7],
        bytes[8], bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    )
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("api_request_id_correlation")?;

    for s in [
        "550e8400-e29b-41d4-a716-446655440000",
        "00000000-0000-4000-8000-000000000000",
        "550e8400e29b41d4a716446655440000",
        "550e8400-e29b-11d4-a716-446655440000",
    ] {
        println!("{s} → {:?}", validate(s));
    }
    let id = generate_deterministic(0x42);
    println!("generated: {id}  →  {:?}", validate(&id));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correlator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_v4_id_passes() {
        assert_eq!(
            validate("550e8400-e29b-41d4-a716-446655440000"),
            IdVerdict::Ok
        );
    }

    #[test]
    fn wrong_length_rejected() {
        assert_eq!(
            validate("550e8400e29b41d4a716446655440000"),
            IdVerdict::WrongLength { len: 32 }
        );
    }

    #[test]
    fn bad_version_rejected() {
        // Version byte (idx 14) is '1' instead of '4'.
        let v = validate("550e8400-e29b-11d4-a716-446655440000");
        assert!(matches!(v, IdVerdict::BadVersion { got: '1' }));
    }

    #[test]
    fn bad_variant_rejected() {
        // Variant byte (idx 19) is 'c' instead of 8/9/a/b.
        let v = validate("550e8400-e29b-41d4-c716-446655440000");
        assert!(matches!(v, IdVerdict::BadVariant { got: 'c' }));
    }

    #[test]
    fn non_hex_rejected() {
        let v = validate("550e8400-e29b-41d4-a716-44665544000z");
        assert!(matches!(v, IdVerdict::NonHexCharacter { ch: 'z', .. }));
    }

    #[test]
    fn bad_hyphen_position_rejected() {
        // Hyphen missing at position 8.
        let v = validate("550e8400_e29b-41d4-a716-446655440000");
        assert_eq!(v, IdVerdict::BadHyphenPositions);
    }

    #[test]
    fn deterministic_generator_is_valid() {
        let id = generate_deterministic(0x42);
        assert_eq!(validate(&id), IdVerdict::Ok);
    }

    #[test]
    fn deterministic_generator_stable_across_calls() {
        let a = generate_deterministic(0x42);
        let b = generate_deterministic(0x42);
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_yield_different_ids() {
        let a = generate_deterministic(0x42);
        let b = generate_deterministic(0x43);
        assert_ne!(a, b);
    }

    #[test]
    fn nil_uuid_v4_form_passes() {
        // All zeroes is technically the "nil" UUID — version 0, not 4.
        // Our validator requires version 4, so this should fail.
        let v = validate("00000000-0000-0000-0000-000000000000");
        assert!(matches!(v, IdVerdict::BadVersion { .. }));
    }
}
