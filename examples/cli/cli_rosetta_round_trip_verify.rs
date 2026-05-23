//! # apr rosetta — Round-Trip Verify
//!
//! `apr rosetta verify <FROM> <TO>` confirms a conversion preserves
//! tensor data: convert FROM→TO→FROM' and check that FROM' == FROM
//! byte-for-byte (or within tolerance for lossy formats). This recipe
//! models the verifier with a hash-then-compare pure function so the
//! contract can be exercised offline.
//!
//! Demonstrates the **ROSETTA.4** recipe for PMAT-094 (apr rosetta coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-ROSETTA-001 + blake3 (round-trip identity)
//!
//! Run with: cargo run --example cli_rosetta_round_trip_verify
//!
//! Added by PMAT-094 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use blake3::Hasher;

#[derive(Debug, Clone, PartialEq)]
pub enum VerifyVerdict {
    BitIdentical,
    Lossy { divergence_bytes: usize },
}

pub fn round_trip_verify(original: &[u8], roundtripped: &[u8]) -> VerifyVerdict {
    let mut h1 = Hasher::new();
    h1.update(original);
    let hash_orig = *h1.finalize().as_bytes();

    let mut h2 = Hasher::new();
    h2.update(roundtripped);
    let hash_rt = *h2.finalize().as_bytes();

    if hash_orig == hash_rt {
        return VerifyVerdict::BitIdentical;
    }
    let n = original.len().min(roundtripped.len());
    let differing = (0..n).filter(|i| original[*i] != roundtripped[*i]).count()
        + original.len().abs_diff(roundtripped.len());
    VerifyVerdict::Lossy {
        divergence_bytes: differing,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_round_trip_verify")?;

    let original: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
    let identical = original.clone();
    let mut tweaked = original.clone();
    tweaked[7] ^= 0x40; // flip a bit
    let mut truncated = original.clone();
    truncated.truncate(1000);

    println!(
        "identical:   {:?}",
        round_trip_verify(&original, &identical)
    );
    println!("one bit:     {:?}", round_trip_verify(&original, &tweaked));
    println!(
        "truncated:   {:?}",
        round_trip_verify(&original, &truncated)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn identical_bytes_yield_bit_identical() {
        let v = round_trip_verify(b"hello", b"hello");
        assert_eq!(v, VerifyVerdict::BitIdentical);
    }

    #[test]
    fn single_byte_change_is_lossy() {
        let v = round_trip_verify(b"hello", b"hellp");
        assert_eq!(
            v,
            VerifyVerdict::Lossy {
                divergence_bytes: 1
            }
        );
    }

    #[test]
    fn truncation_counted_in_divergence() {
        let v = round_trip_verify(b"hello", b"hel");
        // 0 differing in overlap + 2 missing bytes = 2 total divergence.
        assert_eq!(
            v,
            VerifyVerdict::Lossy {
                divergence_bytes: 2
            }
        );
    }

    #[test]
    fn extension_counted_in_divergence() {
        // Symmetric: more bytes in roundtrip than original.
        let v = round_trip_verify(b"hi", b"hill");
        assert_eq!(
            v,
            VerifyVerdict::Lossy {
                divergence_bytes: 2
            }
        );
    }

    #[test]
    fn empty_inputs_are_bit_identical() {
        let v = round_trip_verify(&[], &[]);
        assert_eq!(v, VerifyVerdict::BitIdentical);
    }

    #[test]
    fn large_buffer_uses_blake3_hash_path() {
        // The hash-first path avoids byte-by-byte comparison for the common
        // (clean) case. This test confirms equality holds for a 64KB buffer.
        let buf: Vec<u8> = (0..65_536).map(|i| (i % 256) as u8).collect();
        let v = round_trip_verify(&buf, &buf);
        assert_eq!(v, VerifyVerdict::BitIdentical);
    }
}
