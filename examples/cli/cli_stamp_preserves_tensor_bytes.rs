//! # apr stamp — Tensor Bytes Preserved Verbatim
//!
//! `apr stamp` is GUARANTEED to preserve tensor bytes exactly. Only the
//! provenance fields in the APR v2 header change; not a single tensor byte
//! moves. This recipe demonstrates the byte-identity invariant by writing
//! a synthetic APR v2 buffer, computing a hash of the tensor section,
//! "stamping" only the provenance bytes, and asserting the hash is
//! preserved.
//!
//! Demonstrates the **STAMP.2** recipe for PMAT-088 (apr stamp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SHIP-009 + APR v2 format spec
//!
//! Run with: cargo run --example cli_stamp_preserves_tensor_bytes
//!
//! Added by PMAT-088 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use blake3::Hasher;

/// Synthetic APR-shaped buffer: [magic 4B][provenance 64B][tensor section ...]
fn build_apr_buffer(provenance: &[u8; 64], tensor_payload: &[u8]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4 + 64 + tensor_payload.len());
    buf.extend_from_slice(b"APR2");
    buf.extend_from_slice(provenance);
    buf.extend_from_slice(tensor_payload);
    buf
}

/// Stamp new provenance into an APR buffer; tensor bytes preserved verbatim.
fn stamp_provenance(buf: &[u8], new_provenance: &[u8; 64]) -> Vec<u8> {
    assert!(buf.len() >= 68, "buffer too small to stamp");
    assert_eq!(&buf[..4], b"APR2");
    let mut out = Vec::with_capacity(buf.len());
    out.extend_from_slice(&buf[..4]);
    out.extend_from_slice(new_provenance);
    out.extend_from_slice(&buf[68..]);
    out
}

fn hash_tensor_section(buf: &[u8]) -> [u8; 32] {
    let mut h = Hasher::new();
    h.update(&buf[68..]);
    *h.finalize().as_bytes()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_stamp_preserves_tensor_bytes")?;

    let original_provenance = [b' '; 64];
    let new_provenance = {
        let mut p = [b' '; 64];
        p[..15].copy_from_slice(b"Apache-2.0     ");
        p
    };
    let tensor_payload: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();

    let original = build_apr_buffer(&original_provenance, &tensor_payload);
    let stamped = stamp_provenance(&original, &new_provenance);

    let original_hash = hash_tensor_section(&original);
    let stamped_hash = hash_tensor_section(&stamped);

    let to_hex = |bytes: &[u8]| -> String {
        use std::fmt::Write as _;
        bytes
            .iter()
            .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
                let _ = write!(s, "{b:02x}");
                s
            })
    };
    println!(
        "tensor section hash before stamp:  {}",
        to_hex(&original_hash[..8])
    );
    println!(
        "tensor section hash after stamp:   {}",
        to_hex(&stamped_hash[..8])
    );
    println!("preserved verbatim: {}", original_hash == stamped_hash);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preservation_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn stamp_preserves_tensor_section_hash() {
        let provenance_a = [b' '; 64];
        let provenance_b = [b'X'; 64];
        let payload: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        let original = build_apr_buffer(&provenance_a, &payload);
        let stamped = stamp_provenance(&original, &provenance_b);
        assert_eq!(
            hash_tensor_section(&original),
            hash_tensor_section(&stamped),
            "FALSIFY: stamp must preserve tensor bytes verbatim"
        );
    }

    #[test]
    fn stamp_changes_provenance_section() {
        let provenance_a = [b'A'; 64];
        let provenance_b = [b'B'; 64];
        let payload = b"tensor".to_vec();
        let original = build_apr_buffer(&provenance_a, &payload);
        let stamped = stamp_provenance(&original, &provenance_b);
        assert_ne!(&original[4..68], &stamped[4..68]);
    }

    #[test]
    fn stamp_preserves_apr2_magic() {
        let provenance = [b' '; 64];
        let payload = b"tensor".to_vec();
        let original = build_apr_buffer(&provenance, &payload);
        let stamped = stamp_provenance(&original, &[b'X'; 64]);
        assert_eq!(&stamped[..4], b"APR2");
    }
}
