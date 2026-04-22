//! # Recipe: Hex Pattern Search (Find Magic Bytes in Payload)
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr hex model.apr --find 0xDEADBEEF --offset 0`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example hex_pattern_search` exits 0
//! 2. [x] `cargo test --example hex_pattern_search` passes
//! 3. [x] Deterministic output (seeded bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr hex --find` in-process (no shell-out)
//! 10. [x] Unit tests cover hit, miss, overlap, empty pattern
//!
//! ## Learning Objective
//! Implements a naive byte-pattern search across a synthetic payload and
//! returns all match offsets. Use case: locating embedded magic bytes,
//! signature trailers, or compression headers inside an `.apr` archive.
//!
//! ## Run Command
//! ```bash
//! cargo run --example hex_pattern_search
//! ```
//!
//! ## References
//! - Casey, E. (2011). *Digital Evidence and Computer Crime*. 3rd ed. Academic Press.

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

/// Naive byte pattern search returning all (overlapping) match offsets.
fn find_all(haystack: &[u8], needle: &[u8]) -> Vec<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return Vec::new();
    }
    let mut out = Vec::new();
    for i in 0..=haystack.len() - needle.len() {
        if &haystack[i..i + needle.len()] == needle {
            out.push(i);
        }
    }
    out
}

fn synth_payload() -> Vec<u8> {
    let mut v = Vec::with_capacity(512);
    v.extend_from_slice(b"APR2");
    v.extend_from_slice(&[0; 60]);
    v.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
    v.extend_from_slice(&[0; 30]);
    v.extend_from_slice(b"LZ4\0");
    v.extend_from_slice(&[0; 40]);
    v.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
    v.extend_from_slice(&[0; 200]);
    // Signature trailer.
    v.extend_from_slice(b"SIG0");
    v.extend_from_slice(&[0x42; 64]);
    v
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("hex_pattern_search")?;
    println!("=== Recipe: {} ===", ctx.name());

    let payload = synth_payload();

    let patterns: Vec<(&str, Vec<u8>)> = vec![
        ("APR2", b"APR2".to_vec()),
        ("LZ4\\0", b"LZ4\0".to_vec()),
        ("SIG0", b"SIG0".to_vec()),
        ("DEADBEEF", vec![0xDE, 0xAD, 0xBE, 0xEF]),
        ("ABSENT", vec![0x12, 0x34]),
    ];

    let mut hits = Vec::new();
    println!("\n--- Matches ---");
    for (label, pat) in &patterns {
        let locs = find_all(&payload, pat);
        println!(
            "  pattern={:<10} matches={} offsets={:?}",
            label,
            locs.len(),
            locs
        );
        hits.push(((*label).to_string(), locs));
    }

    let report = json!({
        "recipe": ctx.name(),
        "payload_len": payload.len(),
        "results": hits.iter().map(|(label, locs)| json!({
            "pattern": label,
            "n_matches": locs.len(),
            "offsets": locs,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("hex-pattern.json");
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out, bytes)?;

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_pattern_returns_empty() {
        assert!(find_all(b"abc", b"").is_empty());
    }

    #[test]
    fn pattern_longer_than_haystack() {
        assert!(find_all(b"ab", b"abc").is_empty());
    }

    #[test]
    fn multiple_hits() {
        let hits = find_all(b"aXaXaX", b"aX");
        assert_eq!(hits, vec![0, 2, 4]);
    }

    #[test]
    fn overlapping_hits() {
        let hits = find_all(b"aaaa", b"aa");
        assert_eq!(hits, vec![0, 1, 2]);
    }

    #[test]
    fn synth_has_two_deadbeef() {
        let p = synth_payload();
        let hits = find_all(&p, &[0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn synth_starts_with_apr2() {
        let p = synth_payload();
        let hits = find_all(&p, b"APR2");
        assert_eq!(hits[0], 0);
    }
}
