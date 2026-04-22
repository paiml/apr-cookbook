//! # Recipe: Hex Diff Between Two Model Headers
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr hex --diff model_v1.apr model_v2.apr --bytes 256`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example hex_diff` exits 0
//! 2. [x] `cargo test --example hex_diff` passes
//! 3. [x] Deterministic output (seeded bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr hex --diff` in-process (no shell-out)
//! 10. [x] Unit tests cover identity, point-change, length-mismatch, offset
//!
//! ## Learning Objective
//! Performs byte-level diff between two binary headers, emitting per-byte
//! differences and a summary suitable for forensic inspection of `.apr` file
//! changes. Highlights offsets of changed bytes and runs of equal bytes.
//!
//! ## Run Command
//! ```bash
//! cargo run --example hex_diff
//! ```
//!
//! ## References
//! - Casey, E. (2011). *Digital Evidence and Computer Crime*. 3rd ed. Academic Press.

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone)]
struct HexDiffEntry {
    offset: usize,
    from: u8,
    to: u8,
}

#[derive(Debug, Clone, Default)]
struct HexDiffSummary {
    entries: Vec<HexDiffEntry>,
    n_equal: usize,
    n_changed: usize,
    n_only_in_a: usize,
    n_only_in_b: usize,
}

fn hex_diff(a: &[u8], b: &[u8]) -> HexDiffSummary {
    let min = a.len().min(b.len());
    let mut entries = Vec::new();
    let mut n_equal = 0;
    let mut n_changed = 0;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate().take(min) {
        if x == y {
            n_equal += 1;
        } else {
            n_changed += 1;
            entries.push(HexDiffEntry {
                offset: i,
                from: *x,
                to: *y,
            });
        }
    }
    let (n_only_in_a, n_only_in_b) = if a.len() > b.len() {
        (a.len() - b.len(), 0)
    } else {
        (0, b.len() - a.len())
    };
    HexDiffSummary {
        entries,
        n_equal,
        n_changed,
        n_only_in_a,
        n_only_in_b,
    }
}

fn synth_header(seed: u64, n: usize) -> Vec<u8> {
    // Deterministic synthetic header: cycle of bytes with prefix magic.
    let mut v = b"APR2".to_vec();
    for i in 0..n.saturating_sub(4) {
        let b = ((seed as usize + i * 7) % 256) as u8;
        v.push(b);
    }
    v
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("hex_diff")?;
    println!("=== Recipe: {} ===", ctx.name());

    let a = synth_header(1234, 64);
    let mut b = a.clone();
    // Simulate three header changes: magic version bump + 2 metadata bytes.
    b[3] = b'3';
    b[10] = b[10].wrapping_add(17);
    b[42] = 0xFF;

    let summary = hex_diff(&a, &b);
    println!(
        "\n--- Hex diff summary ---\nequal={} changed={} only_in_a={} only_in_b={}",
        summary.n_equal, summary.n_changed, summary.n_only_in_a, summary.n_only_in_b
    );
    println!("\nChanged bytes:");
    println!("{:>8} {:>6} {:>6}", "offset", "from", "to");
    for e in &summary.entries {
        println!("{:>8} {:>#06x} {:>#06x}", e.offset, e.from, e.to);
    }

    let report = json!({
        "recipe": ctx.name(),
        "len_a": a.len(),
        "len_b": b.len(),
        "summary": {
            "n_equal": summary.n_equal,
            "n_changed": summary.n_changed,
            "n_only_in_a": summary.n_only_in_a,
            "n_only_in_b": summary.n_only_in_b,
        },
        "entries": summary.entries.iter().map(|e| json!({
            "offset": e.offset,
            "from": e.from,
            "to": e.to,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("hex-diff.json");
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
    fn identical_buffers_no_changes() {
        let s = hex_diff(b"abc", b"abc");
        assert_eq!(s.n_changed, 0);
        assert_eq!(s.n_equal, 3);
    }

    #[test]
    fn single_byte_change_detected() {
        let s = hex_diff(b"abc", b"aXc");
        assert_eq!(s.n_changed, 1);
        assert_eq!(s.entries.len(), 1);
        assert_eq!(s.entries[0].offset, 1);
    }

    #[test]
    fn extra_bytes_in_a() {
        let s = hex_diff(b"abcXY", b"abc");
        assert_eq!(s.n_only_in_a, 2);
        assert_eq!(s.n_only_in_b, 0);
    }

    #[test]
    fn extra_bytes_in_b() {
        let s = hex_diff(b"abc", b"abcXY");
        assert_eq!(s.n_only_in_b, 2);
        assert_eq!(s.n_only_in_a, 0);
    }

    #[test]
    fn synth_header_preserves_magic() {
        let v = synth_header(0, 16);
        assert_eq!(&v[..4], b"APR2");
    }
}
