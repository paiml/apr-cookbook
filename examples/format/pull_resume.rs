//! # Recipe: Pull with Resume-from-Offset
//!
//! **Category**: format
//! **CLI Equivalent**: `apr pull model --resume --chunk-size 64KB`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example pull_resume` exits 0
//! 2. [x] `cargo test --example pull_resume` passes
//! 3. [x] Deterministic output (seeded RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr pull --resume` in-process (no shell-out)
//! 10. [x] Unit tests cover resume offset, chunk boundary, full pull, over-read rejection
//!
//! ## Learning Objective
//! Demonstrates a resumable download: chunks a source blob into 64 KiB ranges,
//! simulates a partial download that stops at an arbitrary offset, then resumes
//! from the on-disk offset and verifies the final byte count + digest match
//! the source. Mirrors the rsync-style resume `apr pull --resume` uses.
//!
//! ## Run Command
//! ```bash
//! cargo run --example pull_resume
//! ```
//!
//! ## References
//! - Tridgell, A. & Mackerras, P. (1996). *The rsync algorithm*. Tech report TR-CS-96-05.

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use rand::RngCore;
use serde_json::json;

const CHUNK: usize = 64 * 1024;

/// A resumable byte source: callers can ask for `[offset, offset+len)`.
#[derive(Debug, Clone)]
pub struct ByteSource {
    pub bytes: Vec<u8>,
}

impl ByteSource {
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Read a range, returning `None` if out of bounds.
    pub fn read_range(&self, offset: usize, len: usize) -> Option<&[u8]> {
        let end = offset.checked_add(len)?;
        if end > self.bytes.len() {
            return None;
        }
        Some(&self.bytes[offset..end])
    }

    pub fn hash(&self) -> String {
        blake3::hash(&self.bytes).to_hex().to_string()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PullStats {
    pub resumed_from: usize,
    pub chunks_downloaded: usize,
    pub bytes_downloaded: usize,
    pub total_bytes: usize,
}

pub fn pull_resume(
    src: &ByteSource,
    start_offset: usize,
    chunk_size: usize,
    sink: &mut Vec<u8>,
) -> std::result::Result<PullStats, String> {
    if chunk_size == 0 {
        return Err("chunk_size must be > 0".into());
    }
    if start_offset > src.len() {
        return Err("start_offset beyond source size".into());
    }
    sink.truncate(start_offset);
    let mut cursor = start_offset;
    let mut chunks = 0usize;
    while cursor < src.len() {
        let remaining = src.len() - cursor;
        let take = remaining.min(chunk_size);
        match src.read_range(cursor, take) {
            Some(slice) => sink.extend_from_slice(slice),
            None => return Err(format!("read_range failed at offset {}", cursor)),
        }
        cursor += take;
        chunks += 1;
    }
    Ok(PullStats {
        resumed_from: start_offset,
        chunks_downloaded: chunks,
        bytes_downloaded: sink.len() - start_offset,
        total_bytes: src.len(),
    })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("pull_resume")?;
    println!("=== Recipe: {} ===", ctx.name());

    // Synthesize a 512 KiB payload.
    let mut bytes = vec![0u8; 512 * 1024];
    ctx.rng().fill_bytes(&mut bytes);
    let src = ByteSource { bytes };
    let full_hash = src.hash();

    // Phase 1: partial download (stops 73% through).
    let partial_end = (src.len() as f64 * 0.73) as usize;
    let mut sink = Vec::with_capacity(src.len());
    let first = pull_resume(&src, 0, CHUNK, &mut sink).map_err(CookbookError::Serialization)?;
    // Simulate "stop early": truncate sink back to partial_end.
    sink.truncate(partial_end);

    // Phase 2: resume from on-disk offset.
    let second =
        pull_resume(&src, sink.len(), CHUNK, &mut sink).map_err(CookbookError::Serialization)?;

    let sink_src = ByteSource { bytes: sink };
    let ok = sink_src.hash() == full_hash;
    println!(
        "Phase 1: {} chunks @ {} bytes  (then stopped at {} bytes)",
        first.chunks_downloaded, first.bytes_downloaded, partial_end
    );
    println!(
        "Phase 2: resumed from {} and pulled {} chunks / {} bytes",
        second.resumed_from, second.chunks_downloaded, second.bytes_downloaded
    );
    println!("Final hash match: {}  ({} total bytes)", ok, sink_src.len());

    let report = json!({
        "recipe": ctx.name(),
        "chunk_size": CHUNK,
        "total_bytes": src.len(),
        "partial_end": partial_end,
        "phase_1_chunks": first.chunks_downloaded,
        "phase_2_chunks": second.chunks_downloaded,
        "hash_match": ok,
        "source_hash": full_hash,
    });
    let path = ctx.path("pull-resume.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    ctx.record_metric("chunks_phase_2", second.chunks_downloaded as i64);
    ctx.record_metric("total_bytes", src.len() as i64);
    ctx.record_string_metric("hash_match", if ok { "true" } else { "false" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn src(n: usize) -> ByteSource {
        ByteSource {
            bytes: (0..n).map(|i| (i % 251) as u8).collect(),
        }
    }

    #[test]
    fn full_pull_from_zero() {
        let s = src(1000);
        let mut sink = Vec::new();
        let stats = pull_resume(&s, 0, 128, &mut sink).expect("ok");
        assert_eq!(sink.len(), s.len());
        assert_eq!(stats.bytes_downloaded, s.len());
    }

    #[test]
    fn resume_midway_completes() {
        let s = src(1000);
        let mut sink = Vec::new();
        pull_resume(&s, 0, 128, &mut sink).expect("ok");
        sink.truncate(600);
        let stats = pull_resume(&s, 600, 128, &mut sink).expect("ok");
        assert_eq!(sink.len(), s.len());
        assert_eq!(stats.bytes_downloaded, 400);
        assert_eq!(stats.resumed_from, 600);
    }

    #[test]
    fn offset_beyond_size_errors() {
        let s = src(100);
        let mut sink = Vec::new();
        assert!(pull_resume(&s, 200, 64, &mut sink).is_err());
    }

    #[test]
    fn zero_chunk_size_errors() {
        let s = src(100);
        let mut sink = Vec::new();
        assert!(pull_resume(&s, 0, 0, &mut sink).is_err());
    }

    #[test]
    fn chunk_boundary_even_and_odd() {
        let s = src(1000);
        for chunk in [1usize, 7, 100, 999, 1000, 2000] {
            let mut sink = Vec::new();
            pull_resume(&s, 0, chunk, &mut sink).expect("ok");
            assert_eq!(sink.len(), s.len());
        }
    }
}
