//! # Recipe: Import — Multi-Format Pipeline (SafeTensors + GGUF + APR)
//!
//! **Category**: format
//! **CLI Equivalent**: `apr import --from safetensors --from gguf --from apr --to apr`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example import_multi_format` exits 0
//! 2. [x] `cargo test --example import_multi_format` passes
//! 3. [x] Deterministic output (fixed fixtures)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates multi-format import in-process (no shell-out)
//! 10. [x] Unit tests cover per-format detection, unknown, checksum
//!
//! ## Learning Objective
//! Implements a unified ingestion pipeline that accepts SafeTensors, GGUF, and
//! APR inputs, detects the source format from magic bytes, normalizes to an
//! internal representation, and emits a summary per file.
//!
//! ## Run Command
//! ```bash
//! cargo run --example import_multi_format
//! ```
//!
//! ## References
//! - Wolf, T. et al. (2020). *Transformers*. EMNLP demos. arXiv:1910.03771

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SourceFormat {
    Apr,
    Gguf,
    SafeTensors,
    Unknown,
}

impl SourceFormat {
    fn label(self) -> &'static str {
        match self {
            Self::Apr => "APR",
            Self::Gguf => "GGUF",
            Self::SafeTensors => "SAFETENSORS",
            Self::Unknown => "UNKNOWN",
        }
    }
}

#[derive(Debug, Clone)]
struct ImportEntry {
    name: String,
    source: SourceFormat,
    bytes: usize,
    digest_short: String,
}

/// Detect format by leading bytes. SafeTensors uses a u64 length-prefixed
/// JSON header; for this simulation we prefix with `ST_`.
fn detect_format(bytes: &[u8]) -> SourceFormat {
    if bytes.starts_with(b"APR1") || bytes.starts_with(b"APR2") {
        SourceFormat::Apr
    } else if bytes.starts_with(b"GGUF") {
        SourceFormat::Gguf
    } else if bytes.starts_with(b"ST_") {
        SourceFormat::SafeTensors
    } else {
        SourceFormat::Unknown
    }
}

fn synth_apr() -> Vec<u8> {
    let mut v = b"APR2".to_vec();
    v.extend_from_slice(&[0x01; 64]);
    v
}

fn synth_gguf() -> Vec<u8> {
    let mut v = b"GGUF".to_vec();
    v.extend_from_slice(&[0x02; 64]);
    v
}

fn synth_safetensors() -> Vec<u8> {
    let mut v = b"ST_".to_vec();
    v.extend_from_slice(b"{\"__metadata__\":{\"format\":\"pt\"}}");
    v.extend_from_slice(&[0x03; 32]);
    v
}

fn synth_unknown() -> Vec<u8> {
    b"RAND00".to_vec()
}

fn import_one(name: &str, bytes: &[u8]) -> ImportEntry {
    let fmt = detect_format(bytes);
    let digest = blake3::hash(bytes);
    ImportEntry {
        name: name.to_string(),
        source: fmt,
        bytes: bytes.len(),
        digest_short: digest.to_hex().as_str()[..16].to_string(),
    }
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("import_multi_format")?;
    println!("=== Recipe: {} ===", ctx.name());

    let files = [
        ("llama.apr", synth_apr()),
        ("mistral.gguf", synth_gguf()),
        ("phi.safetensors", synth_safetensors()),
        ("mystery.bin", synth_unknown()),
    ];

    let mut entries = Vec::new();
    println!("\n--- Multi-format import ---");
    for (name, bytes) in &files {
        let e = import_one(name, bytes);
        println!(
            "  {:<22} fmt={:<11} bytes={:>4} digest={:.8}...",
            e.name,
            e.source.label(),
            e.bytes,
            e.digest_short
        );
        entries.push(e);
    }
    let n_known = entries
        .iter()
        .filter(|e| e.source != SourceFormat::Unknown)
        .count();
    println!("\n{}/{} files recognized", n_known, entries.len());

    let report = json!({
        "recipe": ctx.name(),
        "n_files": entries.len(),
        "n_recognized": n_known,
        "entries": entries.iter().map(|e| json!({
            "name": e.name,
            "source": e.source.label(),
            "bytes": e.bytes,
            "digest": e.digest_short,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("import-multi.json");
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
    fn detects_apr_magic() {
        assert_eq!(detect_format(b"APR2\x01\x02"), SourceFormat::Apr);
        assert_eq!(detect_format(b"APR1\x01\x02"), SourceFormat::Apr);
    }

    #[test]
    fn detects_gguf_magic() {
        assert_eq!(detect_format(b"GGUF\x00\x00"), SourceFormat::Gguf);
    }

    #[test]
    fn detects_safetensors_prefix() {
        assert_eq!(detect_format(b"ST_{\"x\":1}"), SourceFormat::SafeTensors);
    }

    #[test]
    fn unknown_leading_bytes() {
        assert_eq!(detect_format(b"RAND00"), SourceFormat::Unknown);
    }

    #[test]
    fn empty_buffer_unknown() {
        assert_eq!(detect_format(&[]), SourceFormat::Unknown);
    }

    #[test]
    fn digest_deterministic() {
        let a = import_one("x", b"hello");
        let b = import_one("x", b"hello");
        assert_eq!(a.digest_short, b.digest_short);
    }
}
