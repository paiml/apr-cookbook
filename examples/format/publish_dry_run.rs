//! # Recipe: HF-Hub Publish Pipeline Dry-Run with Manifest Generation
//!
//! **Category**: format
//! **CLI Equivalent**: `apr publish model.apr --target huggingface --dry-run`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example publish_dry_run` exits 0
//! 2. [x] `cargo test --example publish_dry_run` passes
//! 3. [x] Deterministic output (seeded RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr publish --dry-run` in-process (no shell-out)
//! 10. [x] Unit tests cover manifest structure, file list, hash computation
//!
//! ## Learning Objective
//! Demonstrates a dry-run publish pipeline: enumerates files that would be
//! uploaded, computes per-file blake3 hashes, builds a HF-style manifest
//! (model card + config + weights), and prints the plan without touching the
//! network. Exactly what `apr publish --dry-run` does before a real upload.
//!
//! ## Run Command
//! ```bash
//! cargo run --example publish_dry_run
//! ```
//!
//! ## References
//! - Amershi, S. et al. (2019). *Software Engineering for Machine Learning: A Case Study*. ICSE-SEIP. DOI: 10.1109/ICSE-SEIP.2019.00042

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublishFile {
    pub path: String,
    pub size_bytes: u64,
    pub content_hash: String,
}

#[derive(Debug, Clone)]
pub struct PublishManifest {
    pub target: String,
    pub repo: String,
    pub files: Vec<PublishFile>,
    pub total_bytes: u64,
    pub dry_run: bool,
}

pub fn hash_bytes(b: &[u8]) -> String {
    blake3::hash(b).to_hex().to_string()
}

pub fn build_manifest(
    target: &str,
    repo: &str,
    files: Vec<(&str, Vec<u8>)>,
    dry_run: bool,
) -> PublishManifest {
    let mut entries = Vec::new();
    let mut total: u64 = 0;
    for (path, bytes) in files {
        total += bytes.len() as u64;
        entries.push(PublishFile {
            path: path.to_string(),
            size_bytes: bytes.len() as u64,
            content_hash: hash_bytes(&bytes),
        });
    }
    PublishManifest {
        target: target.into(),
        repo: repo.into(),
        files: entries,
        total_bytes: total,
        dry_run,
    }
}

fn demo_files() -> Vec<(&'static str, Vec<u8>)> {
    vec![
        ("README.md", b"# model card\n".to_vec()),
        (
            "config.json",
            b"{\"architectures\": [\"LlamaForCausalLM\"]}".to_vec(),
        ),
        ("model.safetensors", vec![0xAB; 4096]),
        ("tokenizer.json", b"{\"type\": \"BPE\"}".to_vec()),
    ]
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("publish_dry_run")?;
    println!("=== Recipe: {} ===", ctx.name());

    let manifest = build_manifest("huggingface.co", "paiml/tiny-demo", demo_files(), true);
    println!(
        "Target: {}  Repo: {}  Files: {}  Total: {} bytes  DryRun: {}",
        manifest.target,
        manifest.repo,
        manifest.files.len(),
        manifest.total_bytes,
        manifest.dry_run
    );
    for f in &manifest.files {
        println!(
            "  {:<28} {:>6} bytes  sha256={}",
            f.path,
            f.size_bytes,
            &f.content_hash[..16]
        );
    }

    let report = json!({
        "recipe": ctx.name(),
        "target": manifest.target,
        "repo": manifest.repo,
        "dry_run": manifest.dry_run,
        "total_bytes": manifest.total_bytes,
        "files": manifest.files.iter().map(|f| json!({
            "path": f.path,
            "size_bytes": f.size_bytes,
            "content_hash": f.content_hash,
        })).collect::<Vec<_>>(),
    });
    let path = ctx.path("publish-dry-run.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    ctx.record_metric("files", manifest.files.len() as i64);
    ctx.record_metric("total_bytes", manifest.total_bytes as i64);
    ctx.record_string_metric("mode", if manifest.dry_run { "dry-run" } else { "live" });
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic() {
        let h1 = hash_bytes(b"hello");
        let h2 = hash_bytes(b"hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn different_bytes_different_hash() {
        assert_ne!(hash_bytes(b"a"), hash_bytes(b"b"));
    }

    #[test]
    fn manifest_counts_files() {
        let m = build_manifest(
            "hf.co",
            "u/r",
            vec![("a.txt", vec![0; 10]), ("b.txt", vec![0; 20])],
            true,
        );
        assert_eq!(m.files.len(), 2);
        assert_eq!(m.total_bytes, 30);
    }

    #[test]
    fn dry_run_flag_preserved() {
        let m = build_manifest("hf.co", "u/r", vec![], false);
        assert!(!m.dry_run);
    }

    #[test]
    fn each_file_has_nonempty_hash() {
        let m = build_manifest("hf.co", "u/r", demo_files(), true);
        assert!(m.files.iter().all(|f| !f.content_hash.is_empty()));
    }
}
