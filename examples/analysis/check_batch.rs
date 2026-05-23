//! # Recipe: Batch Health-Check Across a Model Registry
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr check --registry ./models/ --batch --fail-fast=false`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example check_batch` exits 0
//! 2. [x] `cargo test --example check_batch` passes
//! 3. [x] Deterministic output (same seed -> same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr check` batch behavior in-process (no shell-out)
//! 10. [x] Unit tests cover good / bad / corrupted models + aggregation
//!
//! ## Learning Objective
//! Runs a health check across a synthetic "registry" of 6 models -- some valid,
//! some with size/magic-byte regressions -- and produces an aggregate report
//! showing pass/warn/fail counts. This is the batch analog of `apr check`.
//!
//! ## Run Command
//! ```bash
//! cargo run --example check_batch
//! ```
//!
//! ## References
//! - Amershi, S. et al. (2019). *Software Engineering for Machine Learning: A Case Study*. ICSE-SEIP. DOI: 10.1109/ICSE-SEIP.2019.00042

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CheckStatus {
    Pass,
    Warn,
    Fail,
}

impl CheckStatus {
    fn label(self) -> &'static str {
        match self {
            Self::Pass => "PASS",
            Self::Warn => "WARN",
            Self::Fail => "FAIL",
        }
    }
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CheckEntry {
    name: String,
    path: PathBuf,
    size_bytes: usize,
    magic_ok: bool,
    status: CheckStatus,
    notes: Vec<String>,
}

#[derive(Debug, Clone)]
struct BatchReport {
    pass: usize,
    warn: usize,
    fail: usize,
    entries: Vec<CheckEntry>,
}

// ---------------------------------------------------------------------------
// Check logic
// ---------------------------------------------------------------------------

fn check_one(path: &std::path::Path, data: &[u8]) -> CheckEntry {
    let mut notes = Vec::new();
    let magic_ok = data.len() >= 4 && &data[0..4] == b"APR2";
    if !magic_ok {
        notes.push("magic bytes mismatch".to_string());
    }
    let size_bytes = data.len();
    if size_bytes < 64 {
        notes.push(format!("model too small: {} bytes", size_bytes));
    }
    if size_bytes > 1_000_000 {
        notes.push(format!("model unusually large: {} bytes", size_bytes));
    }
    let status = if !magic_ok || size_bytes < 64 {
        CheckStatus::Fail
    } else if size_bytes > 1_000_000 {
        CheckStatus::Warn
    } else {
        CheckStatus::Pass
    };
    CheckEntry {
        name: path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string(),
        path: path.to_path_buf(),
        size_bytes,
        magic_ok,
        status,
        notes,
    }
}

fn aggregate(entries: Vec<CheckEntry>) -> BatchReport {
    let pass = entries
        .iter()
        .filter(|e| e.status == CheckStatus::Pass)
        .count();
    let warn = entries
        .iter()
        .filter(|e| e.status == CheckStatus::Warn)
        .count();
    let fail = entries
        .iter()
        .filter(|e| e.status == CheckStatus::Fail)
        .count();
    BatchReport {
        pass,
        warn,
        fail,
        entries,
    }
}

fn build_synthetic_model(name: &str, dim: usize, corrupt_magic: bool) -> Vec<u8> {
    let seed = hash_name_to_seed(name);
    let payload = generate_model_payload(seed, dim * dim);
    let mut bundle = ModelBundleV2::new()
        .with_name(name)
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("weight", vec![dim, dim], payload)
        .build();
    if corrupt_magic {
        bundle[0] = b'X';
    }
    bundle
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("check_batch")?;
    println!("=== Recipe: {} ===", ctx.name());

    // Synthesize a registry with 6 "models".
    let specs: Vec<(&str, usize, bool, bool)> = vec![
        ("ok-small", 8, false, false),
        ("ok-medium", 32, false, false),
        ("ok-large", 64, false, false),
        ("bad-magic", 16, true, false),
        ("tiny-stub", 0, false, true), // produce an intentionally tiny file
        ("huge", 256, false, false),
    ];

    let mut entries = Vec::new();
    for (name, dim, bad_magic, tiny) in specs {
        let path = ctx.path(&format!("{name}.apr"));
        let data = if tiny {
            vec![b'A', b'P', b'R', b'2'] // just 4 bytes — will flag as too small
        } else {
            build_synthetic_model(name, dim, bad_magic)
        };
        std::fs::write(&path, &data)?;
        // Re-read to validate from disk.
        let on_disk = std::fs::read(&path)?;
        entries.push(check_one(&path, &on_disk));
    }

    let report = aggregate(entries);

    println!("\n--- Batch Health-Check Report ---");
    println!(
        "{:>14} {:>10} {:>10} {:>8} Notes",
        "Model", "Size", "MagicOK", "Status"
    );
    for e in &report.entries {
        println!(
            "{:>14} {:>10} {:>10} {:>8} {}",
            e.name,
            e.size_bytes,
            e.magic_ok,
            e.status.label(),
            e.notes.join("; ")
        );
    }
    println!(
        "\nSummary: {} PASS, {} WARN, {} FAIL",
        report.pass, report.warn, report.fail
    );

    // Sanity: at least one bad_magic should be FAIL.
    assert!(
        report
            .entries
            .iter()
            .any(|e| !e.magic_ok && e.status == CheckStatus::Fail),
        "expected a magic-bytes FAIL entry"
    );

    let out = json!({
        "recipe": ctx.name(),
        "pass": report.pass,
        "warn": report.warn,
        "fail": report.fail,
        "entries": report.entries.iter().map(|e| json!({
            "name": e.name,
            "size_bytes": e.size_bytes,
            "magic_ok": e.magic_ok,
            "status": e.status.label(),
            "notes": e.notes,
        })).collect::<Vec<_>>(),
    });
    let out_path = ctx.path("batch-check.json");
    let out_bytes =
        serde_json::to_vec_pretty(&out).map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out_path, out_bytes)?;

    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_one_good_model() {
        let data = build_synthetic_model("test-ok", 8, false);
        let entry = check_one(std::path::Path::new("/tmp/test-ok.apr"), &data);
        assert!(entry.magic_ok);
        assert_eq!(entry.status, CheckStatus::Pass);
    }

    #[test]
    fn test_check_one_bad_magic_fails() {
        let data = build_synthetic_model("test-bad", 8, true);
        let entry = check_one(std::path::Path::new("/tmp/test-bad.apr"), &data);
        assert!(!entry.magic_ok);
        assert_eq!(entry.status, CheckStatus::Fail);
    }

    #[test]
    fn test_check_one_tiny_fails() {
        let data = vec![b'A', b'P', b'R', b'2'];
        let entry = check_one(std::path::Path::new("/tmp/tiny.apr"), &data);
        assert_eq!(entry.status, CheckStatus::Fail);
    }

    #[test]
    fn test_aggregate_counts() {
        let entries = vec![
            CheckEntry {
                name: "a".into(),
                path: PathBuf::new(),
                size_bytes: 100,
                magic_ok: true,
                status: CheckStatus::Pass,
                notes: vec![],
            },
            CheckEntry {
                name: "b".into(),
                path: PathBuf::new(),
                size_bytes: 2_000_000,
                magic_ok: true,
                status: CheckStatus::Warn,
                notes: vec![],
            },
            CheckEntry {
                name: "c".into(),
                path: PathBuf::new(),
                size_bytes: 10,
                magic_ok: false,
                status: CheckStatus::Fail,
                notes: vec![],
            },
        ];
        let r = aggregate(entries);
        assert_eq!(r.pass, 1);
        assert_eq!(r.warn, 1);
        assert_eq!(r.fail, 1);
    }
}
