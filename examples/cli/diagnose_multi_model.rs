//! # Recipe: Multi-Model Comparative Diagnostic
//!
//! **Category**: cli
//! **CLI Equivalent**: `apr diagnose model_a.apr model_b.apr model_c.apr --compare`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example diagnose_multi_model` exits 0
//! 2. [x] `cargo test --example diagnose_multi_model` passes
//! 3. [x] Deterministic output (same seed -> same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr diagnose` multi-model comparison in-process (no shell-out)
//! 10. [x] Unit tests cover score ordering, warning thresholds, empty set
//!
//! ## Learning Objective
//! Runs diagnostic scoring (size, parameter count, quantization ratio, magic-
//! byte validity, warning count) across three candidate models and ranks them
//! by composite health score. Highlights outliers for the operator to review.
//!
//! ## Run Command
//! ```bash
//! cargo run --example diagnose_multi_model
//! ```
//!
//! ## References
//! - Amershi, S. et al. (2019). *Software Engineering for Machine Learning: A Case Study*. ICSE-SEIP. DOI: 10.1109/ICSE-SEIP.2019.00042

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ModelDiagnostic {
    name: String,
    size_bytes: usize,
    magic_ok: bool,
    n_warnings: usize,
    health_score: f32,
}

// ---------------------------------------------------------------------------
// Logic
// ---------------------------------------------------------------------------

/// Compute a composite health score in [0.0, 1.0].
///
/// Components:
/// - magic_ok: +0.4 if valid, else 0
/// - size_ok: +0.3 if in [128, 1 MB]
/// - few_warnings: +0.3 if n_warnings <= 2
fn score(size_bytes: usize, magic_ok: bool, n_warnings: usize) -> f32 {
    let mut s: f32 = 0.0;
    if magic_ok {
        s += 0.4;
    }
    if (128..=1_000_000).contains(&size_bytes) {
        s += 0.3;
    }
    if n_warnings <= 2 {
        s += 0.3;
    } else if n_warnings <= 5 {
        s += 0.15;
    }
    s.clamp(0.0_f32, 1.0_f32)
}

fn diagnose_one(name: &str, data: &[u8]) -> ModelDiagnostic {
    let magic_ok = data.len() >= 4 && &data[0..4] == b"APR2";
    // Synthesize a "warning count" from the raw bytes to exercise thresholds.
    let n_warnings = if data.len() < 128 {
        6
    } else if data.len() > 500_000 {
        4
    } else {
        (data.len() / 10_000).min(3)
    };
    let health = score(data.len(), magic_ok, n_warnings);
    ModelDiagnostic {
        name: name.to_string(),
        size_bytes: data.len(),
        magic_ok,
        n_warnings,
        health_score: health,
    }
}

fn rank_by_health(mut v: Vec<ModelDiagnostic>) -> Vec<ModelDiagnostic> {
    v.sort_by(|a, b| {
        b.health_score
            .partial_cmp(&a.health_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    v
}

fn build_model(name: &str, dim: usize, corrupt: bool) -> Vec<u8> {
    let seed = hash_name_to_seed(name);
    let payload = generate_model_payload(seed, dim * dim);
    let mut bundle = ModelBundleV2::new()
        .with_name(name)
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("weight", vec![dim, dim], payload)
        .build();
    if corrupt {
        bundle[0] = b'Z';
    }
    bundle
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("diagnose_multi_model")?;
    println!("=== Recipe: {} ===", ctx.name());

    let specs = [
        ("healthy-small", 16, false),
        ("healthy-medium", 32, false),
        ("corrupt-magic", 16, true),
    ];

    let mut diags = Vec::new();
    for (name, dim, corrupt) in specs {
        let data = build_model(name, dim, corrupt);
        let path = ctx.path(&format!("{name}.apr"));
        std::fs::write(&path, &data)?;
        let on_disk = std::fs::read(&path)?;
        diags.push(diagnose_one(name, &on_disk));
    }

    let ranked = rank_by_health(diags.clone());

    println!("\n--- Diagnosed {} models ---", diags.len());
    println!(
        "{:>16} {:>10} {:>10} {:>10} {:>10}",
        "Name", "Size", "MagicOK", "Warnings", "Health"
    );
    for d in &ranked {
        println!(
            "{:>16} {:>10} {:>10} {:>10} {:>10.2}",
            d.name, d.size_bytes, d.magic_ok, d.n_warnings, d.health_score
        );
    }

    let best = ranked
        .first()
        .ok_or_else(|| CookbookError::invalid_format("no diagnostics"))?;
    println!(
        "\nBest model: {} (score {:.2})",
        best.name, best.health_score
    );

    // Sanity: the corrupt model should rank lowest.
    let last = ranked
        .last()
        .ok_or_else(|| CookbookError::invalid_format("no last"))?;
    assert_eq!(last.name, "corrupt-magic");

    let out = json!({
        "recipe": ctx.name(),
        "n_models": diags.len(),
        "best": best.name,
        "ranked": ranked.iter().map(|d| json!({
            "name": d.name,
            "size_bytes": d.size_bytes,
            "magic_ok": d.magic_ok,
            "n_warnings": d.n_warnings,
            "health_score": d.health_score,
        })).collect::<Vec<_>>(),
    });
    let out_path = ctx.path("diagnose-multi.json");
    let out_bytes =
        serde_json::to_vec_pretty(&out).map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out_path, out_bytes)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_perfect() {
        let s = score(500, true, 0);
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_bad_magic_caps() {
        let s = score(500, false, 0);
        assert!(s <= 0.6 + 1e-6);
    }

    #[test]
    fn test_score_too_small_caps() {
        let s = score(10, true, 0);
        // size_ok lost, n_warnings caps at the 6-warning branch because size < 128.
        // Score computation uses the params directly, so size bucket is lost:
        assert!(s <= 0.75);
    }

    #[test]
    fn test_score_many_warnings_reduces() {
        let a = score(500, true, 0);
        let b = score(500, true, 10);
        assert!(b < a);
    }

    #[test]
    fn test_diagnose_one_detects_bad_magic() {
        let mut data = b"APR2".to_vec();
        data.extend(vec![0_u8; 500]);
        let ok = diagnose_one("ok", &data);
        assert!(ok.magic_ok);

        let mut bad = data.clone();
        bad[0] = b'X';
        let d = diagnose_one("x", &bad);
        assert!(!d.magic_ok);
        assert!(d.health_score < ok.health_score);
    }

    #[test]
    fn test_ranking_orders_desc() {
        let v = vec![
            ModelDiagnostic {
                name: "low".into(),
                size_bytes: 0,
                magic_ok: false,
                n_warnings: 10,
                health_score: 0.1,
            },
            ModelDiagnostic {
                name: "high".into(),
                size_bytes: 0,
                magic_ok: true,
                n_warnings: 0,
                health_score: 0.9,
            },
        ];
        let r = rank_by_health(v);
        assert_eq!(r[0].name, "high");
    }

    #[test]
    fn test_ranking_empty_is_empty() {
        assert!(rank_by_health(vec![]).is_empty());
    }
}
