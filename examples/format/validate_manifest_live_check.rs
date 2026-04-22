//! # Recipe: Validate Manifest — Live Readiness Check
//!
//! **Category**: format
//! **CLI Equivalent**: `apr validate-manifest manifest.json --artifact model.apr --live`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example validate_manifest_live_check` exits 0
//! 2. [x] `cargo test --example validate_manifest_live_check` passes
//! 3. [x] Deterministic output (fixed manifest + fixed payload, no network)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Simulates the `--live` mode of `apr validate-manifest`: after the cheap
//! schema checks pass, run a set of readiness checks that require the on-disk
//! artifact. All checks are derived from local `std::fs::metadata` and the
//! parsed manifest — **no network is used**, which keeps the recipe hermetic
//! yet demonstrates the same control flow the real CLI follows.
//!
//! Checks performed:
//! 1. Artifact size under 10 GB (hard gate for some registries).
//! 2. Artifact filename stem matches `manifest.name`.
//! 3. Artifact `last-modified` timestamp within 90 days of "now".
//!
//! *This recipe uses a JSON manifest because `serde_yaml` is not a project
//! dependency — the exact same schema applies to the CLI's YAML form.*
//!
//! ## Run Command
//! ```bash
//! cargo run --example validate_manifest_live_check
//! ```
//!
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning: A Survey of Case Studies*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{Duration, SystemTime};

/// Publish manifest (subset relevant to this recipe).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Manifest {
    pub schema_version: u32,
    pub name: String,
    pub version: String,
    pub hash: String,
    pub artifact_url: String,
}

/// One readiness check outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveCheck {
    pub name: &'static str,
    pub passed: bool,
    pub detail: String,
}

/// Upper bound on artifact size: 10 GiB.
pub const MAX_ARTIFACT_BYTES: u64 = 10 * 1024 * 1024 * 1024;

/// Maximum age of an artifact for "live" publishing (90 days).
pub const MAX_AGE: Duration = Duration::from_secs(60 * 60 * 24 * 90);

/// Check that an artifact is below the size limit.
pub fn check_size(size_bytes: u64) -> LiveCheck {
    let passed = size_bytes <= MAX_ARTIFACT_BYTES;
    LiveCheck {
        name: "size_under_10gb",
        passed,
        detail: format!(
            "size={} bytes (limit={} bytes)",
            size_bytes, MAX_ARTIFACT_BYTES
        ),
    }
}

/// Check that the artifact filename stem matches `manifest.name`.
pub fn check_filename_matches(artifact_path: &Path, expected_name: &str) -> LiveCheck {
    let stem = artifact_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("<unreadable>");
    let passed = stem == expected_name;
    LiveCheck {
        name: "filename_matches_manifest_name",
        passed,
        detail: format!("stem={:?}, expected={:?}", stem, expected_name),
    }
}

/// Check that the artifact was modified within [`MAX_AGE`] of `now`.
pub fn check_recency(modified: SystemTime, now: SystemTime) -> LiveCheck {
    match now.duration_since(modified) {
        Ok(age) if age <= MAX_AGE => LiveCheck {
            name: "modified_within_90_days",
            passed: true,
            detail: format!("age={}s (limit={}s)", age.as_secs(), MAX_AGE.as_secs()),
        },
        Ok(age) => LiveCheck {
            name: "modified_within_90_days",
            passed: false,
            detail: format!(
                "age={}s exceeds limit={}s",
                age.as_secs(),
                MAX_AGE.as_secs()
            ),
        },
        Err(_) => LiveCheck {
            // The artifact's mtime is *after* `now` — treat as a pass (clock
            // skew) but record the detail.
            name: "modified_within_90_days",
            passed: true,
            detail: "artifact mtime is in the future (clock skew) — treated as fresh".to_string(),
        },
    }
}

/// Aggregate a set of checks into a single PASS/FAIL verdict.
pub fn verdict(checks: &[LiveCheck]) -> &'static str {
    if checks.iter().all(|c| c.passed) {
        "PASS"
    } else {
        "FAIL"
    }
}

/// Build the canonical manifest + payload used in this recipe.
pub fn build_happy() -> (Manifest, Vec<u8>) {
    let payload: Vec<u8> = (0u8..200).collect();
    let hash = blake3::hash(&payload).to_hex().to_string();
    let manifest = Manifest {
        schema_version: 1,
        name: "live-demo".to_string(),
        version: "0.1.0".to_string(),
        hash,
        artifact_url: "file://./live-demo.apr".to_string(),
    };
    (manifest, payload)
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("validate_manifest_live_check")?;

    let (manifest, payload) = build_happy();

    // Write manifest + artifact into the isolated tempdir, using a filename
    // stem that matches manifest.name so check #2 passes.
    let manifest_path = ctx.path("manifest.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    let artifact_path = ctx.path(&format!("{}.apr", manifest.name));
    std::fs::write(&artifact_path, &payload)?;

    // Gather metadata for the live checks (no network).
    let meta = std::fs::metadata(&artifact_path)?;
    let size = meta.len();
    let modified = meta.modified().unwrap_or_else(|_| SystemTime::now());
    let now = SystemTime::now();

    let checks = vec![
        check_size(size),
        check_filename_matches(&artifact_path, &manifest.name),
        check_recency(modified, now),
    ];

    println!("=== Recipe: {} ===", ctx.name());
    println!("Manifest: {}", manifest_path.display());
    println!("Artifact: {}", artifact_path.display());
    println!();
    println!("Live readiness checks:");
    for c in &checks {
        let mark = if c.passed { "PASS" } else { "FAIL" };
        println!("  [{}] {:<34} {}", mark, c.name, c.detail);
    }
    let overall = verdict(&checks);
    println!();
    println!("Overall: {}", overall);

    ctx.record_metric("check_count", checks.len() as i64);
    ctx.record_metric(
        "pass_count",
        checks.iter().filter(|c| c.passed).count() as i64,
    );
    ctx.record_string_metric("verdict", overall);
    ctx.record_metric("artifact_bytes", size as i64);

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_under_limit_passes() {
        let c = check_size(1024);
        assert!(c.passed);
    }

    #[test]
    fn oversize_artifact_fails() {
        let c = check_size(MAX_ARTIFACT_BYTES + 1);
        assert!(!c.passed);
    }

    #[test]
    fn filename_must_match_manifest_name() {
        let p = Path::new("/tmp/live-demo.apr");
        assert!(check_filename_matches(p, "live-demo").passed);
        assert!(!check_filename_matches(p, "other-name").passed);
    }

    #[test]
    fn fresh_artifact_passes_recency() {
        let now = SystemTime::now();
        let fresh = now - Duration::from_secs(60);
        assert!(check_recency(fresh, now).passed);
    }

    #[test]
    fn stale_artifact_fails_recency() {
        let now = SystemTime::now();
        let stale = now - (MAX_AGE + Duration::from_secs(60));
        assert!(!check_recency(stale, now).passed);
    }

    #[test]
    fn all_pass_yields_verdict_pass() {
        let checks = vec![
            check_size(10),
            check_filename_matches(Path::new("/tmp/x.apr"), "x"),
            check_recency(SystemTime::now(), SystemTime::now()),
        ];
        assert_eq!(verdict(&checks), "PASS");
    }
}
