//! # Recipe: Validate Manifest — Hash Mismatch
//!
//! **Category**: format
//! **CLI Equivalent**: `apr validate-manifest manifest.json --artifact model.apr`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example validate_manifest_sha_mismatch` exits 0
//! 2. [x] `cargo test --example validate_manifest_sha_mismatch` passes
//! 3. [x] Deterministic output (fixed payload + tampered byte)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Demonstrates the integrity check `apr validate-manifest --artifact <path>`
//! performs: recompute the blake3 digest of the local artifact and verify it
//! agrees with `manifest.hash`. Here the on-disk artifact has been deliberately
//! tampered with so the check must FAIL.
//!
//! *This recipe uses `blake3` (and names the manifest field `hash`) because
//! the project only depends on blake3 — the schema is identical in shape to
//! the real CLI's `sha256` field.*
//!
//! ## Run Command
//! ```bash
//! cargo run --example validate_manifest_sha_mismatch
//! ```
//!
//! ## References
//! - Merkle, R. (1988). *A Digital Signature Based on a Conventional Encryption Function*. CRYPTO. DOI: 10.1007/3-540-48184-2_32

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde::{Deserialize, Serialize};

/// Publish manifest (subset relevant to this recipe).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Manifest {
    pub schema_version: u32,
    pub name: String,
    pub version: String,
    pub hash: String,
    pub artifact_url: String,
}

/// Outcome of a manifest/artifact hash comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HashCheck {
    Match,
    Mismatch { expected: String, actual: String },
}

/// Compute the blake3 digest of `bytes` and compare to the manifest hash.
pub fn verify_artifact_hash(manifest_hash: &str, artifact: &[u8]) -> HashCheck {
    let actual = blake3::hash(artifact).to_hex().to_string();
    if actual == manifest_hash {
        HashCheck::Match
    } else {
        HashCheck::Mismatch {
            expected: manifest_hash.to_string(),
            actual,
        }
    }
}

/// Build the canonical (un-tampered) payload used by this recipe.
pub fn canonical_payload() -> Vec<u8> {
    (0u8..128).collect()
}

/// Tamper with a payload by flipping the last byte. Deterministic.
pub fn tamper(payload: &[u8]) -> Vec<u8> {
    let mut v = payload.to_vec();
    if let Some(last) = v.last_mut() {
        *last ^= 0xFF;
    }
    v
}

/// Build a manifest pinned to the canonical payload's hash (so that a tampered
/// artifact will trigger a mismatch).
pub fn build_manifest_for_canonical(canonical: &[u8]) -> Manifest {
    let hash = blake3::hash(canonical).to_hex().to_string();
    Manifest {
        schema_version: 1,
        name: "integrity-demo".to_string(),
        version: "0.1.0".to_string(),
        hash,
        artifact_url: "file://./integrity-demo.apr".to_string(),
    }
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("validate_manifest_sha_mismatch")?;

    // Stage 1: commit the manifest to what the clean payload hashes to.
    let canonical = canonical_payload();
    let manifest = build_manifest_for_canonical(&canonical);

    let manifest_path = ctx.path("manifest.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    // Stage 2: simulate a corrupted/tampered on-disk artifact.
    let tampered = tamper(&canonical);
    let artifact_path = ctx.path("integrity-demo.apr");
    std::fs::write(&artifact_path, &tampered)?;

    // Stage 3: reparse the manifest and check hash agreement.
    let parsed: Manifest = serde_json::from_slice(&std::fs::read(&manifest_path)?)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    let on_disk = std::fs::read(&artifact_path)?;
    let check = verify_artifact_hash(&parsed.hash, &on_disk);

    println!("=== Recipe: {} ===", ctx.name());
    println!("Manifest path: {}", manifest_path.display());
    println!("Artifact path: {}", artifact_path.display());
    println!("Expected hash: {}", parsed.hash);

    let (verdict, actual) = match &check {
        HashCheck::Match => ("MATCH", parsed.hash.clone()),
        HashCheck::Mismatch { actual, .. } => {
            println!("Actual hash:   {}", actual);
            println!();
            println!("MISMATCH: artifact does not match manifest hash");
            println!("  this is the expected outcome for this recipe — the");
            println!("  on-disk artifact was tampered with by flipping one byte.");
            ("MISMATCH", actual.clone())
        }
    };

    ctx.record_string_metric("verdict", verdict);
    ctx.record_string_metric("actual_hash", actual);
    ctx.record_metric("artifact_bytes", on_disk.len() as i64);

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_payload_matches_its_own_manifest() {
        let payload = canonical_payload();
        let m = build_manifest_for_canonical(&payload);
        assert_eq!(verify_artifact_hash(&m.hash, &payload), HashCheck::Match);
    }

    #[test]
    fn tampered_payload_triggers_mismatch() {
        let payload = canonical_payload();
        let m = build_manifest_for_canonical(&payload);
        let bad = tamper(&payload);
        match verify_artifact_hash(&m.hash, &bad) {
            HashCheck::Mismatch { expected, actual } => {
                assert_eq!(expected, m.hash);
                assert_ne!(actual, m.hash);
            }
            HashCheck::Match => panic!("expected mismatch, got match"),
        }
    }

    #[test]
    fn tamper_is_deterministic() {
        let p = canonical_payload();
        assert_eq!(tamper(&p), tamper(&p));
    }
}
