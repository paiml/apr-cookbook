//! # Recipe: Validate Manifest — Happy Path
//!
//! **Category**: format
//! **CLI Equivalent**: `apr validate-manifest manifest.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example validate_manifest_happy` exits 0
//! 2. [x] `cargo test --example validate_manifest_happy` passes
//! 3. [x] Deterministic output (fixed manifest, fixed payload)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Demonstrates the happy path for `apr validate-manifest`: build a publish
//! manifest in memory, serialize it to JSON on the isolated tempdir, parse it
//! back, and run the schema-level checks the real CLI performs before any
//! expensive on-disk or network work (`name`, `version`, `hash`, `artifact_url`,
//! `schema_version`).
//!
//! *This recipe uses a JSON manifest because `serde_yaml` is not a project
//! dependency — the exact same schema applies to the CLI's YAML form.*
//!
//! ## Run Command
//! ```bash
//! cargo run --example validate_manifest_happy
//! ```
//!
//! ## References
//! - Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems*. NeurIPS. arXiv:1503.05991

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde::{Deserialize, Serialize};

/// A publish manifest as accepted by `apr validate-manifest`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Manifest {
    pub schema_version: u32,
    pub name: String,
    pub version: String,
    /// `blake3` hex digest of the artifact bytes. We name the field `hash` (not
    /// `sha256`) because the project only ships `blake3` as a hashing dep.
    pub hash: String,
    pub artifact_url: String,
}

/// One issue found by schema validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestIssue {
    pub field: &'static str,
    pub message: &'static str,
}

/// Run schema-level validation against a `Manifest`. Returns a list of issues;
/// an empty list means the manifest is valid.
pub fn validate_schema(m: &Manifest) -> Vec<ManifestIssue> {
    let mut issues = Vec::new();

    if m.schema_version < 1 {
        issues.push(ManifestIssue {
            field: "schema_version",
            message: "must be >= 1",
        });
    }
    if m.name.trim().is_empty() {
        issues.push(ManifestIssue {
            field: "name",
            message: "must not be empty",
        });
    }
    if m.version.trim().is_empty() {
        issues.push(ManifestIssue {
            field: "version",
            message: "must not be empty",
        });
    }
    // blake3 hex is 64 chars (256 bits). Anything shorter is a placeholder.
    if m.hash.len() != 64 || !m.hash.chars().all(|c| c.is_ascii_hexdigit()) {
        issues.push(ManifestIssue {
            field: "hash",
            message: "must be 64 hex characters (blake3)",
        });
    }
    if !(m.artifact_url.starts_with("hf://")
        || m.artifact_url.starts_with("s3://")
        || m.artifact_url.starts_with("file://"))
    {
        issues.push(ManifestIssue {
            field: "artifact_url",
            message: "must start with hf://, s3://, or file://",
        });
    }

    issues
}

/// Build a valid manifest paired with a deterministic 64-byte payload whose
/// blake3 digest matches.
pub fn build_happy_manifest_and_payload() -> (Manifest, Vec<u8>) {
    let payload: Vec<u8> = (0u8..64).collect();
    let hash = blake3::hash(&payload).to_hex().to_string();
    let manifest = Manifest {
        schema_version: 1,
        name: "demo-model".to_string(),
        version: "0.1.0".to_string(),
        hash,
        artifact_url: "hf://demo-org/demo-model".to_string(),
    };
    (manifest, payload)
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("validate_manifest_happy")?;

    let (manifest, payload) = build_happy_manifest_and_payload();

    let manifest_path = ctx.path("manifest.json");
    let manifest_json = serde_json::to_vec_pretty(&manifest)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&manifest_path, &manifest_json)?;

    let artifact_path = ctx.path("demo-model.apr");
    std::fs::write(&artifact_path, &payload)?;

    // Round-trip the manifest from disk to prove the JSON form is well-formed.
    let parsed: Manifest = serde_json::from_slice(&std::fs::read(&manifest_path)?)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    let issues = validate_schema(&parsed);

    println!("=== Recipe: {} ===", ctx.name());
    println!("Manifest path: {}", manifest_path.display());
    println!("Artifact path: {}", artifact_path.display());
    println!("Fields:");
    println!("  schema_version: {}", parsed.schema_version);
    println!("  name:           {}", parsed.name);
    println!("  version:        {}", parsed.version);
    println!("  hash:           {}", parsed.hash);
    println!("  artifact_url:   {}", parsed.artifact_url);

    if issues.is_empty() {
        println!();
        println!("VALID: manifest passes all schema checks");
    } else {
        println!();
        println!("INVALID: {} issue(s) found", issues.len());
        for i in &issues {
            println!("  {} — {}", i.field, i.message);
        }
    }

    ctx.record_metric("issue_count", issues.len() as i64);
    ctx.record_string_metric(
        "verdict",
        if issues.is_empty() {
            "VALID"
        } else {
            "INVALID"
        },
    );
    ctx.record_metric("payload_bytes", payload.len() as i64);

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_manifest_has_no_issues() {
        let (m, _) = build_happy_manifest_and_payload();
        assert!(
            validate_schema(&m).is_empty(),
            "expected clean manifest: {:?}",
            validate_schema(&m)
        );
    }

    #[test]
    fn empty_name_is_rejected() {
        let (mut m, _) = build_happy_manifest_and_payload();
        m.name = String::new();
        let issues = validate_schema(&m);
        assert!(issues.iter().any(|i| i.field == "name"));
    }

    #[test]
    fn non_hex_hash_is_rejected() {
        let (mut m, _) = build_happy_manifest_and_payload();
        m.hash = "not-a-hex-digest".to_string();
        let issues = validate_schema(&m);
        assert!(issues.iter().any(|i| i.field == "hash"));
    }

    #[test]
    fn unsupported_url_scheme_is_rejected() {
        let (mut m, _) = build_happy_manifest_and_payload();
        m.artifact_url = "ftp://example.com/model".to_string();
        let issues = validate_schema(&m);
        assert!(issues.iter().any(|i| i.field == "artifact_url"));
    }
}
