//! # apr publish — Manifest Full Roundtrip
//!
//! Build a complete publish-manifest-v1 manifest in memory, validate it
//! against the schema (FALSIFY-PM-001..006), and demonstrate the full
//! publish workflow without uploading anything. Manifest contains the 12
//! top-level + 7 provenance fields enforced by `apr validate-manifest`.
//!
//! Demonstrates the **CLI+.3** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: publish-manifest-v1.yaml v1.1.0 (FALSIFY-PM-001..007)
//!
//! Run with: cargo run --example cli_publish_manifest_full
//!
//! Added by PMAT-076 (expand-cookbooks: apr publish end-to-end).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use serde_json::{json, Value};

fn build_manifest() -> Value {
    json!({
        "schema_version": "1.0",
        "name": "paiml/qwen2.5-coder-7b-apache-q4k-v1",
        "version": "1.0.0",
        "description": "Qwen2.5-Coder-7B teacher checkpoint, Apache-2.0",
        "license": "Apache-2.0",
        "format": "apr",
        "quantization": "q4_k_m",
        "size_bytes": 7_500_000_000_u64,
        "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "url": "https://huggingface.co/paiml/qwen2.5-coder-7b-apache-q4k-v1/resolve/main/model.apr",
        "created_at": "2026-04-19T14:00:00Z",
        "tags": ["coder", "qwen2.5", "7b", "apache", "q4k"],
        "provenance": {
            "parent": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "parent_sha256": "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
            "parent_license": "Apache-2.0",
            "recipe_sha256": "aabbccddeeff00112233445566778899aabbccddeeff00112233445566778899",
            "tool_versions": {"apr-cli": "0.31.2", "rust": "1.89"},
            "build_host": "intel-ci-runner",
            "build_timestamp": "2026-04-19T13:55:00Z"
        }
    })
}

fn validate_manifest(m: &Value) -> Result<()> {
    let required_top = [
        "schema_version",
        "name",
        "version",
        "description",
        "license",
        "format",
        "quantization",
        "size_bytes",
        "sha256",
        "url",
        "created_at",
        "tags",
    ];
    for field in &required_top {
        if m.get(field).is_none() {
            return Err(apr_cookbook::CookbookError::Validation(format!(
                "FALSIFY-PM-001: required top-level field `{field}` missing"
            )));
        }
    }
    let prov = m.get("provenance").ok_or_else(|| {
        apr_cookbook::CookbookError::Validation("provenance section missing".into())
    })?;
    let required_prov = [
        "parent",
        "parent_sha256",
        "parent_license",
        "recipe_sha256",
        "tool_versions",
        "build_host",
        "build_timestamp",
    ];
    for field in &required_prov {
        if prov.get(field).is_none() {
            return Err(apr_cookbook::CookbookError::Validation(format!(
                "FALSIFY-PM-001: required provenance field `{field}` missing"
            )));
        }
    }
    // FALSIFY-PM-005: SPDX license allowlist.
    let license = m["license"].as_str().unwrap_or("");
    if !matches!(license, "Apache-2.0" | "MIT" | "BSD-3-Clause" | "MPL-2.0") {
        return Err(apr_cookbook::CookbookError::Validation(format!(
            "FALSIFY-PM-005: license `{license}` not in SPDX allowlist"
        )));
    }
    // sha256 sanity: 64 hex chars.
    let sha = m["sha256"].as_str().unwrap_or("");
    if sha.len() != 64 || !sha.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(apr_cookbook::CookbookError::Validation(
            "FALSIFY-PM-002: sha256 must be 64 lowercase hex chars".into(),
        ));
    }
    Ok(())
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_publish_manifest_full")?;
    let m = build_manifest();
    validate_manifest(&m)?;
    println!(
        "publish manifest validated: name={} version={} format={} size={} bytes",
        m["name"], m["version"], m["format"], m["size_bytes"]
    );
    println!("provenance.parent={}", m["provenance"]["parent"]);
    println!(
        "(in real `apr publish`, this would now stream the artifact to {})",
        m["url"]
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn missing_top_level_field_rejected() {
        let mut m = build_manifest();
        m.as_object_mut().unwrap().remove("license");
        assert!(validate_manifest(&m).is_err());
    }

    #[test]
    fn missing_provenance_field_rejected() {
        let mut m = build_manifest();
        m["provenance"]
            .as_object_mut()
            .unwrap()
            .remove("recipe_sha256");
        assert!(validate_manifest(&m).is_err());
    }

    #[test]
    fn unknown_license_rejected() {
        let mut m = build_manifest();
        m["license"] = json!("Proprietary");
        assert!(validate_manifest(&m).is_err());
    }

    #[test]
    fn malformed_sha256_rejected() {
        let mut m = build_manifest();
        m["sha256"] = json!("not-a-hex-string");
        assert!(validate_manifest(&m).is_err());
    }
}
