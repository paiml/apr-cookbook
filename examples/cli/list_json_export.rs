//! # Recipe: List — JSON Export for Tooling Integration
//!
//! **Category**: cli
//! **CLI Equivalent**: `apr list --format json --output models.json`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example list_json_export` exits 0
//! 2. [x] `cargo test --example list_json_export` passes
//! 3. [x] Deterministic output (fixed fixtures)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr list --format json` in-process (no shell-out)
//! 10. [x] Unit tests cover schema, ordering, idempotent round-trip
//!
//! ## Learning Objective
//! Emits a machine-readable JSON listing with a stable schema
//! (`{version, generated_at, entries[]}`) designed for piping into
//! downstream tooling. The same bytes are produced on every run (modulo the
//! pinned `generated_at` field, which we set from the recipe seed).
//!
//! ## Run Command
//! ```bash
//! cargo run --example list_json_export
//! ```
//!
//! ## References
//! - Wolf, T. et al. (2020). *Transformers*. EMNLP demos. arXiv:1910.03771

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde::Serialize;
use serde_json::json;

#[derive(Debug, Clone, Serialize)]
struct ListEntryJson {
    name: String,
    size_bytes: u64,
    format: String,
    tags: Vec<String>,
    quantization: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ListExport {
    schema_version: u32,
    generated_at: u64,
    entries: Vec<ListEntryJson>,
}

fn fixture() -> Vec<ListEntryJson> {
    vec![
        ListEntryJson {
            name: "tinyllama-1.1b".into(),
            size_bytes: 600_000_000,
            format: "apr".into(),
            tags: vec!["chat".into(), "small".into()],
            quantization: Some("q4_0".into()),
        },
        ListEntryJson {
            name: "phi-2".into(),
            size_bytes: 2_700_000_000,
            format: "apr".into(),
            tags: vec!["chat".into()],
            quantization: Some("q5_1".into()),
        },
        ListEntryJson {
            name: "mistral-7b".into(),
            size_bytes: 4_200_000_000,
            format: "gguf".into(),
            tags: vec!["chat".into(), "code".into()],
            quantization: Some("q4_k_m".into()),
        },
        ListEntryJson {
            name: "llama-7b".into(),
            size_bytes: 13_000_000_000,
            format: "safetensors".into(),
            tags: vec!["base".into()],
            quantization: None,
        },
    ]
}

fn build_export(entries: Vec<ListEntryJson>, generated_at: u64) -> ListExport {
    let mut entries = entries;
    // Sort by name ascending for reproducibility.
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    ListExport {
        schema_version: 1,
        generated_at,
        entries,
    }
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("list_json_export")?;
    println!("=== Recipe: {} ===", ctx.name());

    let seed = hash_name_to_seed("list_json_export");
    let export = build_export(fixture(), seed);

    let out_path = ctx.path("models.json");
    let bytes = serde_json::to_vec_pretty(&export)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out_path, &bytes)?;

    println!(
        "\nSchema version: {}  Generated at: {}  Entries: {}",
        export.schema_version,
        export.generated_at,
        export.entries.len()
    );
    println!(
        "\n--- JSON listing ---\n{}",
        std::str::from_utf8(&bytes).unwrap_or("<non-utf8>")
    );
    println!("\nWrote: {}", out_path.display());

    // Round-trip check (self-test).
    let parsed: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| CookbookError::Serialization(e.to_string()))?;
    assert_eq!(parsed["schema_version"], 1);
    assert_eq!(parsed["entries"].as_array().map_or(0, Vec::len), 4);

    // Build a summary structure too.
    let summary = json!({
        "recipe": ctx.name(),
        "schema_version": export.schema_version,
        "n_entries": export.entries.len(),
        "output_bytes": bytes.len(),
    });
    let summary_path = ctx.path("summary.json");
    let summary_bytes = serde_json::to_vec_pretty(&summary)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&summary_path, summary_bytes)?;

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_export_sorts_alphabetically() {
        let e = build_export(fixture(), 1);
        let names: Vec<_> = e.entries.iter().map(|x| x.name.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted);
    }

    #[test]
    fn schema_version_is_stable() {
        let e = build_export(fixture(), 42);
        assert_eq!(e.schema_version, 1);
    }

    #[test]
    fn generated_at_echoes_input() {
        let e = build_export(fixture(), 12345);
        assert_eq!(e.generated_at, 12345);
    }

    #[test]
    fn serialization_roundtrip() {
        let e = build_export(fixture(), 0);
        let bytes = serde_json::to_vec_pretty(&e).expect("serialize");
        let v: serde_json::Value = serde_json::from_slice(&bytes).expect("parse");
        assert_eq!(v["entries"].as_array().map(|a| a.len()).unwrap_or(0), 4);
    }

    #[test]
    fn entries_keep_tags_and_quantization() {
        let e = build_export(fixture(), 0);
        let mistral = e
            .entries
            .iter()
            .find(|x| x.name == "mistral-7b")
            .expect("entry");
        assert_eq!(mistral.tags.len(), 2);
        assert_eq!(mistral.quantization.as_deref(), Some("q4_k_m"));
    }
}
