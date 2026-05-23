//! # Recipe: Showcase Export to Markdown Report
//!
//! **Category**: advanced
//! **CLI Equivalent**: `apr showcase --export md --out showcase.md`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example showcase_markdown` exits 0
//! 2. [x] `cargo test --example showcase_markdown` passes
//! 3. [x] Deterministic output (fixed fixtures)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr showcase --export md` rendering in-process
//! 10. [x] Unit tests cover table rendering, header escape, reproducibility
//!
//! ## Learning Objective
//! Demonstrates the report-generation path: take a structured showcase
//! manifest and produce a Markdown file with a GFM table. We validate that
//! the rendering is byte-identical on identical input (idempotent) and that
//! special characters are correctly escaped.
//!
//! ## Run Command
//! ```bash
//! cargo run --example showcase_markdown
//! ```
//!
//! ## References
//! - Abadi, M. et al. (2016). *TensorFlow: A System for Large-Scale Machine Learning*. OSDI. arXiv:1605.08695

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone)]
pub struct ShowcaseEntry {
    pub name: String,
    pub category: String,
    pub latency_ms: f64,
    pub accuracy: f64,
}

pub fn entries() -> Vec<ShowcaseEntry> {
    vec![
        ShowcaseEntry {
            name: "sentiment_classifier".into(),
            category: "nlp".into(),
            latency_ms: 4.1,
            accuracy: 0.912,
        },
        ShowcaseEntry {
            name: "chat_assistant".into(),
            category: "nlp".into(),
            latency_ms: 185.3,
            accuracy: 0.87,
        },
        ShowcaseEntry {
            name: "image_classifier".into(),
            category: "vision".into(),
            latency_ms: 12.8,
            accuracy: 0.885,
        },
        ShowcaseEntry {
            name: "whisper_transcribe".into(),
            category: "speech".into(),
            latency_ms: 620.4,
            accuracy: 0.95,
        },
        ShowcaseEntry {
            name: "vector_search".into(),
            category: "retrieval".into(),
            latency_ms: 2.3,
            accuracy: 0.78,
        },
    ]
}

/// Escape Markdown pipe characters and newlines in a cell value.
pub fn escape_md_cell(s: &str) -> String {
    s.replace('|', r"\|").replace('\n', " ")
}

pub fn render_markdown(entries: &[ShowcaseEntry]) -> String {
    let mut out = String::new();
    out.push_str("# APR Showcase Gallery\n\n");
    out.push_str(&format!("Total demos: **{}**\n\n", entries.len()));
    out.push_str("| Demo | Category | Latency (ms) | Accuracy |\n");
    out.push_str("|------|----------|--------------|----------|\n");
    for e in entries {
        out.push_str(&format!(
            "| {} | {} | {:.1} | {:.3} |\n",
            escape_md_cell(&e.name),
            escape_md_cell(&e.category),
            e.latency_ms,
            e.accuracy
        ));
    }
    out.push_str("\n> Generated deterministically from a fixed manifest.\n");
    out
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("showcase_markdown")?;
    println!("=== Recipe: {} ===", ctx.name());

    let entries = entries();
    let md = render_markdown(&entries);

    let out_md = ctx.path("showcase.md");
    std::fs::write(&out_md, &md)?;

    println!("Rendered {} demos to {}", entries.len(), out_md.display());
    println!("---- PREVIEW ----");
    println!("{}", md);

    let report = json!({
        "recipe": ctx.name(),
        "output_path": out_md.to_string_lossy(),
        "n_entries": entries.len(),
        "n_bytes": md.len(),
    });
    let out_json = ctx.path("showcase-md.json");
    std::fs::write(
        &out_json,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_contains_header_row() {
        let md = render_markdown(&entries());
        assert!(md.contains("| Demo | Category"));
        assert!(md.contains("|------|----------"));
    }

    #[test]
    fn render_contains_all_entry_names() {
        let md = render_markdown(&entries());
        for e in entries() {
            assert!(md.contains(&e.name));
        }
    }

    #[test]
    fn escape_md_cell_handles_pipe() {
        assert_eq!(escape_md_cell("a | b"), r"a \| b");
    }

    #[test]
    fn escape_md_cell_replaces_newline_with_space() {
        assert_eq!(escape_md_cell("a\nb"), "a b");
    }

    #[test]
    fn render_is_deterministic() {
        let md1 = render_markdown(&entries());
        let md2 = render_markdown(&entries());
        assert_eq!(md1, md2);
    }
}
