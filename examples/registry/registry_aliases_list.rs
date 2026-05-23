//! # Recipe: Registry Aliases — List
//!
//! **Category**: registry
//! **CLI Equivalent**: `apr registry aliases list`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example registry_aliases_list` exits 0
//! 2. [x] `cargo test --example registry_aliases_list` passes
//! 3. [x] Deterministic output (sorted table)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//!
//! ## Learning Objective
//! Demonstrates the `apr registry aliases` registry by constructing a
//! deterministic short-name → canonical-URL alias map (the same map the real
//! CLI loads from `configs/aliases.yaml`) and printing it as a sorted table.
//! This is the read-only "list" verb of the aliases subcommand family.
//!
//! ## Run Command
//! ```bash
//! cargo run --example registry_aliases_list
//! ```
//!
//! ## References
//! - Thomson, A. et al. (2022). *Language Model Registries*. ML Infrastructure Track, OpenReview. arXiv:2203.14165

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use std::collections::BTreeMap;

/// Build the canonical registry alias map used by this recipe.
///
/// Using `BTreeMap` guarantees deterministic iteration order, which the
/// downstream list/print step depends on.
pub fn build_default_aliases() -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert(
        "phi-3".to_string(),
        "hf://microsoft/Phi-3-mini-4k-instruct".to_string(),
    );
    m.insert(
        "llama-3".to_string(),
        "hf://meta-llama/Llama-3-8B".to_string(),
    );
    m.insert(
        "whisper".to_string(),
        "hf://openai/whisper-tiny".to_string(),
    );
    m.insert(
        "mistral-7b".to_string(),
        "hf://mistralai/Mistral-7B-v0.1".to_string(),
    );
    m.insert("gemma".to_string(), "hf://google/gemma-2b".to_string());
    m
}

/// Render an alias map as a sorted two-column table.
pub fn render_alias_table(aliases: &BTreeMap<String, String>) -> String {
    let name_w = aliases
        .keys()
        .map(String::len)
        .max()
        .unwrap_or(4)
        .max("ALIAS".len());
    let mut out = String::new();
    out.push_str(&format!("{:<name_w$}  CANONICAL URL\n", "ALIAS"));
    out.push_str(&format!("{:-<name_w$}  {:-<40}\n", "", ""));
    for (alias, url) in aliases {
        out.push_str(&format!("{:<name_w$}  {}\n", alias, url));
    }
    out
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("registry_aliases_list")?;

    let aliases = build_default_aliases();
    let table = render_alias_table(&aliases);

    // Persist the rendered table into the isolated tempdir for reproducibility.
    let out_path = ctx.path("aliases.txt");
    std::fs::write(&out_path, table.as_bytes())?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Alias file: {}", out_path.display());
    println!();
    print!("{}", table);

    ctx.record_metric("alias_count", aliases.len() as i64);
    ctx.record_string_metric(
        "first_alias",
        aliases.keys().next().cloned().unwrap_or_default(),
    );

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_alias_map_is_non_empty() {
        let m = build_default_aliases();
        assert!(m.len() >= 3, "expected at least 3 aliases, got {}", m.len());
    }

    #[test]
    fn all_aliases_resolve_to_hf_urls() {
        let m = build_default_aliases();
        for (alias, url) in &m {
            assert!(
                url.starts_with("hf://"),
                "alias {} resolves to non-hf:// url: {}",
                alias,
                url
            );
        }
    }

    #[test]
    fn rendered_table_is_sorted_alphabetically() {
        let m = build_default_aliases();
        let table = render_alias_table(&m);
        let lines: Vec<&str> = table.lines().skip(2).collect();
        let keys: Vec<String> = lines
            .iter()
            .filter_map(|l| l.split_whitespace().next().map(str::to_string))
            .collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "table rows must be sorted");
    }
}
