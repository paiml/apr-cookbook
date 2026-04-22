//! # Recipe: List — Size Filter + Sort
//!
//! **Category**: cli
//! **CLI Equivalent**: `apr list --min-size 100MB --sort size --desc`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example list_size_filter` exits 0
//! 2. [x] `cargo test --example list_size_filter` passes
//! 3. [x] Deterministic output (fixed fixtures)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr list --min-size --sort` in-process (no shell-out)
//! 10. [x] Unit tests cover filter thresholds, ordering, ties
//!
//! ## Learning Objective
//! Demonstrates list-view filtering + ordering. Given a registry of model
//! entries with sizes, filters by a minimum size threshold and sorts by size
//! in descending order (tie-breaker: name ascending) — mirroring the real
//! `apr list` CLI's `--min-size --sort size` flags.
//!
//! ## Run Command
//! ```bash
//! cargo run --example list_size_filter
//! ```
//!
//! ## References
//! - Wolf, T. et al. (2020). *Transformers*. EMNLP demos. arXiv:1910.03771

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

#[derive(Debug, Clone)]
struct ListEntry {
    name: String,
    size_bytes: u64,
    format: &'static str,
}

fn fixture() -> Vec<ListEntry> {
    vec![
        ListEntry {
            name: "tinyllama-1.1b.apr".into(),
            size_bytes: 600_000_000,
            format: "apr",
        },
        ListEntry {
            name: "phi-2.apr".into(),
            size_bytes: 2_700_000_000,
            format: "apr",
        },
        ListEntry {
            name: "mistral-7b.gguf".into(),
            size_bytes: 4_200_000_000,
            format: "gguf",
        },
        ListEntry {
            name: "llama-7b.safetensors".into(),
            size_bytes: 13_000_000_000,
            format: "safetensors",
        },
        ListEntry {
            name: "tiny-distilbert.apr".into(),
            size_bytes: 80_000_000,
            format: "apr",
        },
        ListEntry {
            name: "debug_stub.apr".into(),
            size_bytes: 1_024,
            format: "apr",
        },
    ]
}

fn filter_by_min_size(entries: &[ListEntry], min: u64) -> Vec<ListEntry> {
    entries
        .iter()
        .filter(|e| e.size_bytes >= min)
        .cloned()
        .collect()
}

fn sort_by_size_desc(entries: &mut [ListEntry]) {
    entries.sort_by(|a, b| {
        b.size_bytes
            .cmp(&a.size_bytes)
            .then_with(|| a.name.cmp(&b.name))
    });
}

fn human_bytes(n: u64) -> String {
    const KB: u64 = 1 << 10;
    const MB: u64 = 1 << 20;
    const GB: u64 = 1 << 30;
    if n >= GB {
        format!("{:.2} GB", n as f64 / GB as f64)
    } else if n >= MB {
        format!("{:.2} MB", n as f64 / MB as f64)
    } else if n >= KB {
        format!("{:.2} KB", n as f64 / KB as f64)
    } else {
        format!("{} B", n)
    }
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("list_size_filter")?;
    println!("=== Recipe: {} ===", ctx.name());

    let all = fixture();
    let min_size = 100 * 1024 * 1024; // 100 MB

    let mut filtered = filter_by_min_size(&all, min_size);
    sort_by_size_desc(&mut filtered);

    println!("\nRegistry:   {} total entries", all.len());
    println!("Min size:   {}", human_bytes(min_size));
    println!("After filter: {} entries\n", filtered.len());
    println!("{:<30} {:<14} {:<12}", "Name", "Size", "Format");
    for e in &filtered {
        println!(
            "{:<30} {:<14} {:<12}",
            e.name,
            human_bytes(e.size_bytes),
            e.format
        );
    }

    let report = json!({
        "recipe": ctx.name(),
        "n_total": all.len(),
        "min_size_bytes": min_size,
        "n_filtered": filtered.len(),
        "entries": filtered.iter().map(|e| json!({
            "name": e.name,
            "size_bytes": e.size_bytes,
            "format": e.format,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("list-filter.json");
    let bytes = serde_json::to_vec_pretty(&report)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out, bytes)?;

    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_removes_small_entries() {
        let f = filter_by_min_size(&fixture(), 100_000_000);
        assert!(f.iter().all(|e| e.size_bytes >= 100_000_000));
        assert!(!f.iter().any(|e| e.name == "debug_stub.apr"));
    }

    #[test]
    fn sort_desc_by_size() {
        let mut f = fixture();
        sort_by_size_desc(&mut f);
        for w in f.windows(2) {
            assert!(w[0].size_bytes >= w[1].size_bytes);
        }
    }

    #[test]
    fn tie_breaker_sorts_names_ascending() {
        let mut f = vec![
            ListEntry {
                name: "b.apr".into(),
                size_bytes: 100,
                format: "apr",
            },
            ListEntry {
                name: "a.apr".into(),
                size_bytes: 100,
                format: "apr",
            },
        ];
        sort_by_size_desc(&mut f);
        assert_eq!(f[0].name, "a.apr");
    }

    #[test]
    fn human_bytes_formats_gb_mb_kb() {
        assert!(human_bytes(2_147_483_648).contains("GB"));
        assert!(human_bytes(2 * 1024 * 1024).contains("MB"));
        assert!(human_bytes(2048).contains("KB"));
        assert_eq!(human_bytes(500), "500 B");
    }

    #[test]
    fn filter_zero_threshold_keeps_all() {
        let all = fixture();
        let f = filter_by_min_size(&all, 0);
        assert_eq!(f.len(), all.len());
    }
}
