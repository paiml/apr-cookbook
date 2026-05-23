//! # Recipe: Hot-Region Inverse Map (SASS addr → source line)
//!
//! **Category**: gpu
//! **CLI Equivalent**: `apr ptx-map kernel.cubin --hot-regions sample.csv --top 5`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example ptx_map_hot_regions` exits 0
//! 2. [x] `cargo test --example ptx_map_hot_regions` passes
//! 3. [x] Deterministic output (no RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr ptx-map --hot-regions` in-process (no shell-out)
//! 10. [x] Unit tests cover sample aggregation, top-N truncation, empty input
//!
//! ## Learning Objective
//! Demonstrates the hot-region inverse workflow: consume PC samples from a
//! profiler, project each SASS address to a source line via the PTX debug
//! map, accumulate per-line hit counts, and report the top-N hottest source
//! lines. Mirrors `apr ptx-map --hot-regions`.
//!
//! ## Run Command
//! ```bash
//! cargo run --example ptx_map_hot_regions
//! ```
//!
//! ## References
//! - Lattner, C. et al. (2021). *MLIR: Scaling Compiler Infrastructure for Domain Specific Computation*. CGO. arXiv:2002.11054

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct MapEntry {
    pub sass_addr: u32,
    pub src_line: u32,
    pub src_file: String,
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub sass_addr: u32,
    pub hits: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HotLine {
    pub file: String,
    pub line: u32,
    pub hits: u32,
}

fn nearest_entry(map: &[MapEntry], addr: u32) -> Option<&MapEntry> {
    map.iter()
        .filter(|e| e.sass_addr <= addr)
        .max_by_key(|e| e.sass_addr)
}

pub fn aggregate_hot_lines(samples: &[Sample], map: &[MapEntry]) -> Vec<HotLine> {
    let mut tally: BTreeMap<(String, u32), u32> = BTreeMap::new();
    for s in samples {
        if let Some(e) = nearest_entry(map, s.sass_addr) {
            *tally.entry((e.src_file.clone(), e.src_line)).or_insert(0) += s.hits;
        }
    }
    let mut out: Vec<HotLine> = tally
        .into_iter()
        .map(|((file, line), hits)| HotLine { file, line, hits })
        .collect();
    out.sort_by(|a, b| {
        b.hits
            .cmp(&a.hits)
            .then(a.file.cmp(&b.file))
            .then(a.line.cmp(&b.line))
    });
    out
}

pub fn top_n(lines: &[HotLine], n: usize) -> Vec<HotLine> {
    lines.iter().take(n).cloned().collect()
}

fn build_map() -> Vec<MapEntry> {
    vec![
        MapEntry {
            sass_addr: 0x00,
            src_line: 10,
            src_file: "kernel.cu".into(),
        },
        MapEntry {
            sass_addr: 0x08,
            src_line: 12,
            src_file: "kernel.cu".into(),
        },
        MapEntry {
            sass_addr: 0x10,
            src_line: 14,
            src_file: "kernel.cu".into(),
        },
        MapEntry {
            sass_addr: 0x20,
            src_line: 20,
            src_file: "kernel.cu".into(),
        },
    ]
}

fn build_samples() -> Vec<Sample> {
    vec![
        Sample {
            sass_addr: 0x00,
            hits: 50,
        },
        Sample {
            sass_addr: 0x04,
            hits: 80,
        },
        Sample {
            sass_addr: 0x0c,
            hits: 20,
        },
        Sample {
            sass_addr: 0x18,
            hits: 200,
        },
        Sample {
            sass_addr: 0x24,
            hits: 10,
        },
    ]
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("ptx_map_hot_regions")?;
    println!("=== Recipe: {} ===", ctx.name());

    let map = build_map();
    let samples = build_samples();
    let hot = aggregate_hot_lines(&samples, &map);
    let top = top_n(&hot, 3);

    println!("Top {} hottest source lines:", top.len());
    for h in &top {
        println!("  {}:{:<4} hits={}", h.file, h.line, h.hits);
    }

    let report = json!({
        "recipe": ctx.name(),
        "sample_count": samples.len(),
        "distinct_hot_lines": hot.len(),
        "top": top.iter().map(|h| json!({
            "file": h.file,
            "line": h.line,
            "hits": h.hits,
        })).collect::<Vec<_>>(),
    });
    let path = ctx.path("ptx-map-hot.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    ctx.record_metric("distinct_hot_lines", hot.len() as i64);
    ctx.record_metric("top_lines", top.len() as i64);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_samples_give_empty_result() {
        let map = build_map();
        assert!(aggregate_hot_lines(&[], &map).is_empty());
    }

    #[test]
    fn aggregates_same_line() {
        let map = build_map();
        let samples = vec![
            Sample {
                sass_addr: 0x00,
                hits: 10,
            },
            Sample {
                sass_addr: 0x04,
                hits: 20,
            },
        ];
        // 0x04 → nearest 0x00 → line 10
        let hot = aggregate_hot_lines(&samples, &map);
        assert_eq!(hot.len(), 1);
        assert_eq!(hot[0].line, 10);
        assert_eq!(hot[0].hits, 30);
    }

    #[test]
    fn top_n_truncates() {
        let lines = vec![
            HotLine {
                file: "a".into(),
                line: 1,
                hits: 10,
            },
            HotLine {
                file: "a".into(),
                line: 2,
                hits: 5,
            },
        ];
        assert_eq!(top_n(&lines, 1).len(), 1);
    }

    #[test]
    fn top_n_zero_empty() {
        let lines = vec![HotLine {
            file: "a".into(),
            line: 1,
            hits: 10,
        }];
        assert!(top_n(&lines, 0).is_empty());
    }

    #[test]
    fn sort_descending_by_hits() {
        let map = build_map();
        let samples = build_samples();
        let hot = aggregate_hot_lines(&samples, &map);
        for w in hot.windows(2) {
            assert!(w[0].hits >= w[1].hits);
        }
    }
}
