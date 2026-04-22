//! # Recipe: SASS↔PTX Line-Map Reader
//!
//! **Category**: gpu
//! **CLI Equivalent**: `apr ptx-map kernel.cubin --direction sass-to-ptx`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example ptx_map_sass_to_ptx` exits 0
//! 2. [x] `cargo test --example ptx_map_sass_to_ptx` passes
//! 3. [x] Deterministic output (no RNG)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr ptx-map` in-process (no shell-out)
//! 10. [x] Unit tests cover forward lookup, missing address, range queries
//!
//! ## Learning Objective
//! Demonstrates a SASS↔PTX line-map reader: builds the bidirectional mapping
//! a cubin exposes via its debug info, then answers forward (SASS address →
//! PTX line) and reverse queries. Mirrors `apr ptx-map --direction sass-to-ptx`.
//!
//! ## Run Command
//! ```bash
//! cargo run --example ptx_map_sass_to_ptx
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
    pub ptx_line: u32,
}

#[derive(Debug, Clone, Default)]
pub struct PtxMap {
    // Sorted by sass_addr.
    pub entries: Vec<MapEntry>,
}

impl PtxMap {
    pub fn new(mut entries: Vec<MapEntry>) -> Self {
        entries.sort_by_key(|e| e.sass_addr);
        Self { entries }
    }

    /// Exact lookup: address → PTX line.
    pub fn sass_to_ptx(&self, addr: u32) -> Option<u32> {
        self.entries
            .iter()
            .find(|e| e.sass_addr == addr)
            .map(|e| e.ptx_line)
    }

    /// Nearest-preceding lookup (what most debuggers actually want).
    pub fn sass_to_ptx_nearest(&self, addr: u32) -> Option<u32> {
        let idx = self
            .entries
            .partition_point(|e| e.sass_addr <= addr)
            .checked_sub(1)?;
        self.entries.get(idx).map(|e| e.ptx_line)
    }

    /// Reverse lookup: PTX line → all SASS addresses it maps to.
    pub fn ptx_to_sass(&self, line: u32) -> Vec<u32> {
        self.entries
            .iter()
            .filter(|e| e.ptx_line == line)
            .map(|e| e.sass_addr)
            .collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

fn build_demo_map() -> PtxMap {
    PtxMap::new(vec![
        MapEntry {
            sass_addr: 0x0000,
            ptx_line: 12,
        },
        MapEntry {
            sass_addr: 0x0008,
            ptx_line: 13,
        },
        MapEntry {
            sass_addr: 0x0010,
            ptx_line: 14,
        },
        MapEntry {
            sass_addr: 0x0018,
            ptx_line: 14,
        },
        MapEntry {
            sass_addr: 0x0020,
            ptx_line: 15,
        },
        MapEntry {
            sass_addr: 0x0028,
            ptx_line: 16,
        },
    ])
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("ptx_map_sass_to_ptx")?;
    println!("=== Recipe: {} ===", ctx.name());

    let map = build_demo_map();
    println!("Map entries: {}", map.len());
    for e in &map.entries {
        println!("  0x{:04x} -> PTX:{}", e.sass_addr, e.ptx_line);
    }

    let queries: [u32; 4] = [0x0010, 0x0014, 0x0028, 0x0100];
    let mut results = BTreeMap::new();
    for q in queries {
        let exact = map.sass_to_ptx(q);
        let near = map.sass_to_ptx_nearest(q);
        results.insert(
            format!("0x{:04x}", q),
            json!({
                "exact": exact,
                "nearest": near,
            }),
        );
        println!("query 0x{:04x}: exact={:?} nearest={:?}", q, exact, near);
    }

    let report = json!({
        "recipe": ctx.name(),
        "entries": map.len(),
        "queries": results,
    });
    let path = ctx.path("ptx-map-sass-to-ptx.json");
    std::fs::write(
        &path,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    ctx.record_metric("entries", map.len() as i64);
    ctx.record_metric("queries", queries.len() as i64);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_lookup_returns_line() {
        let m = build_demo_map();
        assert_eq!(m.sass_to_ptx(0x0010), Some(14));
    }

    #[test]
    fn exact_lookup_miss_returns_none() {
        let m = build_demo_map();
        assert_eq!(m.sass_to_ptx(0x0007), None);
    }

    #[test]
    fn nearest_lookup_rounds_down() {
        let m = build_demo_map();
        assert_eq!(m.sass_to_ptx_nearest(0x0014), Some(14));
    }

    #[test]
    fn nearest_below_first_is_none() {
        let m = PtxMap::new(vec![MapEntry {
            sass_addr: 0x100,
            ptx_line: 1,
        }]);
        assert_eq!(m.sass_to_ptx_nearest(0x10), None);
    }

    #[test]
    fn reverse_lookup_multiple_addrs() {
        let m = build_demo_map();
        let sass = m.ptx_to_sass(14);
        assert_eq!(sass, vec![0x0010, 0x0018]);
    }
}
