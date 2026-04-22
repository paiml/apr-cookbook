//! # Recipe: Tree Diff Between Two Architectures
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr tree diff a.apr b.apr`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example tree_arch_diff` exits 0
//! 2. [x] `cargo test --example tree_arch_diff` passes
//! 3. [x] Deterministic output (fixed architectures)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr tree diff` in-process
//! 10. [x] Unit tests cover added/removed/changed, unchanged, dotted-path keys
//!
//! ## Learning Objective
//! Demonstrates structural comparison of two model architectures. Nodes are
//! addressed by dotted path; we categorize each node as added, removed,
//! changed (same path, different param count), or unchanged. Output mirrors
//! `apr tree diff`.
//!
//! ## Run Command
//! ```bash
//! cargo run --example tree_arch_diff
//! ```
//!
//! ## References
//! - Cytron, R. et al. (1991). *Efficiently Computing Static Single Assignment Form and the Control Dependence Graph*. ACM TOPLAS. DOI: 10.1145/115372.115320

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChangeKind {
    Added,
    Removed,
    Changed,
    Unchanged,
}

impl ChangeKind {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Added => "added",
            Self::Removed => "removed",
            Self::Changed => "changed",
            Self::Unchanged => "unchanged",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ArchDiffEntry {
    pub path: String,
    pub kind: ChangeKind,
    pub left_params: Option<u64>,
    pub right_params: Option<u64>,
}

pub type FlatArch = BTreeMap<String, u64>;

pub fn diff_archs(left: &FlatArch, right: &FlatArch) -> Vec<ArchDiffEntry> {
    let mut out = Vec::new();
    let all_paths: std::collections::BTreeSet<&String> = left.keys().chain(right.keys()).collect();

    for p in all_paths {
        let l = left.get(p).copied();
        let r = right.get(p).copied();
        let kind = match (l, r) {
            (None, Some(_)) => ChangeKind::Added,
            (Some(_), None) => ChangeKind::Removed,
            (Some(lp), Some(rp)) if lp != rp => ChangeKind::Changed,
            _ => ChangeKind::Unchanged,
        };
        out.push(ArchDiffEntry {
            path: p.clone(),
            kind,
            left_params: l,
            right_params: r,
        });
    }
    out
}

fn arch_a() -> FlatArch {
    let mut m = BTreeMap::new();
    m.insert("model.embedding".into(), 32_000_000);
    m.insert("model.layer_0.attn".into(), 12_500_000);
    m.insert("model.layer_0.ffn".into(), 25_000_000);
    m.insert("model.layer_1.attn".into(), 12_500_000);
    m.insert("model.layer_1.ffn".into(), 25_000_000);
    m.insert("model.lm_head".into(), 32_000_000);
    m
}

fn arch_b() -> FlatArch {
    let mut m = BTreeMap::new();
    m.insert("model.embedding".into(), 32_000_000); // unchanged
    m.insert("model.layer_0.attn".into(), 14_000_000); // changed: +1.5M
    m.insert("model.layer_0.ffn".into(), 25_000_000); // unchanged
                                                      // layer_1 removed entirely
    m.insert("model.layer_2.attn".into(), 13_000_000); // added
    m.insert("model.layer_2.ffn".into(), 26_000_000); // added
    m.insert("model.lm_head".into(), 32_000_000); // unchanged
    m
}

fn main() -> Result<()> {
    let ctx = RecipeContext::new("tree_arch_diff")?;
    println!("=== Recipe: {} ===", ctx.name());

    let a = arch_a();
    let b = arch_b();
    let diff = diff_archs(&a, &b);

    let mut counts = [0u32; 4];
    for e in &diff {
        match e.kind {
            ChangeKind::Added => counts[0] += 1,
            ChangeKind::Removed => counts[1] += 1,
            ChangeKind::Changed => counts[2] += 1,
            ChangeKind::Unchanged => counts[3] += 1,
        }
    }

    println!(
        "Diff summary: +{} added, -{} removed, ~{} changed, ={} unchanged",
        counts[0], counts[1], counts[2], counts[3]
    );
    println!();
    for e in &diff {
        let sigil = match e.kind {
            ChangeKind::Added => "+",
            ChangeKind::Removed => "-",
            ChangeKind::Changed => "~",
            ChangeKind::Unchanged => "=",
        };
        println!(
            "  {} {:<30}  left={:?}  right={:?}",
            sigil, e.path, e.left_params, e.right_params
        );
    }

    let report = json!({
        "recipe": ctx.name(),
        "n_added": counts[0],
        "n_removed": counts[1],
        "n_changed": counts[2],
        "n_unchanged": counts[3],
        "entries": diff.iter().map(|e| json!({
            "path": e.path,
            "kind": e.kind.label(),
            "left_params": e.left_params,
            "right_params": e.right_params,
        })).collect::<Vec<_>>(),
    });
    let out = ctx.path("tree-arch-diff.json");
    std::fs::write(
        &out,
        serde_json::to_vec_pretty(&report)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?,
    )?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_archs_yield_all_unchanged() {
        let a = arch_a();
        let diff = diff_archs(&a, &a);
        assert!(diff.iter().all(|e| e.kind == ChangeKind::Unchanged));
    }

    #[test]
    fn detects_removed_paths() {
        let diff = diff_archs(&arch_a(), &arch_b());
        assert!(diff
            .iter()
            .any(|e| e.path == "model.layer_1.attn" && e.kind == ChangeKind::Removed));
    }

    #[test]
    fn detects_added_paths() {
        let diff = diff_archs(&arch_a(), &arch_b());
        assert!(diff
            .iter()
            .any(|e| e.path == "model.layer_2.attn" && e.kind == ChangeKind::Added));
    }

    #[test]
    fn detects_changed_params() {
        let diff = diff_archs(&arch_a(), &arch_b());
        let changed = diff
            .iter()
            .find(|e| e.path == "model.layer_0.attn")
            .expect("should find");
        assert_eq!(changed.kind, ChangeKind::Changed);
    }

    #[test]
    fn change_kind_label_consistent() {
        assert_eq!(ChangeKind::Added.label(), "added");
        assert_eq!(ChangeKind::Removed.label(), "removed");
        assert_eq!(ChangeKind::Changed.label(), "changed");
        assert_eq!(ChangeKind::Unchanged.label(), "unchanged");
    }
}
