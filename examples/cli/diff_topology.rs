//! # Recipe: Topology Diff (Architecture Changes)
//!
//! **Category**: cli
//! **CLI Equivalent**: `apr diff model_v1.apr model_v2.apr --topology`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example diff_topology` exits 0
//! 2. [x] `cargo test --example diff_topology` passes
//! 3. [x] Deterministic output (same seed -> same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr diff --topology` in-process (no shell-out)
//! 10. [x] Unit tests cover added / removed / reshaped / unchanged
//!
//! ## Learning Objective
//! Reports structural (topology) differences between two model descriptions:
//! added tensors, removed tensors, shape-reshaped tensors, and unchanged
//! tensors -- ignoring weight values entirely. This is the diff dimension that
//! flags architecture-level refactors.
//!
//! ## Run Command
//! ```bash
//! cargo run --example diff_topology
//! ```
//!
//! ## References
//! - Amershi, S. et al. (2019). *Software Engineering for Machine Learning: A Case Study*. ICSE-SEIP. DOI: 10.1109/ICSE-SEIP.2019.00042

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct TensorSpec {
    name: String,
    shape: Vec<usize>,
    dtype: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChangeKind {
    Added,
    Removed,
    Reshaped,
    DtypeChanged,
    Unchanged,
}

impl ChangeKind {
    fn label(self) -> &'static str {
        match self {
            Self::Added => "ADDED",
            Self::Removed => "REMOVED",
            Self::Reshaped => "RESHAPED",
            Self::DtypeChanged => "DTYPE_CHANGED",
            Self::Unchanged => "UNCHANGED",
        }
    }
}

#[derive(Debug, Clone)]
struct DiffEntry {
    name: String,
    kind: ChangeKind,
    from: Option<TensorSpec>,
    to: Option<TensorSpec>,
}

#[derive(Debug, Clone, Default)]
struct DiffSummary {
    added: usize,
    removed: usize,
    reshaped: usize,
    dtype_changed: usize,
    unchanged: usize,
}

// ---------------------------------------------------------------------------
// Logic
// ---------------------------------------------------------------------------

fn topology_diff(a: &[TensorSpec], b: &[TensorSpec]) -> (Vec<DiffEntry>, DiffSummary) {
    let map_a: BTreeMap<&str, &TensorSpec> = a.iter().map(|t| (t.name.as_str(), t)).collect();
    let map_b: BTreeMap<&str, &TensorSpec> = b.iter().map(|t| (t.name.as_str(), t)).collect();
    let keys: BTreeSet<&str> = map_a.keys().chain(map_b.keys()).copied().collect();

    let mut entries = Vec::new();
    let mut summary = DiffSummary::default();

    for k in keys {
        let ta = map_a.get(k);
        let tb = map_b.get(k);
        match (ta, tb) {
            (Some(x), None) => {
                summary.removed += 1;
                entries.push(DiffEntry {
                    name: k.to_string(),
                    kind: ChangeKind::Removed,
                    from: Some((*x).clone()),
                    to: None,
                });
            }
            (None, Some(y)) => {
                summary.added += 1;
                entries.push(DiffEntry {
                    name: k.to_string(),
                    kind: ChangeKind::Added,
                    from: None,
                    to: Some((*y).clone()),
                });
            }
            (Some(x), Some(y)) => {
                if x.shape != y.shape {
                    summary.reshaped += 1;
                    entries.push(DiffEntry {
                        name: k.to_string(),
                        kind: ChangeKind::Reshaped,
                        from: Some((*x).clone()),
                        to: Some((*y).clone()),
                    });
                } else if x.dtype != y.dtype {
                    summary.dtype_changed += 1;
                    entries.push(DiffEntry {
                        name: k.to_string(),
                        kind: ChangeKind::DtypeChanged,
                        from: Some((*x).clone()),
                        to: Some((*y).clone()),
                    });
                } else {
                    summary.unchanged += 1;
                    entries.push(DiffEntry {
                        name: k.to_string(),
                        kind: ChangeKind::Unchanged,
                        from: Some((*x).clone()),
                        to: Some((*y).clone()),
                    });
                }
            }
            (None, None) => {}
        }
    }
    (entries, summary)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("diff_topology")?;
    println!("=== Recipe: {} ===", ctx.name());

    let v1 = vec![
        TensorSpec {
            name: "embed".into(),
            shape: vec![1024, 128],
            dtype: "fp32".into(),
        },
        TensorSpec {
            name: "attn.qkv".into(),
            shape: vec![128, 384],
            dtype: "fp32".into(),
        },
        TensorSpec {
            name: "ffn.up".into(),
            shape: vec![128, 512],
            dtype: "fp32".into(),
        },
        TensorSpec {
            name: "ffn.down".into(),
            shape: vec![512, 128],
            dtype: "fp32".into(),
        },
        TensorSpec {
            name: "old_aux".into(),
            shape: vec![64, 64],
            dtype: "fp32".into(),
        },
    ];

    let v2 = vec![
        TensorSpec {
            name: "embed".into(),
            shape: vec![2048, 128],
            dtype: "fp32".into(),
        }, // reshaped
        TensorSpec {
            name: "attn.qkv".into(),
            shape: vec![128, 384],
            dtype: "fp16".into(),
        }, // dtype change
        TensorSpec {
            name: "ffn.up".into(),
            shape: vec![128, 512],
            dtype: "fp32".into(),
        }, // unchanged
        TensorSpec {
            name: "ffn.down".into(),
            shape: vec![512, 128],
            dtype: "fp32".into(),
        }, // unchanged
        TensorSpec {
            name: "ln.scale".into(),
            shape: vec![128],
            dtype: "fp32".into(),
        }, // added
    ];

    let (entries, summary) = topology_diff(&v1, &v2);

    println!("\n--- Topology Diff ---");
    println!("{:>14} {:>14} Change", "Name", "Kind");
    for e in &entries {
        let change = match e.kind {
            ChangeKind::Added => format!(
                "+ {:?}",
                e.to.as_ref().map(|t| t.shape.clone()).unwrap_or_default()
            ),
            ChangeKind::Removed => format!(
                "- {:?}",
                e.from.as_ref().map(|t| t.shape.clone()).unwrap_or_default()
            ),
            ChangeKind::Reshaped => format!(
                "{:?} -> {:?}",
                e.from.as_ref().map(|t| t.shape.clone()).unwrap_or_default(),
                e.to.as_ref().map(|t| t.shape.clone()).unwrap_or_default()
            ),
            ChangeKind::DtypeChanged => format!(
                "{} -> {}",
                e.from.as_ref().map(|t| t.dtype.clone()).unwrap_or_default(),
                e.to.as_ref().map(|t| t.dtype.clone()).unwrap_or_default()
            ),
            ChangeKind::Unchanged => "=".to_string(),
        };
        println!("{:>14} {:>14} {}", e.name, e.kind.label(), change);
    }

    println!(
        "\nSummary: +{}, -{}, reshaped {}, dtype-changed {}, unchanged {}",
        summary.added, summary.removed, summary.reshaped, summary.dtype_changed, summary.unchanged
    );

    // Sanity.
    assert_eq!(summary.added, 1);
    assert_eq!(summary.removed, 1);
    assert_eq!(summary.reshaped, 1);
    assert_eq!(summary.dtype_changed, 1);
    assert_eq!(summary.unchanged, 2);

    let out = json!({
        "recipe": ctx.name(),
        "summary": {
            "added": summary.added,
            "removed": summary.removed,
            "reshaped": summary.reshaped,
            "dtype_changed": summary.dtype_changed,
            "unchanged": summary.unchanged,
        },
        "entries": entries.iter().map(|e| json!({
            "name": e.name,
            "kind": e.kind.label(),
            "from_shape": e.from.as_ref().map(|t| t.shape.clone()),
            "to_shape": e.to.as_ref().map(|t| t.shape.clone()),
            "from_dtype": e.from.as_ref().map(|t| t.dtype.clone()),
            "to_dtype": e.to.as_ref().map(|t| t.dtype.clone()),
        })).collect::<Vec<_>>(),
    });
    let out_path = ctx.path("topology-diff.json");
    let out_bytes =
        serde_json::to_vec_pretty(&out).map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out_path, out_bytes)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn t(name: &str, shape: Vec<usize>, dtype: &str) -> TensorSpec {
        TensorSpec {
            name: name.into(),
            shape,
            dtype: dtype.into(),
        }
    }

    #[test]
    fn test_empty_inputs() {
        let (e, s) = topology_diff(&[], &[]);
        assert!(e.is_empty());
        assert_eq!(s.added, 0);
        assert_eq!(s.removed, 0);
    }

    #[test]
    fn test_all_added() {
        let (_, s) = topology_diff(&[], &[t("x", vec![2], "fp32")]);
        assert_eq!(s.added, 1);
    }

    #[test]
    fn test_all_removed() {
        let (_, s) = topology_diff(&[t("x", vec![2], "fp32")], &[]);
        assert_eq!(s.removed, 1);
    }

    #[test]
    fn test_reshape_detected() {
        let a = [t("x", vec![2, 3], "fp32")];
        let b = [t("x", vec![4, 3], "fp32")];
        let (_, s) = topology_diff(&a, &b);
        assert_eq!(s.reshaped, 1);
    }

    #[test]
    fn test_dtype_change_detected() {
        let a = [t("x", vec![2, 3], "fp32")];
        let b = [t("x", vec![2, 3], "fp16")];
        let (_, s) = topology_diff(&a, &b);
        assert_eq!(s.dtype_changed, 1);
    }

    #[test]
    fn test_unchanged_tensor() {
        let a = [t("x", vec![2, 3], "fp32")];
        let b = [t("x", vec![2, 3], "fp32")];
        let (_, s) = topology_diff(&a, &b);
        assert_eq!(s.unchanged, 1);
    }

    #[test]
    fn test_deterministic_order() {
        let a = vec![t("b", vec![1], "fp32"), t("a", vec![1], "fp32")];
        let b = vec![t("a", vec![1], "fp32"), t("b", vec![1], "fp32")];
        let (entries, _) = topology_diff(&a, &b);
        // Sorted by name via BTreeSet: a, b.
        assert_eq!(entries[0].name, "a");
        assert_eq!(entries[1].name, "b");
    }
}
