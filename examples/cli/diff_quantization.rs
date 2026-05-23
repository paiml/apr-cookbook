//! # Recipe: Quantization Diff (Dtype Changes)
//!
//! **Category**: cli
//! **CLI Equivalent**: `apr diff model_fp32.apr model_int8.apr --quantization`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example diff_quantization` exits 0
//! 2. [x] `cargo test --example diff_quantization` passes
//! 3. [x] Deterministic output (same seed -> same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr diff --quantization` in-process (no shell-out)
//! 10. [x] Unit tests cover size reduction, dtype enum, mixed-precision
//!
//! ## Learning Objective
//! Compares two model variants by dtype distribution and size savings.
//! Produces a per-tensor report (old dtype -> new dtype, bytes saved) and a
//! whole-model summary. This is the quantization axis of `apr diff`.
//!
//! ## Run Command
//! ```bash
//! cargo run --example diff_quantization
//! ```
//!
//! ## References
//! - Jacob, B. et al. (2018). *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference*. CVPR. arXiv:1712.05877

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dtype {
    Fp32,
    Fp16,
    Int8,
    Int4,
}

impl Dtype {
    fn bytes(self) -> f64 {
        match self {
            Self::Fp32 => 4.0,
            Self::Fp16 => 2.0,
            Self::Int8 => 1.0,
            Self::Int4 => 0.5,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
        }
    }
}

#[derive(Debug, Clone)]
struct TensorSpec {
    name: String,
    n_elements: usize,
    dtype: Dtype,
}

#[derive(Debug, Clone)]
struct QuantChange {
    name: String,
    n_elements: usize,
    old_dtype: Dtype,
    new_dtype: Dtype,
    old_bytes: usize,
    new_bytes: usize,
    savings_bytes: i64, // can be negative (unlikely but possible)
}

#[derive(Debug, Clone, Default)]
struct QuantSummary {
    total_old_bytes: usize,
    total_new_bytes: usize,
    total_savings_bytes: i64,
    savings_pct: f32,
    changes: Vec<QuantChange>,
}

// ---------------------------------------------------------------------------
// Logic
// ---------------------------------------------------------------------------

fn diff_quant(old: &[TensorSpec], new: &[TensorSpec]) -> QuantSummary {
    let new_map: BTreeMap<&str, &TensorSpec> = new.iter().map(|t| (t.name.as_str(), t)).collect();
    let mut summary = QuantSummary::default();
    for o in old {
        if let Some(n) = new_map.get(o.name.as_str()) {
            if o.n_elements == n.n_elements {
                let ob = (o.n_elements as f64 * o.dtype.bytes()) as usize;
                let nb = (n.n_elements as f64 * n.dtype.bytes()) as usize;
                let savings = ob as i64 - nb as i64;
                summary.changes.push(QuantChange {
                    name: o.name.clone(),
                    n_elements: o.n_elements,
                    old_dtype: o.dtype,
                    new_dtype: n.dtype,
                    old_bytes: ob,
                    new_bytes: nb,
                    savings_bytes: savings,
                });
                summary.total_old_bytes += ob;
                summary.total_new_bytes += nb;
            }
        }
    }
    summary.total_savings_bytes = summary.total_old_bytes as i64 - summary.total_new_bytes as i64;
    summary.savings_pct = if summary.total_old_bytes > 0 {
        (summary.total_savings_bytes as f32 / summary.total_old_bytes as f32) * 100.0
    } else {
        0.0
    };
    summary
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("diff_quantization")?;
    println!("=== Recipe: {} ===", ctx.name());

    let old = vec![
        TensorSpec {
            name: "embed".into(),
            n_elements: 1024 * 128,
            dtype: Dtype::Fp32,
        },
        TensorSpec {
            name: "attn.qkv".into(),
            n_elements: 128 * 384,
            dtype: Dtype::Fp32,
        },
        TensorSpec {
            name: "ffn.up".into(),
            n_elements: 128 * 512,
            dtype: Dtype::Fp32,
        },
        TensorSpec {
            name: "ffn.down".into(),
            n_elements: 512 * 128,
            dtype: Dtype::Fp32,
        },
        TensorSpec {
            name: "ln.scale".into(),
            n_elements: 128,
            dtype: Dtype::Fp32,
        },
    ];
    let new = vec![
        TensorSpec {
            name: "embed".into(),
            n_elements: 1024 * 128,
            dtype: Dtype::Int8,
        },
        TensorSpec {
            name: "attn.qkv".into(),
            n_elements: 128 * 384,
            dtype: Dtype::Int4,
        },
        TensorSpec {
            name: "ffn.up".into(),
            n_elements: 128 * 512,
            dtype: Dtype::Fp16,
        },
        TensorSpec {
            name: "ffn.down".into(),
            n_elements: 512 * 128,
            dtype: Dtype::Fp16,
        },
        TensorSpec {
            name: "ln.scale".into(),
            n_elements: 128,
            dtype: Dtype::Fp32,
        },
    ];

    let summary = diff_quant(&old, &new);

    println!("\n--- Quantization Diff ---");
    println!(
        "{:>14} {:>10} {:>6} {:>6} {:>12} {:>12} {:>12}",
        "Name", "Elements", "Old", "New", "OldBytes", "NewBytes", "Savings"
    );
    for c in &summary.changes {
        println!(
            "{:>14} {:>10} {:>6} {:>6} {:>12} {:>12} {:>+12}",
            c.name,
            c.n_elements,
            c.old_dtype.label(),
            c.new_dtype.label(),
            c.old_bytes,
            c.new_bytes,
            c.savings_bytes
        );
    }

    println!(
        "\nTotal old: {} bytes, new: {} bytes, savings: {} bytes ({:.1}%)",
        summary.total_old_bytes,
        summary.total_new_bytes,
        summary.total_savings_bytes,
        summary.savings_pct
    );

    // Sanity: mixed quantization must yield positive savings.
    assert!(summary.total_savings_bytes > 0);

    let out = json!({
        "recipe": ctx.name(),
        "total_old_bytes": summary.total_old_bytes,
        "total_new_bytes": summary.total_new_bytes,
        "total_savings_bytes": summary.total_savings_bytes,
        "savings_pct": summary.savings_pct,
        "changes": summary.changes.iter().map(|c| json!({
            "name": c.name,
            "n_elements": c.n_elements,
            "old_dtype": c.old_dtype.label(),
            "new_dtype": c.new_dtype.label(),
            "old_bytes": c.old_bytes,
            "new_bytes": c.new_bytes,
            "savings_bytes": c.savings_bytes,
        })).collect::<Vec<_>>(),
    });
    let out_path = ctx.path("quant-diff.json");
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

    #[test]
    fn test_bytes_per_dtype() {
        assert!((Dtype::Fp32.bytes() - 4.0).abs() < 1e-9);
        assert!((Dtype::Fp16.bytes() - 2.0).abs() < 1e-9);
        assert!((Dtype::Int8.bytes() - 1.0).abs() < 1e-9);
        assert!((Dtype::Int4.bytes() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_no_dtype_change_zero_savings() {
        let a = vec![TensorSpec {
            name: "t".into(),
            n_elements: 1000,
            dtype: Dtype::Fp32,
        }];
        let b = a.clone();
        let s = diff_quant(&a, &b);
        assert_eq!(s.total_savings_bytes, 0);
        assert!((s.savings_pct - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_fp32_to_int8_saves_75pct() {
        let a = vec![TensorSpec {
            name: "t".into(),
            n_elements: 1000,
            dtype: Dtype::Fp32,
        }];
        let b = vec![TensorSpec {
            name: "t".into(),
            n_elements: 1000,
            dtype: Dtype::Int8,
        }];
        let s = diff_quant(&a, &b);
        // fp32 -> int8 is 4->1 bytes per elem, so 75% savings.
        assert!((s.savings_pct - 75.0).abs() < 0.1);
    }

    #[test]
    fn test_fp32_to_int4_saves_87_5pct() {
        let a = vec![TensorSpec {
            name: "t".into(),
            n_elements: 1000,
            dtype: Dtype::Fp32,
        }];
        let b = vec![TensorSpec {
            name: "t".into(),
            n_elements: 1000,
            dtype: Dtype::Int4,
        }];
        let s = diff_quant(&a, &b);
        assert!((s.savings_pct - 87.5).abs() < 0.1);
    }

    #[test]
    fn test_element_count_mismatch_skipped() {
        let a = vec![TensorSpec {
            name: "t".into(),
            n_elements: 100,
            dtype: Dtype::Fp32,
        }];
        let b = vec![TensorSpec {
            name: "t".into(),
            n_elements: 200,
            dtype: Dtype::Int8,
        }];
        let s = diff_quant(&a, &b);
        assert!(s.changes.is_empty());
    }

    #[test]
    fn test_missing_tensor_skipped() {
        let a = vec![TensorSpec {
            name: "a".into(),
            n_elements: 10,
            dtype: Dtype::Fp32,
        }];
        let b = vec![TensorSpec {
            name: "b".into(),
            n_elements: 10,
            dtype: Dtype::Int8,
        }];
        let s = diff_quant(&a, &b);
        assert!(s.changes.is_empty());
    }
}
