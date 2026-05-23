//! # apr rosetta diff-tensors — Layout Mismatch Check
//!
//! `apr rosetta diff-tensors <REF> <TEST>` detects the GH-186 class of
//! bug: GGML stores weights as `[in_dim, out_dim]` while most ML
//! frameworks expect `[out_dim, in_dim]`. A model that ships with the
//! wrong convention produces garbage output (PAD token floods) but
//! passes every structural check. This recipe builds the layout-check
//! pure function and asserts the contract.
//!
//! Demonstrates the **ROSETTA-DIFF.1** recipe for PMAT-097 (apr rosetta diff-tensors coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender GH-188 + GGML weight-layout convention
//!
//! Run with: cargo run --example cli_rosetta_diff_tensors_layout_check
//!
//! Added by PMAT-097 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    pub name: String,
    pub dims: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutFinding {
    DimsTransposed {
        name: String,
        ref_dims: Vec<u64>,
        test_dims: Vec<u64>,
    },
    RankMismatch {
        name: String,
        ref_rank: usize,
        test_rank: usize,
    },
    AbsentInTest {
        name: String,
    },
    ExtraInTest {
        name: String,
    },
}

pub fn diff_layouts(reference: &[TensorShape], test: &[TensorShape]) -> Vec<LayoutFinding> {
    let mut out = Vec::new();
    for r in reference {
        let Some(t) = test.iter().find(|x| x.name == r.name) else {
            out.push(LayoutFinding::AbsentInTest {
                name: r.name.clone(),
            });
            continue;
        };
        if r.dims.len() != t.dims.len() {
            out.push(LayoutFinding::RankMismatch {
                name: r.name.clone(),
                ref_rank: r.dims.len(),
                test_rank: t.dims.len(),
            });
            continue;
        }
        if r.dims == t.dims {
            continue;
        }
        // Same multiset of dims but different order = transposed layout.
        let mut a = r.dims.clone();
        let mut b = t.dims.clone();
        a.sort_unstable();
        b.sort_unstable();
        if a == b {
            out.push(LayoutFinding::DimsTransposed {
                name: r.name.clone(),
                ref_dims: r.dims.clone(),
                test_dims: t.dims.clone(),
            });
        }
    }
    for t in test {
        if !reference.iter().any(|x| x.name == t.name) {
            out.push(LayoutFinding::ExtraInTest {
                name: t.name.clone(),
            });
        }
    }
    out
}

pub fn filter_mismatches(findings: Vec<LayoutFinding>) -> Vec<LayoutFinding> {
    findings
        .into_iter()
        .filter(|f| {
            !matches!(
                f,
                LayoutFinding::AbsentInTest { .. } | LayoutFinding::ExtraInTest { .. }
            )
        })
        .collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_diff_tensors_layout_check")?;

    let reference = vec![
        TensorShape {
            name: "model.layers.0.q_proj.weight".into(),
            dims: vec![3584, 3584],
        },
        TensorShape {
            name: "model.layers.0.k_proj.weight".into(),
            dims: vec![512, 3584],
        },
        TensorShape {
            name: "lm_head.weight".into(),
            dims: vec![152064, 3584],
        },
    ];
    let test_transposed = vec![
        TensorShape {
            name: "model.layers.0.q_proj.weight".into(),
            dims: vec![3584, 3584],
        },
        TensorShape {
            name: "model.layers.0.k_proj.weight".into(),
            dims: vec![3584, 512],
        }, // ⚠
        TensorShape {
            name: "lm_head.weight".into(),
            dims: vec![3584, 152064],
        }, // ⚠
    ];

    let findings = diff_layouts(&reference, &test_transposed);
    println!("=== Recipe: cli_rosetta_diff_tensors_layout_check ===");
    println!("findings: {}", findings.len());
    for f in &findings {
        println!("  {f:?}");
    }
    println!(
        "\n--mismatches-only filter: {}",
        filter_mismatches(findings).len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(n: &str, d: Vec<u64>) -> TensorShape {
        TensorShape {
            name: n.into(),
            dims: d,
        }
    }

    #[test]
    fn layout_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn identical_shapes_have_no_findings() {
        let a = vec![ts("a", vec![3, 4])];
        let b = vec![ts("a", vec![3, 4])];
        assert!(diff_layouts(&a, &b).is_empty());
    }

    #[test]
    fn transposed_dims_flagged() {
        // The GH-186 case: same multiset of dims, different order.
        let a = vec![ts("k_proj", vec![512, 3584])];
        let b = vec![ts("k_proj", vec![3584, 512])];
        let f = diff_layouts(&a, &b);
        assert_eq!(f.len(), 1);
        assert!(matches!(f[0], LayoutFinding::DimsTransposed { .. }));
    }

    #[test]
    fn rank_mismatch_distinct_from_transposition() {
        // [3, 4] vs [3, 4, 5] is a rank-3 vs rank-2 issue, not a transpose.
        let a = vec![ts("x", vec![3, 4])];
        let b = vec![ts("x", vec![3, 4, 5])];
        let f = diff_layouts(&a, &b);
        assert_eq!(f.len(), 1);
        assert!(matches!(f[0], LayoutFinding::RankMismatch { .. }));
    }

    #[test]
    fn missing_tensor_flagged_as_absent_in_test() {
        let a = vec![ts("x", vec![3, 4]), ts("y", vec![1])];
        let b = vec![ts("x", vec![3, 4])];
        let f = diff_layouts(&a, &b);
        assert!(f
            .iter()
            .any(|x| matches!(x, LayoutFinding::AbsentInTest { name } if name == "y")));
    }

    #[test]
    fn extra_tensor_flagged() {
        // Test model has tensors not in reference — operator probably forgot
        // to filter or used the wrong checkpoint.
        let a = vec![ts("x", vec![3, 4])];
        let b = vec![ts("x", vec![3, 4]), ts("z", vec![1])];
        let f = diff_layouts(&a, &b);
        assert!(f
            .iter()
            .any(|x| matches!(x, LayoutFinding::ExtraInTest { name } if name == "z")));
    }

    #[test]
    fn mismatches_only_filter_drops_absent_and_extra() {
        let a = vec![ts("x", vec![3, 4]), ts("y", vec![1])];
        let b = vec![ts("x", vec![4, 3]), ts("z", vec![1])];
        let all = diff_layouts(&a, &b);
        let only_mm = filter_mismatches(all);
        assert!(only_mm.iter().all(|f| matches!(
            f,
            LayoutFinding::DimsTransposed { .. } | LayoutFinding::RankMismatch { .. }
        )));
    }

    #[test]
    fn different_dims_not_transposed_yields_no_finding() {
        // [3, 4] vs [5, 6] is just different — the lint focuses on the
        // transposition footgun, not arbitrary shape changes.
        let a = vec![ts("x", vec![3, 4])];
        let b = vec![ts("x", vec![5, 6])];
        let f = diff_layouts(&a, &b);
        assert!(f.is_empty());
    }
}
