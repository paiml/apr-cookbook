//! # apr diff — Tensor Shape Compatibility Validator
//!
//! `apr diff <A> <B>` requires per-tensor shape compatibility. Rules:
//! identical names + identical shapes = comparable; same name +
//! different shape = ShapeMismatch (often signals architecture
//! divergence); name-only-on-one-side = NameOnlyIn(A|B). This recipe
//! builds the matcher.
//!
//! Demonstrates the **DIFF.2** recipe for PMAT-118 (apr diff coverage —
//! closing F-invariant gap from 1 → 4 recipes).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DIFF-001
//!
//! Run with: cargo run --example cli_diff_shape_compatibility
//!
//! Added by PMAT-118 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::HashMap;

#[derive(Debug, PartialEq, Eq)]
pub struct TensorRef {
    pub name: String,
    pub shape: Vec<u32>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum ShapeStatus {
    Comparable,
    ShapeMismatch { left: Vec<u32>, right: Vec<u32> },
    NameOnlyInLeft,
    NameOnlyInRight,
}

pub fn diff_shapes(left: &[TensorRef], right: &[TensorRef]) -> Vec<(String, ShapeStatus)> {
    let mut out = Vec::new();
    let lmap: HashMap<&str, &Vec<u32>> = left.iter().map(|t| (t.name.as_str(), &t.shape)).collect();
    let rmap: HashMap<&str, &Vec<u32>> =
        right.iter().map(|t| (t.name.as_str(), &t.shape)).collect();
    let mut all_names: Vec<&str> = lmap.keys().chain(rmap.keys()).copied().collect();
    all_names.sort_unstable();
    all_names.dedup();
    for name in all_names {
        let status = match (lmap.get(name), rmap.get(name)) {
            (Some(ls), Some(rs)) if ls == rs => ShapeStatus::Comparable,
            (Some(ls), Some(rs)) => ShapeStatus::ShapeMismatch {
                left: (*ls).clone(),
                right: (*rs).clone(),
            },
            (Some(_), None) => ShapeStatus::NameOnlyInLeft,
            (None, Some(_)) => ShapeStatus::NameOnlyInRight,
            (None, None) => continue,
        };
        out.push((name.to_string(), status));
    }
    out
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_diff_shape_compatibility")?;

    let left = vec![
        TensorRef {
            name: "embed.weight".into(),
            shape: vec![32_000, 4096],
        },
        TensorRef {
            name: "layer.0.attn.q".into(),
            shape: vec![4096, 4096],
        },
        TensorRef {
            name: "removed.in.right".into(),
            shape: vec![1],
        },
    ];
    let right = vec![
        TensorRef {
            name: "embed.weight".into(),
            shape: vec![32_000, 4096],
        },
        TensorRef {
            name: "layer.0.attn.q".into(),
            shape: vec![4096, 5120],
        },
        TensorRef {
            name: "added.in.right".into(),
            shape: vec![1],
        },
    ];
    for (name, st) in diff_shapes(&left, &right) {
        println!("{name:<32} → {st:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(name: &str, shape: Vec<u32>) -> TensorRef {
        TensorRef {
            name: name.into(),
            shape,
        }
    }

    #[test]
    fn diff_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn identical_tensors_comparable() {
        let l = vec![t("a", vec![10, 20])];
        let r = vec![t("a", vec![10, 20])];
        let d = diff_shapes(&l, &r);
        assert_eq!(d, vec![("a".into(), ShapeStatus::Comparable)]);
    }

    #[test]
    fn shape_mismatch_detected() {
        let l = vec![t("a", vec![10, 20])];
        let r = vec![t("a", vec![10, 30])];
        let d = diff_shapes(&l, &r);
        assert!(matches!(d[0].1, ShapeStatus::ShapeMismatch { .. }));
    }

    #[test]
    fn name_only_in_left_detected() {
        let l = vec![t("only_left", vec![1])];
        let r: Vec<TensorRef> = vec![];
        let d = diff_shapes(&l, &r);
        assert_eq!(d, vec![("only_left".into(), ShapeStatus::NameOnlyInLeft)]);
    }

    #[test]
    fn name_only_in_right_detected() {
        let l: Vec<TensorRef> = vec![];
        let r = vec![t("only_right", vec![1])];
        let d = diff_shapes(&l, &r);
        assert_eq!(d, vec![("only_right".into(), ShapeStatus::NameOnlyInRight)]);
    }

    #[test]
    fn empty_inputs_yield_empty() {
        assert!(diff_shapes(&[], &[]).is_empty());
    }

    #[test]
    fn output_sorted_by_name() {
        let l = vec![t("z", vec![1]), t("a", vec![1])];
        let r = vec![t("m", vec![1])];
        let d = diff_shapes(&l, &r);
        let names: Vec<&str> = d.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["a", "m", "z"]);
    }

    #[test]
    fn mixed_diff_returns_all_categories() {
        let l = vec![t("same", vec![1]), t("only_l", vec![2])];
        let r = vec![t("same", vec![1]), t("only_r", vec![2])];
        let d = diff_shapes(&l, &r);
        assert_eq!(d.len(), 3);
    }

    #[test]
    fn higher_dim_mismatch_detected() {
        let l = vec![t("a", vec![1, 2, 3])];
        let r = vec![t("a", vec![1, 2])];
        let d = diff_shapes(&l, &r);
        assert!(matches!(d[0].1, ShapeStatus::ShapeMismatch { .. }));
    }
}
