//! # apr probar — Golden-Reference Diff
//!
//! `apr probar <FILE> --golden ./goldens --tolerance 0.98 --assert` compares
//! freshly-exported tensors against the committed golden reference using
//! cosine similarity. With `--assert`, divergence below tolerance triggers
//! a non-zero exit code (CI mode). This recipe builds the per-layer diff
//! decision tree as a pure function so the verdict can be previewed
//! without running the full export pipeline.
//!
//! Demonstrates the **PROBAR.4** recipe for PMAT-093 (apr probar coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-481 + cosine-similarity convention
//!
//! Run with: cargo run --example cli_probar_golden_diff
//!
//! Added by PMAT-093 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct LayerDelta {
    pub layer: String,
    pub cosine: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DiffVerdict {
    Pass,
    BelowTolerance(Vec<LayerDelta>),
}

pub fn cosine_sim(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

pub fn diff_against_goldens(deltas: &[LayerDelta], tolerance: f64) -> DiffVerdict {
    let bad: Vec<LayerDelta> = deltas
        .iter()
        .filter(|d| d.cosine < tolerance)
        .cloned()
        .collect();
    if bad.is_empty() {
        DiffVerdict::Pass
    } else {
        DiffVerdict::BelowTolerance(bad)
    }
}

pub fn exit_code(v: &DiffVerdict, assert_mode: bool) -> i32 {
    match (v, assert_mode) {
        (DiffVerdict::Pass, _) => 0,
        (DiffVerdict::BelowTolerance(_), true) => 65, // EX_DATAERR per sysexits.h
        (DiffVerdict::BelowTolerance(_), false) => 0, // report-only
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_probar_golden_diff")?;

    let deltas = [
        LayerDelta {
            layer: "model.embed_tokens".into(),
            cosine: 0.999,
        },
        LayerDelta {
            layer: "model.layers.0.q_proj".into(),
            cosine: 0.985,
        },
        LayerDelta {
            layer: "model.layers.0.k_proj".into(),
            cosine: 0.940,
        }, // ⚠
        LayerDelta {
            layer: "lm_head".into(),
            cosine: 0.998,
        },
    ];

    for tol in [0.95, 0.98, 0.999] {
        let v = diff_against_goldens(&deltas, tol);
        println!(
            "tol={tol:.3}  verdict={v:?}  exit(--assert)={}",
            exit_code(&v, true)
        );
    }

    let same_vec = vec![1.0, 2.0, 3.0];
    println!("\ncosine_sim self: {:.6}", cosine_sim(&same_vec, &same_vec));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diff_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn cosine_self_is_one() {
        let v = vec![0.6, 0.8];
        assert!((cosine_sim(&v, &v) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_sim(&a, &b) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_with_zero_vector_is_zero() {
        // Avoid divide-by-zero — return 0 not NaN.
        let a = vec![1.0, 2.0];
        let z = vec![0.0, 0.0];
        assert_eq!(cosine_sim(&a, &z), 0.0);
        assert_eq!(cosine_sim(&z, &a), 0.0);
    }

    #[test]
    fn all_layers_above_tolerance_passes() {
        let deltas = [
            LayerDelta {
                layer: "a".into(),
                cosine: 0.99,
            },
            LayerDelta {
                layer: "b".into(),
                cosine: 0.995,
            },
        ];
        assert_eq!(diff_against_goldens(&deltas, 0.98), DiffVerdict::Pass);
    }

    #[test]
    fn one_layer_below_tolerance_fails() {
        let deltas = [
            LayerDelta {
                layer: "a".into(),
                cosine: 0.99,
            },
            LayerDelta {
                layer: "b".into(),
                cosine: 0.90,
            },
        ];
        let v = diff_against_goldens(&deltas, 0.98);
        if let DiffVerdict::BelowTolerance(bad) = v {
            assert_eq!(bad.len(), 1);
            assert_eq!(bad[0].layer, "b");
        } else {
            panic!("expected BelowTolerance");
        }
    }

    #[test]
    fn assert_mode_returns_nonzero_on_divergence() {
        let v = DiffVerdict::BelowTolerance(vec![LayerDelta {
            layer: "x".into(),
            cosine: 0.5,
        }]);
        assert_ne!(exit_code(&v, true), 0);
        // Without --assert the exit code is 0 (report-only).
        assert_eq!(exit_code(&v, false), 0);
    }

    #[test]
    fn exact_tolerance_boundary_passes() {
        // cosine == tolerance is conservative-pass (matches PROBAR.4 contract).
        let deltas = [LayerDelta {
            layer: "a".into(),
            cosine: 0.98,
        }];
        assert_eq!(diff_against_goldens(&deltas, 0.98), DiffVerdict::Pass);
    }
}
