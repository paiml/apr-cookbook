//! # apr rosetta compare-inference — Per-Token Logit Diff
//!
//! `apr rosetta compare-inference` compares per-token next-token logit
//! distributions and reports which positions exceed the divergence
//! tolerance. This recipe builds the diff function as L∞ over per-vocab
//! logit values and asserts the comparison contract: empty distributions
//! return None (not 0), shape mismatches return an explicit error, and
//! NaN at any position propagates through.
//!
//! Demonstrates the **ROSETTA-CMP.2** recipe for PMAT-096 (apr rosetta compare-inference coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-114 + L∞ logit divergence convention
//!
//! Run with: cargo run --example cli_rosetta_compare_inference_logit_diff
//!
//! Added by PMAT-096 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub enum DiffResult {
    Match,
    Diverged { position: usize, l_inf: f64 },
    ShapeMismatch { len_a: usize, len_b: usize },
    HasNaN { position: usize },
    Empty,
}

pub fn compare_token_logits(
    logits_a: &[Vec<f64>],
    logits_b: &[Vec<f64>],
    tolerance: f64,
) -> DiffResult {
    if logits_a.is_empty() || logits_b.is_empty() {
        return DiffResult::Empty;
    }
    if logits_a.len() != logits_b.len() {
        return DiffResult::ShapeMismatch {
            len_a: logits_a.len(),
            len_b: logits_b.len(),
        };
    }

    for (pos, (a, b)) in logits_a.iter().zip(logits_b).enumerate() {
        if a.len() != b.len() {
            return DiffResult::ShapeMismatch {
                len_a: a.len(),
                len_b: b.len(),
            };
        }
        let mut max_diff = 0.0f64;
        for (x, y) in a.iter().zip(b) {
            if x.is_nan() || y.is_nan() {
                return DiffResult::HasNaN { position: pos };
            }
            let d = (x - y).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        if max_diff > tolerance {
            return DiffResult::Diverged {
                position: pos,
                l_inf: max_diff,
            };
        }
    }
    DiffResult::Match
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_compare_inference_logit_diff")?;

    let a = vec![vec![1.0, 2.0, 3.0], vec![0.5, 0.5, 0.5]];
    let b_match = vec![vec![1.0, 2.0, 3.0], vec![0.5, 0.5, 0.5]];
    let b_close = vec![vec![1.001, 1.999, 3.0001], vec![0.5, 0.5, 0.5]];
    let b_diverged = vec![vec![1.0, 2.0, 3.0], vec![0.5, 5.0, 0.5]];

    println!(
        "identical:       {:?}",
        compare_token_logits(&a, &b_match, 0.01)
    );
    println!(
        "close (tol 0.1): {:?}",
        compare_token_logits(&a, &b_close, 0.1)
    );
    println!(
        "close (tol 1e-6):{:?}",
        compare_token_logits(&a, &b_close, 1e-6)
    );
    println!(
        "diverged:        {:?}",
        compare_token_logits(&a, &b_diverged, 0.1)
    );
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
    fn identical_logits_match() {
        let a = vec![vec![1.0, 2.0, 3.0]];
        let b = vec![vec![1.0, 2.0, 3.0]];
        assert_eq!(compare_token_logits(&a, &b, 0.0), DiffResult::Match);
    }

    #[test]
    fn within_tolerance_matches() {
        let a = vec![vec![1.0, 2.0]];
        let b = vec![vec![1.05, 1.95]];
        assert_eq!(compare_token_logits(&a, &b, 0.1), DiffResult::Match);
    }

    #[test]
    fn divergence_reports_position_and_l_inf() {
        let a = vec![vec![1.0, 2.0], vec![0.0, 0.0]];
        let b = vec![vec![1.0, 2.0], vec![0.0, 5.0]];
        let r = compare_token_logits(&a, &b, 0.1);
        assert_eq!(
            r,
            DiffResult::Diverged {
                position: 1,
                l_inf: 5.0
            }
        );
    }

    #[test]
    fn shape_mismatch_at_token_level_detected() {
        let a = vec![vec![1.0, 2.0, 3.0]];
        let b = vec![vec![1.0, 2.0]]; // different vocab len
        let r = compare_token_logits(&a, &b, 0.1);
        assert_eq!(r, DiffResult::ShapeMismatch { len_a: 3, len_b: 2 });
    }

    #[test]
    fn shape_mismatch_at_sequence_level_detected() {
        let a = vec![vec![1.0], vec![2.0]];
        let b = vec![vec![1.0]];
        let r = compare_token_logits(&a, &b, 0.1);
        assert_eq!(r, DiffResult::ShapeMismatch { len_a: 2, len_b: 1 });
    }

    #[test]
    fn nan_propagates() {
        let a = vec![vec![1.0, f64::NAN]];
        let b = vec![vec![1.0, 2.0]];
        let r = compare_token_logits(&a, &b, 0.1);
        assert_eq!(r, DiffResult::HasNaN { position: 0 });
    }

    #[test]
    fn empty_input_returns_empty_not_match() {
        assert_eq!(compare_token_logits(&[], &[], 0.1), DiffResult::Empty);
    }
}
