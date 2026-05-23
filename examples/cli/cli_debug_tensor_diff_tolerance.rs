//! # apr debug --diff-tolerance — Tensor-Element Tolerance Validator
//!
//! `apr debug diff` compares two tensors element-wise. Tolerance:
//! atol (absolute, e.g., 1e-7) + rtol (relative, e.g., 1e-5). Match
//! iff |a − b| ≤ atol + rtol · |b|. NaN matches NaN; ±inf matches
//! same-signed inf. This recipe builds the validator.
//!
//! Demonstrates the **DBG.6** recipe for PMAT-117 (apr debug coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DBG-001 + numpy.allclose semantics
//!
//! Run with: cargo run --example cli_debug_tensor_diff_tolerance
//!
//! Added by PMAT-117 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DiffVerdict {
    Match,
    Mismatch { index: usize, a: f64, b: f64 },
    LengthMismatch { left: usize, right: usize },
    Empty,
}

pub fn close(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    (a - b).abs() <= atol + rtol * b.abs()
}

pub fn diff(a: &[f64], b: &[f64], atol: f64, rtol: f64) -> DiffVerdict {
    if a.is_empty() && b.is_empty() {
        return DiffVerdict::Empty;
    }
    if a.len() != b.len() {
        return DiffVerdict::LengthMismatch {
            left: a.len(),
            right: b.len(),
        };
    }
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        if !close(*x, *y, atol, rtol) {
            return DiffVerdict::Mismatch {
                index: i,
                a: *x,
                b: *y,
            };
        }
    }
    DiffVerdict::Match
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_debug_tensor_diff_tolerance")?;

    let a = [1.0, 2.0, 3.0];
    let b = [1.0 + 1e-9, 2.0, 3.0 - 1e-9];
    println!("close: {:?}", diff(&a, &b, 1e-7, 1e-5));
    let c = [1.0, 2.0, 5.0];
    println!("far:   {:?}", diff(&a, &c, 1e-7, 1e-5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn identical_tensors_match() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0, 3.0];
        assert_eq!(diff(&a, &b, 0.0, 0.0), DiffVerdict::Match);
    }

    #[test]
    fn within_tolerance_match() {
        let a = [1.0, 2.0];
        let b = [1.0 + 1e-9, 2.0 - 1e-9];
        assert_eq!(diff(&a, &b, 1e-7, 1e-5), DiffVerdict::Match);
    }

    #[test]
    fn beyond_tolerance_mismatch() {
        let a = [1.0, 2.0];
        let b = [1.0, 5.0];
        let v = diff(&a, &b, 1e-7, 1e-5);
        assert!(matches!(v, DiffVerdict::Mismatch { index: 1, .. }));
    }

    #[test]
    fn length_mismatch_detected() {
        let a = [1.0, 2.0];
        let b = [1.0];
        let v = diff(&a, &b, 0.0, 0.0);
        assert!(matches!(v, DiffVerdict::LengthMismatch { .. }));
    }

    #[test]
    fn empty_inputs_yield_empty() {
        assert_eq!(diff(&[], &[], 0.0, 0.0), DiffVerdict::Empty);
    }

    #[test]
    fn nan_matches_nan() {
        assert!(close(f64::NAN, f64::NAN, 0.0, 0.0));
    }

    #[test]
    fn pos_inf_matches_pos_inf() {
        assert!(close(f64::INFINITY, f64::INFINITY, 0.0, 0.0));
    }

    #[test]
    fn pos_inf_does_not_match_neg_inf() {
        assert!(!close(f64::INFINITY, f64::NEG_INFINITY, 0.0, 0.0));
    }

    #[test]
    fn rtol_scales_with_value() {
        // 1% relative tolerance: 1.0 vs 1.005 OK, 100.0 vs 100.5 OK.
        assert!(close(1.0, 1.005, 0.0, 0.01));
        assert!(close(100.0, 100.5, 0.0, 0.01));
        // But 1.0 vs 1.5 → not OK.
        assert!(!close(1.0, 1.5, 0.0, 0.01));
    }

    #[test]
    fn first_mismatch_index_reported() {
        // Mismatch at idx 1, then more at idx 2.
        let a = [0.0, 1.0, 2.0, 3.0];
        let b = [0.0, 5.0, 10.0, 15.0];
        let v = diff(&a, &b, 0.0, 0.0);
        if let DiffVerdict::Mismatch { index, .. } = v {
            assert_eq!(index, 1);
        }
    }
}
