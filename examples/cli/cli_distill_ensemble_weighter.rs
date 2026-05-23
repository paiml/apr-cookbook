//! # apr distill --ensemble — Teacher Weight Validator
//!
//! Ensemble distillation accepts N teachers with explicit per-teacher
//! weights. Constraints: weights sum to 1.0 (within tolerance), all
//! non-negative, length matches teacher count. Uniform default = 1/N.
//! This recipe codifies the validator + uniform fallback.
//!
//! Demonstrates the **DISTILL.6** recipe for PMAT-113 (apr distill coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender DISTILL-001 + Lan et al. 2018 (Knowledge ensemble)
//!
//! Run with: cargo run --example cli_distill_ensemble_weighter
//!
//! Added by PMAT-113 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const SUM_TOLERANCE: f64 = 1e-6;

#[derive(Debug, PartialEq)]
pub enum WeightVerdict {
    Ok,
    LengthMismatch { weights: usize, teachers: usize },
    NegativeWeight { index: usize, value: f64 },
    NonFiniteWeight { index: usize },
    DoesNotSumToOne { sum: f64 },
    EmptyEnsemble,
}

pub fn validate(weights: &[f64], num_teachers: usize) -> WeightVerdict {
    if num_teachers == 0 {
        return WeightVerdict::EmptyEnsemble;
    }
    if weights.len() != num_teachers {
        return WeightVerdict::LengthMismatch {
            weights: weights.len(),
            teachers: num_teachers,
        };
    }
    for (i, w) in weights.iter().enumerate() {
        if !w.is_finite() {
            return WeightVerdict::NonFiniteWeight { index: i };
        }
        if *w < 0.0 {
            return WeightVerdict::NegativeWeight {
                index: i,
                value: *w,
            };
        }
    }
    let sum: f64 = weights.iter().sum();
    if (sum - 1.0).abs() > SUM_TOLERANCE {
        return WeightVerdict::DoesNotSumToOne { sum };
    }
    WeightVerdict::Ok
}

pub fn uniform_default(num_teachers: usize) -> Vec<f64> {
    if num_teachers == 0 {
        return Vec::new();
    }
    let w = 1.0 / num_teachers as f64;
    vec![w; num_teachers]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_distill_ensemble_weighter")?;

    let n = 4;
    let uniform = uniform_default(n);
    println!("uniform({n}): {uniform:?}  →  {:?}", validate(&uniform, n));

    let bad = vec![0.5, 0.3, 0.1, 0.2]; // 1.1 sum
    println!("bad sum: {bad:?}  →  {:?}", validate(&bad, 4));

    let neg = vec![0.5, -0.1, 0.4, 0.2];
    println!("neg: {neg:?}  →  {:?}", validate(&neg, 4));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn uniform_passes() {
        let w = uniform_default(4);
        assert_eq!(validate(&w, 4), WeightVerdict::Ok);
    }

    #[test]
    fn explicit_unequal_passes() {
        let w = vec![0.5, 0.3, 0.2];
        assert_eq!(validate(&w, 3), WeightVerdict::Ok);
    }

    #[test]
    fn length_mismatch_rejected() {
        let w = vec![0.5, 0.5];
        let v = validate(&w, 4);
        assert!(matches!(v, WeightVerdict::LengthMismatch { .. }));
    }

    #[test]
    fn negative_weight_rejected() {
        let w = vec![0.5, -0.1, 0.4, 0.2];
        let v = validate(&w, 4);
        assert!(matches!(v, WeightVerdict::NegativeWeight { index: 1, .. }));
    }

    #[test]
    fn nan_weight_rejected() {
        let w = vec![0.5, f64::NAN, 0.5];
        let v = validate(&w, 3);
        assert!(matches!(v, WeightVerdict::NonFiniteWeight { index: 1 }));
    }

    #[test]
    fn sum_not_one_rejected() {
        let w = vec![0.5, 0.3, 0.1]; // 0.9
        let v = validate(&w, 3);
        assert!(matches!(v, WeightVerdict::DoesNotSumToOne { .. }));
    }

    #[test]
    fn small_tolerance_accepted() {
        // Within 1e-6 of 1.0 should pass.
        let w = vec![0.5, 0.5 + 1e-9];
        assert_eq!(validate(&w, 2), WeightVerdict::Ok);
    }

    #[test]
    fn empty_ensemble_rejected() {
        assert_eq!(validate(&[], 0), WeightVerdict::EmptyEnsemble);
    }

    #[test]
    fn uniform_default_sums_to_one() {
        for n in [1usize, 2, 5, 10, 100] {
            let w = uniform_default(n);
            let sum: f64 = w.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "n={n}, sum={sum}");
        }
    }
}
