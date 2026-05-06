//! # Distillation Intermediate Feature Matching
//!
//! Match student intermediate features to teacher's via MSE. When
//! dimensions differ, project student features through a linear
//! adapter to teacher dim. This recipe builds the adapter shape
//! checker + MSE loss calculator.
//!
//! Demonstrates the **DIST.15** recipe for PMAT-141 (distillation round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Romero et al. (2014). FitNets: Hints for Thin Deep Nets. arXiv:1412.6550.
//!
//! Run with: cargo run --example distill_intermediate_feature_match
//!
//! Added by PMAT-141 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FeatureVerdict {
    Ok {
        mse: f64,
        adapter_required: bool,
        adapter_dim: Option<(usize, usize)>,
    },
    DimensionMismatch {
        teacher: usize,
        student: usize,
    },
    EmptyFeatures,
    InvalidValues,
}

pub fn compute(
    teacher: &[f64],
    student: &[f64],
    teacher_dim: usize,
    student_dim: usize,
) -> FeatureVerdict {
    if teacher.is_empty() || student.is_empty() {
        return FeatureVerdict::EmptyFeatures;
    }
    if teacher.len() != teacher_dim || student.len() != student_dim {
        return FeatureVerdict::DimensionMismatch {
            teacher: teacher_dim,
            student: student_dim,
        };
    }
    if teacher.iter().chain(student.iter()).any(|x| !x.is_finite()) {
        return FeatureVerdict::InvalidValues;
    }
    let adapter_required = teacher_dim != student_dim;
    let adapter_dim = if adapter_required {
        Some((student_dim, teacher_dim))
    } else {
        None
    };
    // When dims match, MSE directly. When dims differ, return zero MSE
    // (real implementation would apply learned adapter; this recipe is
    // the planning stage).
    let mse = if adapter_required {
        0.0
    } else {
        let n = teacher.len() as f64;
        teacher
            .iter()
            .zip(student.iter())
            .map(|(t, s)| (t - s).powi(2))
            .sum::<f64>()
            / n
    };
    FeatureVerdict::Ok {
        mse,
        adapter_required,
        adapter_dim,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_intermediate_feature_match")?;

    let teacher = [1.0, 2.0, 3.0, 4.0];
    let student = [1.1, 2.1, 2.9, 4.05];
    println!("matching dims: {:?}", compute(&teacher, &student, 4, 4));

    let small_student = [1.0, 2.0];
    println!(
        "needs adapter: {:?}",
        compute(&teacher, &small_student, 4, 2)
    );

    println!("empty: {:?}", compute(&[], &[], 0, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn equal_dims_no_adapter() {
        let v = compute(&[1.0, 2.0], &[1.0, 2.0], 2, 2);
        if let FeatureVerdict::Ok {
            adapter_required, ..
        } = v
        {
            assert!(!adapter_required);
        }
    }

    #[test]
    fn unequal_dims_adapter_required() {
        let v = compute(&[1.0, 2.0, 3.0], &[1.0, 2.0], 3, 2);
        if let FeatureVerdict::Ok {
            adapter_required,
            adapter_dim,
            ..
        } = v
        {
            assert!(adapter_required);
            assert_eq!(adapter_dim, Some((2, 3)));
        }
    }

    #[test]
    fn identical_features_zero_mse() {
        let f = [1.0, 2.0, 3.0];
        if let FeatureVerdict::Ok { mse, .. } = compute(&f, &f, 3, 3) {
            assert!(mse.abs() < 1e-12);
        }
    }

    #[test]
    fn close_features_small_mse() {
        let teacher = [1.0, 2.0, 3.0];
        let student = [1.1, 2.1, 2.9];
        if let FeatureVerdict::Ok { mse, .. } = compute(&teacher, &student, 3, 3) {
            assert!(mse < 0.1);
        }
    }

    #[test]
    fn divergent_features_large_mse() {
        let teacher = [1.0, 2.0, 3.0];
        let student = [-3.0, 0.0, 5.0];
        if let FeatureVerdict::Ok { mse, .. } = compute(&teacher, &student, 3, 3) {
            assert!(mse > 1.0);
        }
    }

    #[test]
    fn empty_features_rejected() {
        assert_eq!(compute(&[], &[], 0, 0), FeatureVerdict::EmptyFeatures);
    }

    #[test]
    fn shape_mismatch_rejected() {
        let v = compute(&[1.0], &[1.0], 2, 2);
        assert!(matches!(v, FeatureVerdict::DimensionMismatch { .. }));
    }

    #[test]
    fn nan_features_rejected() {
        let v = compute(&[1.0, f64::NAN], &[1.0, 2.0], 2, 2);
        assert_eq!(v, FeatureVerdict::InvalidValues);
    }

    #[test]
    fn adapter_dim_for_2_to_3() {
        let v = compute(&[1.0, 2.0, 3.0], &[1.0, 2.0], 3, 2);
        if let FeatureVerdict::Ok { adapter_dim, .. } = v {
            assert_eq!(adapter_dim, Some((2, 3)));
        }
    }

    #[test]
    fn adapter_dim_none_when_equal() {
        let v = compute(&[1.0, 2.0], &[1.0, 2.0], 2, 2);
        if let FeatureVerdict::Ok { adapter_dim, .. } = v {
            assert_eq!(adapter_dim, None);
        }
    }

    #[test]
    fn mse_sum_matches_formula() {
        let teacher = [1.0, 2.0, 3.0, 4.0];
        let student = [2.0, 3.0, 4.0, 5.0];
        // Each diff = -1, square = 1, sum = 4, MSE = 1.
        if let FeatureVerdict::Ok { mse, .. } = compute(&teacher, &student, 4, 4) {
            assert!((mse - 1.0).abs() < 1e-9);
        }
    }
}
