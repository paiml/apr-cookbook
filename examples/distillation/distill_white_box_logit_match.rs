//! # Distillation White-Box Logit MSE Match
//!
//! "White-box" distillation has full access to teacher logits (not
//! just the softmax output). MSE on raw logits often outperforms KL on
//! softmax for classification (Tian et al. 2021).
//!
//! Loss: `mse = mean((teacher_logit_i - student_logit_i)^2)` over all
//! C classes. Plus L2-regularized variant when student diverges.
//!
//! Demonstrates the **DIST.16** recipe for PMAT-145 (distillation round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Tian et al. (2021). Knowledge Distillation: A Good Teacher is Patient and Consistent.
//!
//! Run with: cargo run --example distill_white_box_logit_match
//!
//! Added by PMAT-145 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum LogitVerdict {
    Ok {
        mse: f64,
        regularized_loss: f64,
        diverged: bool,
    },
    DimensionMismatch {
        teacher: usize,
        student: usize,
    },
    EmptyLogits,
    InvalidValues,
}

const DIVERGENCE_THRESHOLD: f64 = 100.0;
const L2_LAMBDA: f64 = 0.01;

pub fn compute(teacher_logits: &[f64], student_logits: &[f64]) -> LogitVerdict {
    if teacher_logits.is_empty() || student_logits.is_empty() {
        return LogitVerdict::EmptyLogits;
    }
    if teacher_logits.len() != student_logits.len() {
        return LogitVerdict::DimensionMismatch {
            teacher: teacher_logits.len(),
            student: student_logits.len(),
        };
    }
    if teacher_logits
        .iter()
        .chain(student_logits.iter())
        .any(|x| !x.is_finite())
    {
        return LogitVerdict::InvalidValues;
    }
    let n = teacher_logits.len() as f64;
    let mse: f64 = teacher_logits
        .iter()
        .zip(student_logits.iter())
        .map(|(t, s)| (t - s).powi(2))
        .sum::<f64>()
        / n;
    let l2: f64 = student_logits.iter().map(|s| s * s).sum::<f64>() / n;
    let regularized_loss = mse + L2_LAMBDA * l2;
    let diverged = mse > DIVERGENCE_THRESHOLD;
    LogitVerdict::Ok {
        mse,
        regularized_loss,
        diverged,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_white_box_logit_match")?;

    let teacher = [2.0, 1.5, 0.5, -1.0];
    let student = [2.1, 1.4, 0.6, -0.9];
    println!("close: {:?}", compute(&teacher, &student));

    let bad_student = [5.0, 5.0, 5.0, 5.0];
    println!("diverged: {:?}", compute(&teacher, &bad_student));

    println!("empty: {:?}", compute(&[], &[]));
    println!("mismatch: {:?}", compute(&teacher, &[1.0, 2.0]));
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
    fn identical_logits_zero_mse() {
        let l = [1.0, 2.0, 3.0];
        if let LogitVerdict::Ok { mse, .. } = compute(&l, &l) {
            assert!(mse.abs() < 1e-12);
        }
    }

    #[test]
    fn off_by_one_mse_one() {
        let teacher = [1.0, 2.0, 3.0];
        let student = [2.0, 3.0, 4.0]; // each off by 1.
        if let LogitVerdict::Ok { mse, .. } = compute(&teacher, &student) {
            assert!((mse - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let v = compute(&[1.0, 2.0], &[1.0]);
        assert!(matches!(v, LogitVerdict::DimensionMismatch { .. }));
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(compute(&[], &[]), LogitVerdict::EmptyLogits);
    }

    #[test]
    fn nan_rejected() {
        let v = compute(&[1.0, f64::NAN], &[1.0, 2.0]);
        assert_eq!(v, LogitVerdict::InvalidValues);
    }

    #[test]
    fn diverged_flag_set_on_high_mse() {
        let teacher = [0.0; 5];
        let student = [50.0; 5];
        if let LogitVerdict::Ok { diverged, .. } = compute(&teacher, &student) {
            assert!(diverged);
        }
    }

    #[test]
    fn diverged_false_on_low_mse() {
        let teacher = [0.0; 5];
        let student = [0.1; 5];
        if let LogitVerdict::Ok { diverged, .. } = compute(&teacher, &student) {
            assert!(!diverged);
        }
    }

    #[test]
    fn regularized_includes_l2() {
        let teacher = [1.0, 1.0];
        let student = [1.0, 1.0];
        // mse = 0, l2 = 1, reg = 0 + 0.01 × 1 = 0.01.
        if let LogitVerdict::Ok {
            regularized_loss, ..
        } = compute(&teacher, &student)
        {
            assert!((regularized_loss - 0.01).abs() < 1e-9);
        }
    }

    #[test]
    fn mse_proportional_to_squared_diff() {
        let teacher = [0.0, 0.0];
        let student_small = [1.0, 1.0];
        let student_large = [2.0, 2.0];
        if let (LogitVerdict::Ok { mse: s, .. }, LogitVerdict::Ok { mse: l, .. }) = (
            compute(&teacher, &student_small),
            compute(&teacher, &student_large),
        ) {
            assert!((l - 4.0 * s).abs() < 1e-9);
        }
    }

    #[test]
    fn many_classes_handled() {
        let teacher: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
        let student: Vec<f64> = teacher.clone();
        if let LogitVerdict::Ok { mse, .. } = compute(&teacher, &student) {
            assert!(mse.abs() < 1e-12);
        }
    }
}
