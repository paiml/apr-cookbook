//! # Distillation Quantile Calibration
//!
//! Distilled student often produces logits that are more peaked than
//! the teacher (under-calibrated). Fix: temperature-scale the student
//! logits so its quantile distribution matches the teacher's.
//! Procedure: pair (teacher_q, student_q) for q in [0.1, 0.25, 0.5,
//! 0.75, 0.9]; compute scale T that minimizes squared-error of
//! quantiles after dividing by T.
//!
//! Demonstrates the **DIST.13** recipe for PMAT-141 (distillation round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Guo et al. (2017). On Calibration of Modern Neural Networks. arXiv:1706.04599.
//!
//! Run with: cargo run --example distill_quantile_calibration
//!
//! Added by PMAT-141 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CalibrationVerdict {
    Ok { temperature: f64, residual: f64 },
    InvalidQuantiles,
    DegenerateRange,
}

pub fn calibrate(teacher_quantiles: &[f64], student_quantiles: &[f64]) -> CalibrationVerdict {
    if teacher_quantiles.len() != student_quantiles.len()
        || teacher_quantiles.is_empty()
        || teacher_quantiles
            .iter()
            .chain(student_quantiles.iter())
            .any(|x| !x.is_finite())
    {
        return CalibrationVerdict::InvalidQuantiles;
    }
    let teacher_max = teacher_quantiles
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let teacher_min = teacher_quantiles
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    if (teacher_max - teacher_min).abs() < 1e-9 {
        return CalibrationVerdict::DegenerateRange;
    }
    // Closed-form least-squares scale: T* = sum(s × t) / sum(t × t)
    // (s_i = student / T should match t_i, so T = student / teacher).
    let dot: f64 = student_quantiles
        .iter()
        .zip(teacher_quantiles.iter())
        .map(|(s, t)| s * t)
        .sum();
    let teacher_norm_sq: f64 = teacher_quantiles.iter().map(|t| t * t).sum();
    if teacher_norm_sq == 0.0 {
        return CalibrationVerdict::DegenerateRange;
    }
    let temperature = dot / teacher_norm_sq;
    let residual: f64 = student_quantiles
        .iter()
        .zip(teacher_quantiles.iter())
        .map(|(s, t)| {
            let diff = s / temperature - t;
            diff * diff
        })
        .sum();
    CalibrationVerdict::Ok {
        temperature,
        residual,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_quantile_calibration")?;

    // Student is 2× more peaked than teacher.
    let teacher = [0.1, 0.5, 1.0, 1.5, 2.0];
    let student = [0.2, 1.0, 2.0, 3.0, 4.0];
    println!("2x peaked: {:?}", calibrate(&teacher, &student));

    // Already calibrated.
    println!("identical: {:?}", calibrate(&teacher, &teacher));

    println!("empty: {:?}", calibrate(&[], &[]));
    println!("mismatch: {:?}", calibrate(&[0.1, 0.5], &[0.1, 0.5, 1.0]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn two_x_peaked_yields_two() {
        let teacher = [0.1, 0.5, 1.0, 1.5, 2.0];
        let student = [0.2, 1.0, 2.0, 3.0, 4.0];
        if let CalibrationVerdict::Ok { temperature, .. } = calibrate(&teacher, &student) {
            assert!((temperature - 2.0).abs() < 1e-9);
        }
    }

    #[test]
    fn identical_yields_one() {
        let q = [0.1, 0.5, 1.0, 1.5, 2.0];
        if let CalibrationVerdict::Ok { temperature, .. } = calibrate(&q, &q) {
            assert!((temperature - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn degenerate_teacher_rejected() {
        let q = [1.0, 1.0, 1.0];
        let v = calibrate(&q, &[2.0, 2.0, 2.0]);
        assert_eq!(v, CalibrationVerdict::DegenerateRange);
    }

    #[test]
    fn empty_quantiles_rejected() {
        assert_eq!(calibrate(&[], &[]), CalibrationVerdict::InvalidQuantiles);
    }

    #[test]
    fn mismatched_lengths_rejected() {
        let v = calibrate(&[0.1, 0.5], &[0.1, 0.5, 1.0]);
        assert_eq!(v, CalibrationVerdict::InvalidQuantiles);
    }

    #[test]
    fn nan_values_rejected() {
        let teacher = [0.1, f64::NAN, 1.0];
        let student = [0.2, 1.0, 2.0];
        assert_eq!(
            calibrate(&teacher, &student),
            CalibrationVerdict::InvalidQuantiles
        );
    }

    #[test]
    fn residual_zero_when_perfectly_scaled() {
        let teacher = [0.1, 0.5, 1.0, 1.5, 2.0];
        let student = [0.3, 1.5, 3.0, 4.5, 6.0]; // exactly 3× teacher.
        if let CalibrationVerdict::Ok { residual, .. } = calibrate(&teacher, &student) {
            assert!(residual < 1e-9);
        }
    }

    #[test]
    fn residual_positive_for_imperfect() {
        // Non-uniformly perturbed student.
        let teacher = [0.1, 0.5, 1.0];
        let student = [0.5, 0.5, 2.0];
        if let CalibrationVerdict::Ok { residual, .. } = calibrate(&teacher, &student) {
            assert!(residual > 0.0);
        }
    }

    #[test]
    fn temperature_positive_for_positive_inputs() {
        let teacher = [0.5, 1.0, 1.5];
        let student = [0.1, 0.2, 0.3];
        if let CalibrationVerdict::Ok { temperature, .. } = calibrate(&teacher, &student) {
            assert!(temperature > 0.0);
        }
    }

    #[test]
    fn under_peaked_student_yields_below_one() {
        // Student less peaked → T < 1.
        let teacher = [0.1, 0.5, 1.0, 1.5, 2.0];
        let student = [0.05, 0.25, 0.5, 0.75, 1.0]; // half teacher
        if let CalibrationVerdict::Ok { temperature, .. } = calibrate(&teacher, &student) {
            assert!(temperature < 1.0);
        }
    }
}
