//! # Distillation Temperature Picker
//!
//! KD softmax temperature T flattens the teacher's distribution to
//! transmit dark knowledge. Heuristic per Hinton et al.:
//!
//! - Small student (≤25% teacher params) → T = 4..10
//! - Medium student (25-75%) → T = 2..4
//! - Near-equal student → T = 1..2
//!
//! Plus warm-start: increase T early in training, decay toward 1 as
//! student converges. This recipe builds the picker.
//!
//! Demonstrates the **DIST.10** recipe for PMAT-137 (distillation coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hinton et al. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531.
//!
//! Run with: cargo run --example distill_temperature_picker
//!
//! Added by PMAT-137 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TempVerdict {
    Ok { temperature: f64 },
    InvalidRatio,
    InvalidProgress,
}

pub fn pick(student_to_teacher_param_ratio: f64, training_progress: f64) -> TempVerdict {
    if !student_to_teacher_param_ratio.is_finite()
        || student_to_teacher_param_ratio <= 0.0
        || student_to_teacher_param_ratio > 2.0
    {
        return TempVerdict::InvalidRatio;
    }
    if !training_progress.is_finite() || !(0.0..=1.0).contains(&training_progress) {
        return TempVerdict::InvalidProgress;
    }
    let base_t = if student_to_teacher_param_ratio <= 0.25 {
        7.0
    } else if student_to_teacher_param_ratio <= 0.75 {
        3.0
    } else {
        1.5
    };
    // Decay toward 1.0 as training_progress → 1.0.
    let temperature = 1.0 + (base_t - 1.0) * (1.0 - training_progress);
    TempVerdict::Ok { temperature }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_temperature_picker")?;

    let cases = [
        (0.1_f64, 0.0_f64),
        (0.5, 0.0),
        (1.0, 0.0),
        (0.1, 0.5),
        (0.1, 1.0),
    ];
    for (r, p) in cases {
        println!("ratio={r} progress={p} → {:?}", pick(r, p));
    }
    println!("invalid ratio: {:?}", pick(3.0, 0.5));
    println!("invalid progress: {:?}", pick(0.5, 1.5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t_at(ratio: f64, progress: f64) -> f64 {
        if let TempVerdict::Ok { temperature } = pick(ratio, progress) {
            temperature
        } else {
            f64::NAN
        }
    }

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn small_student_starts_at_seven() {
        assert!((t_at(0.1, 0.0) - 7.0).abs() < 1e-9);
    }

    #[test]
    fn medium_student_starts_at_three() {
        assert!((t_at(0.5, 0.0) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn near_equal_student_starts_at_1_5() {
        assert!((t_at(1.0, 0.0) - 1.5).abs() < 1e-9);
    }

    #[test]
    fn end_of_training_t_is_one() {
        // At progress=1.0, T decays to 1.0 regardless of ratio.
        assert!((t_at(0.1, 1.0) - 1.0).abs() < 1e-9);
        assert!((t_at(1.0, 1.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn t_monotonically_decreasing_in_progress() {
        let t0 = t_at(0.1, 0.0);
        let t1 = t_at(0.1, 0.5);
        let t2 = t_at(0.1, 1.0);
        assert!(t0 > t1);
        assert!(t1 > t2);
    }

    #[test]
    fn invalid_ratio_zero_rejected() {
        assert_eq!(pick(0.0, 0.5), TempVerdict::InvalidRatio);
    }

    #[test]
    fn invalid_ratio_too_large_rejected() {
        assert_eq!(pick(3.0, 0.5), TempVerdict::InvalidRatio);
    }

    #[test]
    fn invalid_progress_below_zero_rejected() {
        assert_eq!(pick(0.5, -0.1), TempVerdict::InvalidProgress);
    }

    #[test]
    fn invalid_progress_above_one_rejected() {
        assert_eq!(pick(0.5, 1.5), TempVerdict::InvalidProgress);
    }

    #[test]
    fn nan_inputs_rejected() {
        assert_eq!(pick(f64::NAN, 0.5), TempVerdict::InvalidRatio);
        assert_eq!(pick(0.5, f64::NAN), TempVerdict::InvalidProgress);
    }

    #[test]
    fn t_always_at_least_one() {
        // For any valid input, T should be ≥ 1.0.
        for r in [0.1f64, 0.3, 0.5, 0.8, 1.0, 1.5] {
            for p in [0.0_f64, 0.25, 0.5, 0.75, 1.0] {
                let v = t_at(r, p);
                assert!(v >= 1.0, "T={v} for ({r},{p})");
            }
        }
    }
}
