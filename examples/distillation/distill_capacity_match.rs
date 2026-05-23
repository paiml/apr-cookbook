//! # Distillation Student Capacity Picker
//!
//! Pick student parameter count given target accuracy gap from teacher:
//!   ≤1% gap → 50% of teacher
//!   ≤3% gap → 25% of teacher
//!   ≤5% gap → 12% of teacher
//!   ≤10% gap → 5% of teacher
//!
//! Demonstrates the **DIST.28** recipe for PMAT-154 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: DistilBERT/TinyBERT capacity-vs-accuracy curves.
//!
//! Run with: cargo run --example distill_capacity_match
//!
//! Added by PMAT-154 (catalog 1009→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CapacityVerdict {
    Ok {
        student_params: u64,
        compression_ratio: f64,
    },
    InvalidTeacherSize,
    InvalidAccuracyGap,
    UnreachableGap {
        min_achievable_gap_pct: f64,
    },
}

pub fn pick(teacher_params: u64, target_accuracy_gap_pct: f64) -> CapacityVerdict {
    if teacher_params == 0 {
        return CapacityVerdict::InvalidTeacherSize;
    }
    if !target_accuracy_gap_pct.is_finite() || target_accuracy_gap_pct < 0.0 {
        return CapacityVerdict::InvalidAccuracyGap;
    }
    if target_accuracy_gap_pct < 0.5 {
        return CapacityVerdict::UnreachableGap {
            min_achievable_gap_pct: 0.5,
        };
    }
    let fraction = if target_accuracy_gap_pct <= 1.0 {
        0.50
    } else if target_accuracy_gap_pct <= 3.0 {
        0.25
    } else if target_accuracy_gap_pct <= 5.0 {
        0.12
    } else if target_accuracy_gap_pct <= 10.0 {
        0.05
    } else {
        0.02
    };
    let student_params = (teacher_params as f64 * fraction) as u64;
    let compression_ratio = teacher_params as f64 / student_params as f64;
    CapacityVerdict::Ok {
        student_params,
        compression_ratio,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_capacity_match")?;

    let teacher = 7_000_000_000u64;
    println!("≤1% gap: {:?}", pick(teacher, 1.0));
    println!("≤3% gap: {:?}", pick(teacher, 3.0));
    println!("≤5% gap: {:?}", pick(teacher, 5.0));
    println!("≤10% gap: {:?}", pick(teacher, 10.0));
    println!("≤20% gap: {:?}", pick(teacher, 20.0));
    println!("invalid: {:?}", pick(0, 5.0));
    println!("unreachable: {:?}", pick(teacher, 0.1));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn small_gap_large_student() {
        let v = pick(7_000_000_000, 1.0);
        if let CapacityVerdict::Ok { student_params, .. } = v {
            // 50% of 7B = 3.5B.
            assert!(student_params > 3_000_000_000);
        }
    }

    #[test]
    fn large_gap_tiny_student() {
        let v = pick(7_000_000_000, 15.0);
        if let CapacityVerdict::Ok { student_params, .. } = v {
            // 2% of 7B = 140M.
            assert!(student_params < 200_000_000);
        }
    }

    #[test]
    fn invalid_zero_teacher() {
        assert_eq!(pick(0, 5.0), CapacityVerdict::InvalidTeacherSize);
    }

    #[test]
    fn invalid_negative_gap() {
        assert_eq!(pick(1_000_000, -1.0), CapacityVerdict::InvalidAccuracyGap);
    }

    #[test]
    fn unreachable_gap_below_min() {
        let v = pick(1_000_000, 0.1);
        assert!(matches!(v, CapacityVerdict::UnreachableGap { .. }));
    }

    #[test]
    fn nan_gap_invalid() {
        assert_eq!(
            pick(1_000_000, f64::NAN),
            CapacityVerdict::InvalidAccuracyGap
        );
    }

    #[test]
    fn larger_gap_smaller_student() {
        let v_small = pick(7_000_000_000, 1.0);
        let v_large = pick(7_000_000_000, 10.0);
        if let (
            CapacityVerdict::Ok {
                student_params: s, ..
            },
            CapacityVerdict::Ok {
                student_params: l, ..
            },
        ) = (v_small, v_large)
        {
            assert!(l < s);
        }
    }

    #[test]
    fn compression_ratio_increases_with_gap() {
        let v_small = pick(7_000_000_000, 1.0);
        let v_large = pick(7_000_000_000, 10.0);
        if let (
            CapacityVerdict::Ok {
                compression_ratio: cs,
                ..
            },
            CapacityVerdict::Ok {
                compression_ratio: cl,
                ..
            },
        ) = (v_small, v_large)
        {
            assert!(cl > cs);
        }
    }

    #[test]
    fn boundary_at_5_pct_uses_12_fraction() {
        let v = pick(1_000_000_000, 5.0);
        if let CapacityVerdict::Ok { student_params, .. } = v {
            // 12% of 1B = 120M.
            assert!(student_params >= 100_000_000);
            assert!(student_params <= 130_000_000);
        }
    }

    #[test]
    fn boundary_at_10_pct_uses_5_fraction() {
        let v = pick(1_000_000_000, 10.0);
        if let CapacityVerdict::Ok { student_params, .. } = v {
            // 5% of 1B = 50M.
            assert!(student_params >= 40_000_000);
            assert!(student_params <= 60_000_000);
        }
    }
}
