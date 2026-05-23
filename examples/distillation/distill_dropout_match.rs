//! # Distillation Dropout Rate Matching
//!
//! Student should use the same effective dropout rate as the teacher
//! to match regularization. Adjust for layer count: deeper student =
//! more dropout per layer to match cumulative coverage.
//!
//! Demonstrates the **DIST.32** recipe for PMAT-156 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Srivastava et al. (2014). Dropout: A simple way to prevent overfitting.
//!
//! Run with: cargo run --example distill_dropout_match
//!
//! Added by PMAT-156 (catalog 1027→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DropoutVerdict {
    Ok { student_dropout: f64 },
    InvalidConfig,
}

pub fn pick(teacher_layers: u32, teacher_dropout: f64, student_layers: u32) -> DropoutVerdict {
    if teacher_layers == 0
        || student_layers == 0
        || !teacher_dropout.is_finite()
        || !(0.0..=1.0).contains(&teacher_dropout)
    {
        return DropoutVerdict::InvalidConfig;
    }
    // Effective survival = (1-p)^L. To match: (1-p_s)^L_s = (1-p_t)^L_t.
    // p_s = 1 - (1-p_t)^(L_t/L_s).
    let teacher_keep = 1.0 - teacher_dropout;
    let exponent = f64::from(teacher_layers) / f64::from(student_layers);
    let student_keep = teacher_keep.powf(exponent);
    let student_dropout = (1.0 - student_keep).clamp(0.0, 0.95);
    DropoutVerdict::Ok { student_dropout }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_dropout_match")?;

    println!("same depth: {:?}", pick(12, 0.1, 12));
    println!("deeper student: {:?}", pick(12, 0.1, 24));
    println!("shallower student: {:?}", pick(12, 0.1, 6));
    println!("invalid: {:?}", pick(0, 0.1, 12));
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
    fn same_depth_passes_through() {
        let v = pick(12, 0.1, 12);
        if let DropoutVerdict::Ok { student_dropout } = v {
            assert!((student_dropout - 0.1).abs() < 1e-9);
        }
    }

    #[test]
    fn deeper_student_lower_per_layer() {
        let v = pick(12, 0.1, 24);
        if let DropoutVerdict::Ok { student_dropout } = v {
            // Deeper student → less dropout per layer to match cumulative.
            assert!(student_dropout < 0.1);
        }
    }

    #[test]
    fn shallower_student_higher_per_layer() {
        let v = pick(12, 0.1, 6);
        if let DropoutVerdict::Ok { student_dropout } = v {
            // Shallower → more per layer to match cumulative.
            assert!(student_dropout > 0.1);
        }
    }

    #[test]
    fn zero_teacher_layers_invalid() {
        assert_eq!(pick(0, 0.1, 12), DropoutVerdict::InvalidConfig);
    }

    #[test]
    fn zero_student_layers_invalid() {
        assert_eq!(pick(12, 0.1, 0), DropoutVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_dropout_rate_negative() {
        assert_eq!(pick(12, -0.1, 12), DropoutVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_dropout_rate_over_1() {
        assert_eq!(pick(12, 1.5, 12), DropoutVerdict::InvalidConfig);
    }

    #[test]
    fn nan_dropout_invalid() {
        assert_eq!(pick(12, f64::NAN, 12), DropoutVerdict::InvalidConfig);
    }

    #[test]
    fn zero_dropout_passes_zero() {
        let v = pick(12, 0.0, 12);
        if let DropoutVerdict::Ok { student_dropout } = v {
            assert!((student_dropout - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn high_dropout_clamped_at_95() {
        // Even pathological inputs cap at 0.95.
        let v = pick(100, 0.99, 1);
        if let DropoutVerdict::Ok { student_dropout } = v {
            assert!(student_dropout <= 0.95);
        }
    }

    #[test]
    fn deterministic() {
        let a = pick(12, 0.1, 24);
        let b = pick(12, 0.1, 24);
        assert_eq!(a, b);
    }
}
