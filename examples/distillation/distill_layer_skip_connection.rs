//! # Distillation Layer Skip-Connection Map
//!
//! Build mapping where every Nth teacher hidden state feeds the next
//! student layer (skip-connection). Useful when student is much
//! shallower but should still get teacher's deeper representations.
//!
//! Demonstrates the **DIST.41** recipe for PMAT-159 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: He et al. (2016) ResNet identity skip + KD adaptation.
//!
//! Run with: cargo run --example distill_layer_skip_connection
//!
//! Added by PMAT-159 (catalog 1054→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SkipMapVerdict {
    Ok { teacher_to_student: Vec<(u32, u32)> },
    InvalidLayerCounts,
    StudentDeeperThanTeacher,
}

pub fn build(teacher_layers: u32, student_layers: u32) -> SkipMapVerdict {
    if teacher_layers == 0 || student_layers == 0 {
        return SkipMapVerdict::InvalidLayerCounts;
    }
    if student_layers > teacher_layers {
        return SkipMapVerdict::StudentDeeperThanTeacher;
    }
    let stride = teacher_layers / student_layers;
    let mut map = Vec::with_capacity(student_layers as usize);
    for s in 0..student_layers {
        let t = (s * stride + stride / 2).min(teacher_layers - 1);
        map.push((t, s));
    }
    SkipMapVerdict::Ok {
        teacher_to_student: map,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_layer_skip_connection")?;

    println!("12→6: {:?}", build(12, 6));
    println!("12→4: {:?}", build(12, 4));
    println!("equal: {:?}", build(8, 8));
    println!("invalid: {:?}", build(4, 8));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn map_length_matches_student() {
        let v = build(12, 6);
        if let SkipMapVerdict::Ok { teacher_to_student } = v {
            assert_eq!(teacher_to_student.len(), 6);
        }
    }

    #[test]
    fn passthrough_when_equal() {
        let v = build(6, 6);
        if let SkipMapVerdict::Ok { teacher_to_student } = v {
            // stride=1, so each student maps to teacher i + 0 = i.
            for (t, s) in teacher_to_student {
                assert_eq!(t, s);
            }
        }
    }

    #[test]
    fn invalid_zero_teacher() {
        assert_eq!(build(0, 4), SkipMapVerdict::InvalidLayerCounts);
    }

    #[test]
    fn invalid_zero_student() {
        assert_eq!(build(12, 0), SkipMapVerdict::InvalidLayerCounts);
    }

    #[test]
    fn student_deeper_rejected() {
        assert_eq!(build(4, 8), SkipMapVerdict::StudentDeeperThanTeacher);
    }

    #[test]
    fn teacher_indices_in_range() {
        let v = build(12, 4);
        if let SkipMapVerdict::Ok { teacher_to_student } = v {
            for (t, _) in teacher_to_student {
                assert!(t < 12);
            }
        }
    }

    #[test]
    fn student_indices_increasing() {
        let v = build(12, 4);
        if let SkipMapVerdict::Ok { teacher_to_student } = v {
            for w in teacher_to_student.windows(2) {
                assert!(w[1].1 > w[0].1);
            }
        }
    }

    #[test]
    fn teacher_indices_strictly_increasing() {
        let v = build(20, 5);
        if let SkipMapVerdict::Ok { teacher_to_student } = v {
            for w in teacher_to_student.windows(2) {
                assert!(w[1].0 > w[0].0);
            }
        }
    }

    #[test]
    fn single_student_layer_picks_middle() {
        let v = build(10, 1);
        if let SkipMapVerdict::Ok { teacher_to_student } = v {
            // stride=10, t = 0*10 + 10/2 = 5.
            assert_eq!(teacher_to_student[0].0, 5);
        }
    }

    #[test]
    fn deterministic() {
        let a = build(12, 4);
        let b = build(12, 4);
        assert_eq!(a, b);
    }
}
