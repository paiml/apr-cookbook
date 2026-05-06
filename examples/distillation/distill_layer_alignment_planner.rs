//! # Distillation Layer Alignment Planner
//!
//! Layer-wise distillation aligns student layer i to a teacher layer
//! T(i). When teacher has more layers than student, evenly subsample:
//! T(i) = round(i × (teacher_layers - 1) / (student_layers - 1)).
//! Reverse not allowed (student deeper than teacher → reject). This
//! recipe builds the planner.
//!
//! Demonstrates the **DIST.12** recipe for PMAT-137 (distillation coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Sun et al. (2019). Patient Knowledge Distillation. arXiv:1908.09355.
//!
//! Run with: cargo run --example distill_layer_alignment_planner
//!
//! Added by PMAT-137 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AlignmentVerdict {
    Ok { mapping: Vec<usize> },
    StudentDeeperThanTeacher { student: u32, teacher: u32 },
    InvalidLayers,
}

pub fn plan(student_layers: u32, teacher_layers: u32) -> AlignmentVerdict {
    if student_layers == 0 || teacher_layers == 0 {
        return AlignmentVerdict::InvalidLayers;
    }
    if student_layers > teacher_layers {
        return AlignmentVerdict::StudentDeeperThanTeacher {
            student: student_layers,
            teacher: teacher_layers,
        };
    }
    if student_layers == 1 {
        // Single student layer → align to last teacher layer.
        return AlignmentVerdict::Ok {
            mapping: vec![(teacher_layers - 1) as usize],
        };
    }
    let mapping: Vec<usize> = (0..student_layers)
        .map(|i| {
            (f64::from(i) * f64::from(teacher_layers - 1) / f64::from(student_layers - 1)).round()
                as usize
        })
        .collect();
    AlignmentVerdict::Ok { mapping }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_layer_alignment_planner")?;

    println!("4-into-12: {:?}", plan(4, 12));
    println!("6-into-24: {:?}", plan(6, 24));
    println!("12-into-12: {:?}", plan(12, 12));
    println!("1-into-12: {:?}", plan(1, 12));
    println!("16-into-12 (reject): {:?}", plan(16, 12));
    println!("invalid: {:?}", plan(0, 12));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn equal_layers_identity_mapping() {
        let v = plan(4, 4);
        if let AlignmentVerdict::Ok { mapping } = v {
            assert_eq!(mapping, vec![0, 1, 2, 3]);
        }
    }

    #[test]
    fn typical_4_into_12() {
        let v = plan(4, 12);
        if let AlignmentVerdict::Ok { mapping } = v {
            assert_eq!(mapping.len(), 4);
            assert_eq!(mapping[0], 0);
            assert_eq!(mapping[3], 11);
        }
    }

    #[test]
    fn first_layer_aligns_to_zero() {
        let v = plan(6, 24);
        if let AlignmentVerdict::Ok { mapping } = v {
            assert_eq!(mapping[0], 0);
        }
    }

    #[test]
    fn last_layer_aligns_to_last() {
        let v = plan(6, 24);
        if let AlignmentVerdict::Ok { mapping } = v {
            assert_eq!(*mapping.last().unwrap(), 23);
        }
    }

    #[test]
    fn mapping_monotone_non_decreasing() {
        let v = plan(6, 24);
        if let AlignmentVerdict::Ok { mapping } = v {
            for w in mapping.windows(2) {
                assert!(w[1] >= w[0]);
            }
        }
    }

    #[test]
    fn single_student_layer_goes_to_last() {
        let v = plan(1, 12);
        if let AlignmentVerdict::Ok { mapping } = v {
            assert_eq!(mapping, vec![11]);
        }
    }

    #[test]
    fn student_deeper_rejected() {
        let v = plan(16, 12);
        assert!(matches!(
            v,
            AlignmentVerdict::StudentDeeperThanTeacher {
                student: 16,
                teacher: 12
            }
        ));
    }

    #[test]
    fn zero_layers_invalid() {
        assert_eq!(plan(0, 12), AlignmentVerdict::InvalidLayers);
        assert_eq!(plan(4, 0), AlignmentVerdict::InvalidLayers);
    }

    #[test]
    fn evenly_spaced_for_3_into_9() {
        // 3 layers from 9: 0, 4, 8.
        let v = plan(3, 9);
        if let AlignmentVerdict::Ok { mapping } = v {
            assert_eq!(mapping, vec![0, 4, 8]);
        }
    }

    #[test]
    fn three_into_24_correct() {
        // student 3, teacher 24 → 0, 12 (round(11.5)), 23.
        let v = plan(3, 24);
        if let AlignmentVerdict::Ok { mapping } = v {
            assert_eq!(mapping[0], 0);
            assert_eq!(*mapping.last().unwrap(), 23);
        }
    }
}
