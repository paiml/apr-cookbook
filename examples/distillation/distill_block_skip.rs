//! # Distillation Teacher-Block Skip Map
//!
//! When student is N× shallower than teacher, pick which teacher
//! blocks to use as guidance. Strategy: even-stride sampling preserves
//! depth diversity better than copying first/last K blocks.
//!
//! Demonstrates the **DIST.36** recipe for PMAT-157 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Sanh et al. (2019) DistilBERT layer-skipping init.
//!
//! Run with: cargo run --example distill_block_skip
//!
//! Added by PMAT-157 (catalog 1036→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SkipVerdict {
    Ok { selected_blocks: Vec<u32> },
    StudentDeeperThanTeacher,
    InvalidBlockCounts,
}

pub fn pick(teacher_blocks: u32, student_blocks: u32) -> SkipVerdict {
    if teacher_blocks == 0 || student_blocks == 0 {
        return SkipVerdict::InvalidBlockCounts;
    }
    if student_blocks > teacher_blocks {
        return SkipVerdict::StudentDeeperThanTeacher;
    }
    if student_blocks == teacher_blocks {
        return SkipVerdict::Ok {
            selected_blocks: (0..teacher_blocks).collect(),
        };
    }
    let mut selected = Vec::with_capacity(student_blocks as usize);
    for i in 0..student_blocks {
        // Even stride: floor((i + 0.5) × T / S).
        let f = (f64::from(i) + 0.5) * f64::from(teacher_blocks) / f64::from(student_blocks);
        selected.push((f as u32).min(teacher_blocks - 1));
    }
    SkipVerdict::Ok {
        selected_blocks: selected,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_block_skip")?;

    println!("12→6: {:?}", pick(12, 6));
    println!("12→4: {:?}", pick(12, 4));
    println!("12→1: {:?}", pick(12, 1));
    println!("equal: {:?}", pick(12, 12));
    println!("invalid: {:?}", pick(4, 8));
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
    fn even_stride_for_2x_compress() {
        let v = pick(12, 6);
        if let SkipVerdict::Ok { selected_blocks } = v {
            assert_eq!(selected_blocks.len(), 6);
            // Strictly increasing.
            for w in selected_blocks.windows(2) {
                assert!(w[0] < w[1]);
            }
        }
    }

    #[test]
    fn first_block_near_zero() {
        let v = pick(12, 4);
        if let SkipVerdict::Ok { selected_blocks } = v {
            assert!(selected_blocks[0] <= 1);
        }
    }

    #[test]
    fn last_block_near_max() {
        let v = pick(12, 4);
        if let SkipVerdict::Ok { selected_blocks } = v {
            assert!(*selected_blocks.last().unwrap() >= 10);
        }
    }

    #[test]
    fn passthrough_when_equal() {
        let v = pick(6, 6);
        if let SkipVerdict::Ok { selected_blocks } = v {
            assert_eq!(selected_blocks, vec![0, 1, 2, 3, 4, 5]);
        }
    }

    #[test]
    fn single_student_block_is_middle() {
        let v = pick(12, 1);
        if let SkipVerdict::Ok { selected_blocks } = v {
            // Should pick block near middle.
            assert!(selected_blocks[0] == 5 || selected_blocks[0] == 6);
        }
    }

    #[test]
    fn invalid_zero_teacher() {
        assert_eq!(pick(0, 4), SkipVerdict::InvalidBlockCounts);
    }

    #[test]
    fn invalid_zero_student() {
        assert_eq!(pick(12, 0), SkipVerdict::InvalidBlockCounts);
    }

    #[test]
    fn student_deeper_rejected() {
        assert_eq!(pick(4, 8), SkipVerdict::StudentDeeperThanTeacher);
    }

    #[test]
    fn no_duplicates_in_output() {
        let v = pick(20, 5);
        if let SkipVerdict::Ok { selected_blocks } = v {
            let unique: std::collections::BTreeSet<u32> = selected_blocks.iter().copied().collect();
            assert_eq!(unique.len(), selected_blocks.len());
        }
    }

    #[test]
    fn all_blocks_in_range() {
        let v = pick(12, 6);
        if let SkipVerdict::Ok { selected_blocks } = v {
            assert!(selected_blocks.iter().all(|b| *b < 12));
        }
    }

    #[test]
    fn deterministic() {
        let a = pick(12, 4);
        let b = pick(12, 4);
        assert_eq!(a, b);
    }
}
