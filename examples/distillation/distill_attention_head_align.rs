//! # Distillation Attention-Head Alignment Map
//!
//! When student has fewer attention heads than teacher, we need a map:
//! which student head learns from which teacher head(s)? Strategy:
//! group teacher heads (consecutive blocks) → average to one student head.
//!
//! Demonstrates the **DIST.33** recipe for PMAT-156 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Sun et al. (2020). MobileBERT — head pruning during distillation.
//!
//! Run with: cargo run --example distill_attention_head_align
//!
//! Added by PMAT-156 (catalog 1027→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AlignVerdict {
    Ok { mapping: Vec<Vec<u32>> },
    InvalidHeadCounts,
    StudentMoreHeadsThanTeacher,
}

pub fn align(teacher_heads: u32, student_heads: u32) -> AlignVerdict {
    if teacher_heads == 0 || student_heads == 0 {
        return AlignVerdict::InvalidHeadCounts;
    }
    if student_heads > teacher_heads {
        return AlignVerdict::StudentMoreHeadsThanTeacher;
    }
    let group_size = teacher_heads / student_heads;
    let remainder = teacher_heads % student_heads;
    let mut mapping: Vec<Vec<u32>> = Vec::with_capacity(student_heads as usize);
    let mut t = 0u32;
    for s in 0..student_heads {
        // Distribute remainder across first `remainder` student heads.
        let extra = u32::from(s < remainder);
        let block: Vec<u32> = (0..group_size + extra).map(|i| t + i).collect();
        t += group_size + extra;
        mapping.push(block);
    }
    AlignVerdict::Ok { mapping }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("distill_attention_head_align")?;

    println!("12→6: {:?}", align(12, 6));
    println!("12→4 even: {:?}", align(12, 4));
    println!("12→5 uneven: {:?}", align(12, 5));
    println!("invalid: {:?}", align(4, 8));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn even_split() {
        let v = align(12, 6);
        if let AlignVerdict::Ok { mapping } = v {
            assert_eq!(mapping.len(), 6);
            assert!(mapping.iter().all(|b| b.len() == 2));
        }
    }

    #[test]
    fn one_to_one_when_equal() {
        let v = align(12, 12);
        if let AlignVerdict::Ok { mapping } = v {
            assert_eq!(mapping.len(), 12);
            assert!(mapping.iter().all(|b| b.len() == 1));
        }
    }

    #[test]
    fn uneven_distributes_remainder() {
        let v = align(12, 5);
        if let AlignVerdict::Ok { mapping } = v {
            // 12/5 = 2 r 2 → first 2 groups have 3, last 3 have 2.
            assert_eq!(mapping[0].len(), 3);
            assert_eq!(mapping[1].len(), 3);
            assert_eq!(mapping[2].len(), 2);
            let total: usize = mapping.iter().map(Vec::len).sum();
            assert_eq!(total, 12);
        }
    }

    #[test]
    fn no_overlap_in_mapping() {
        let v = align(12, 4);
        if let AlignVerdict::Ok { mapping } = v {
            let all: Vec<u32> = mapping.iter().flatten().copied().collect();
            let unique: std::collections::BTreeSet<u32> = all.iter().copied().collect();
            assert_eq!(all.len(), unique.len());
        }
    }

    #[test]
    fn covers_all_teacher_heads() {
        let v = align(12, 4);
        if let AlignVerdict::Ok { mapping } = v {
            let all: std::collections::BTreeSet<u32> = mapping.iter().flatten().copied().collect();
            for i in 0..12 {
                assert!(all.contains(&i));
            }
        }
    }

    #[test]
    fn zero_teacher_invalid() {
        assert_eq!(align(0, 4), AlignVerdict::InvalidHeadCounts);
    }

    #[test]
    fn zero_student_invalid() {
        assert_eq!(align(12, 0), AlignVerdict::InvalidHeadCounts);
    }

    #[test]
    fn student_too_many_rejected() {
        assert_eq!(align(4, 8), AlignVerdict::StudentMoreHeadsThanTeacher);
    }

    #[test]
    fn mapping_is_consecutive() {
        let v = align(12, 4);
        if let AlignVerdict::Ok { mapping } = v {
            // Each block should be consecutive integers.
            for block in &mapping {
                for w in block.windows(2) {
                    assert_eq!(w[1], w[0] + 1);
                }
            }
        }
    }

    #[test]
    fn deterministic() {
        let a = align(12, 4);
        let b = align(12, 4);
        assert_eq!(a, b);
    }
}
