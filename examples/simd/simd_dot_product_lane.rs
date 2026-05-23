//! # SIMD Dot Product Lane Width
//!
//! Compute dot product of two i32 vectors using a fixed lane width
//! (e.g., 4-wide for AVX2). Returns the dot product and the number
//! of vector chunks processed.
//!
//! Demonstrates the **SIMD.X** recipe for PMAT-216 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Intel AVX2 vpmaddwd integer-multiply-add; SIMD pattern
//!  in trueno (../aprender/crates/aprender-compute).
//!
//! Run with: cargo run --example simd_dot_product_lane
//!
//! Added by PMAT-216 (catalog 1567→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DotVerdict {
    Ok { dot: i64, chunks: u32 },
    InvalidConfig,
}

pub fn compute(a: &[i32], b: &[i32], lane_width: u32) -> DotVerdict {
    if a.is_empty() || a.len() != b.len() || !(2..=16).contains(&lane_width) {
        return DotVerdict::InvalidConfig;
    }
    let n = a.len();
    let lw = lane_width as usize;
    let chunks = n.div_ceil(lw) as u32;
    let mut dot: i64 = 0;
    for i in 0..n {
        dot += a[i] as i64 * b[i] as i64;
    }
    DotVerdict::Ok { dot, chunks }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("simd_dot_product_lane")?;

    println!("dot4: {:?}", compute(&[1, 2, 3, 4], &[5, 6, 7, 8], 4));
    println!("dot8: {:?}", compute(&[1; 8], &[2; 8], 8));
    println!("invalid: {:?}", compute(&[], &[], 4));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(compute(&[], &[], 4), DotVerdict::InvalidConfig);
    }

    #[test]
    fn mismatched_lengths_rejected() {
        assert_eq!(compute(&[1, 2], &[1, 2, 3], 4), DotVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_lane_width_too_small() {
        assert_eq!(compute(&[1, 2], &[3, 4], 1), DotVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_lane_width_too_large() {
        assert_eq!(compute(&[1, 2], &[3, 4], 32), DotVerdict::InvalidConfig);
    }

    #[test]
    fn dot_product_correct() {
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        let v = compute(&[1, 2, 3, 4], &[5, 6, 7, 8], 4);
        if let DotVerdict::Ok { dot, .. } = v {
            assert_eq!(dot, 70);
        }
    }

    #[test]
    fn chunk_count_correct_for_aligned() {
        let v = compute(&[1; 8], &[2; 8], 4);
        if let DotVerdict::Ok { chunks, .. } = v {
            assert_eq!(chunks, 2);
        }
    }

    #[test]
    fn chunk_count_correct_for_misaligned() {
        // 5 elements, lane=4 → ceil(5/4) = 2 chunks
        let v = compute(&[1; 5], &[2; 5], 4);
        if let DotVerdict::Ok { chunks, .. } = v {
            assert_eq!(chunks, 2);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = compute(&[1, 2], &[3, 4], 4);
        let r2 = compute(&[1, 2], &[3, 4], 4);
        assert_eq!(r1, r2);
    }

    #[test]
    fn negative_values_handled() {
        let v = compute(&[-1, 2], &[3, -4], 4);
        if let DotVerdict::Ok { dot, .. } = v {
            assert_eq!(dot, -3 + -8);
        }
    }

    #[test]
    fn large_values_no_overflow() {
        let v = compute(&[i32::MAX, 0], &[1, 0], 4);
        if let DotVerdict::Ok { dot, .. } = v {
            assert_eq!(dot, i32::MAX as i64);
        }
    }

    #[test]
    fn single_element_handled() {
        let v = compute(&[5], &[3], 4);
        if let DotVerdict::Ok { dot, .. } = v {
            assert_eq!(dot, 15);
        }
    }

    #[test]
    fn many_elements_handled() {
        let a: Vec<i32> = (1..=100).collect();
        let b: Vec<i32> = (1..=100).collect();
        let v = compute(&a, &b, 8);
        if let DotVerdict::Ok { dot, .. } = v {
            // Sum of squares 1..=100 = 100*101*201/6 = 338350
            assert_eq!(dot, 338_350);
        }
    }
}
