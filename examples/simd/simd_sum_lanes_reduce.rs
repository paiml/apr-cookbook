//! # SIMD Sum-Lanes Reduce
//!
//! Compute horizontal sum across SIMD lanes (i32). Returns total sum
//! and absolute mean.
//!
//! Demonstrates the **SIMD.X** recipe for PMAT-224 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AVX2 _mm256_reduce_add_epi32; ARM NEON VADDV.S32;
//!  trueno horizontal-reduce kernel pattern.
//!
//! Run with: cargo run --example simd_sum_lanes_reduce
//!
//! Added by PMAT-224 (catalog 1639→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SumVerdict {
    Ok { sum: i64, abs_mean: u32 },
    InvalidConfig,
}

pub fn reduce(lanes: &[i32]) -> SumVerdict {
    if lanes.is_empty() {
        return SumVerdict::InvalidConfig;
    }
    let sum: i64 = lanes.iter().map(|v| *v as i64).sum();
    let abs_mean = (sum.unsigned_abs() / lanes.len() as u64) as u32;
    SumVerdict::Ok { sum, abs_mean }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("simd_sum_lanes_reduce")?;

    println!("simple: {:?}", reduce(&[1, 2, 3, 4]));
    println!("mixed: {:?}", reduce(&[-5, 10, -15, 20]));
    println!("invalid: {:?}", reduce(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reducer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(reduce(&[]), SumVerdict::InvalidConfig);
    }

    #[test]
    fn sum_correct() {
        let v = reduce(&[1, 2, 3, 4]);
        if let SumVerdict::Ok { sum, .. } = v {
            assert_eq!(sum, 10);
        }
    }

    #[test]
    fn abs_mean_correct() {
        let v = reduce(&[2, 4, 6, 8]);
        if let SumVerdict::Ok { abs_mean, .. } = v {
            assert_eq!(abs_mean, 5);
        }
    }

    #[test]
    fn negative_values_handled() {
        let v = reduce(&[-1, -2, -3]);
        if let SumVerdict::Ok { sum, .. } = v {
            assert_eq!(sum, -6);
        }
    }

    #[test]
    fn mixed_signs_correct() {
        let v = reduce(&[-5, 10, -15, 20]);
        if let SumVerdict::Ok { sum, .. } = v {
            assert_eq!(sum, 10);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = reduce(&[1, 2, 3]);
        let r2 = reduce(&[1, 2, 3]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn single_lane_handled() {
        let v = reduce(&[42]);
        if let SumVerdict::Ok { sum, abs_mean } = v {
            assert_eq!(sum, 42);
            assert_eq!(abs_mean, 42);
        }
    }

    #[test]
    fn zero_lane_zero_sum() {
        let v = reduce(&[0]);
        if let SumVerdict::Ok { sum, .. } = v {
            assert_eq!(sum, 0);
        }
    }

    #[test]
    fn large_value_no_overflow() {
        let v = reduce(&[i32::MAX, i32::MAX]);
        if let SumVerdict::Ok { sum, .. } = v {
            assert_eq!(sum, i32::MAX as i64 * 2);
        }
    }

    #[test]
    fn abs_mean_uses_unsigned_abs() {
        let v = reduce(&[-100, -200]);
        if let SumVerdict::Ok { abs_mean, .. } = v {
            assert_eq!(abs_mean, 150);
        }
    }

    #[test]
    fn many_lanes_handled() {
        let lanes: Vec<i32> = (1..=100).collect();
        let v = reduce(&lanes);
        if let SumVerdict::Ok { sum, .. } = v {
            // Sum 1..=100 = 5050.
            assert_eq!(sum, 5050);
        }
    }

    #[test]
    fn balanced_signs_zero_sum() {
        let v = reduce(&[5, -5, 10, -10]);
        if let SumVerdict::Ok { sum, abs_mean } = v {
            assert_eq!(sum, 0);
            assert_eq!(abs_mean, 0);
        }
    }
}
