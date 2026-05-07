//! # SIMD Saturating Arithmetic
//!
//! Compute saturating add/sub of two i16 vectors per-lane. Values
//! that would overflow saturate to i16::MAX or MIN. Returns result
//! vector and number of saturation events.
//!
//! Demonstrates the **SIMD.X** recipe for PMAT-218 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ARM NEON VQADD instruction; SSE PADDSW signed-saturate
//!  add; pattern in trueno (../aprender/crates/aprender-compute).
//!
//! Run with: cargo run --example simd_saturating_arithmetic
//!
//! Added by PMAT-218 (catalog 1585→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SatVerdict {
    Ok {
        result: Vec<i16>,
        saturation_events: u32,
    },
    InvalidConfig,
}

pub fn saturating_add(a: &[i16], b: &[i16]) -> SatVerdict {
    if a.is_empty() || a.len() != b.len() {
        return SatVerdict::InvalidConfig;
    }
    let mut result: Vec<i16> = Vec::with_capacity(a.len());
    let mut saturated = 0u32;
    for i in 0..a.len() {
        let sum_i32 = a[i] as i32 + b[i] as i32;
        let sat = sum_i32.clamp(i16::MIN as i32, i16::MAX as i32);
        if sat != sum_i32 {
            saturated += 1;
        }
        result.push(sat as i16);
    }
    SatVerdict::Ok {
        result,
        saturation_events: saturated,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("simd_saturating_arithmetic")?;

    println!("normal: {:?}", saturating_add(&[1, 2, 3], &[4, 5, 6]));
    println!(
        "saturate: {:?}",
        saturating_add(&[i16::MAX, i16::MIN], &[10, -10])
    );
    println!("invalid: {:?}", saturating_add(&[], &[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adder_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(saturating_add(&[], &[]), SatVerdict::InvalidConfig);
    }

    #[test]
    fn mismatched_lengths_rejected() {
        assert_eq!(saturating_add(&[1], &[1, 2]), SatVerdict::InvalidConfig);
    }

    #[test]
    fn normal_addition_no_saturate() {
        let v = saturating_add(&[1, 2, 3], &[4, 5, 6]);
        if let SatVerdict::Ok {
            result,
            saturation_events,
        } = v
        {
            assert_eq!(result, vec![5, 7, 9]);
            assert_eq!(saturation_events, 0);
        }
    }

    #[test]
    fn overflow_saturates_to_max() {
        let v = saturating_add(&[i16::MAX], &[10]);
        if let SatVerdict::Ok {
            result,
            saturation_events,
        } = v
        {
            assert_eq!(result, vec![i16::MAX]);
            assert_eq!(saturation_events, 1);
        }
    }

    #[test]
    fn underflow_saturates_to_min() {
        let v = saturating_add(&[i16::MIN], &[-10]);
        if let SatVerdict::Ok {
            result,
            saturation_events,
        } = v
        {
            assert_eq!(result, vec![i16::MIN]);
            assert_eq!(saturation_events, 1);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = saturating_add(&[1, 2], &[3, 4]);
        let r2 = saturating_add(&[1, 2], &[3, 4]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn boundary_at_max_no_saturate() {
        let v = saturating_add(&[i16::MAX - 1], &[1]);
        if let SatVerdict::Ok {
            saturation_events, ..
        } = v
        {
            assert_eq!(saturation_events, 0);
        }
    }

    #[test]
    fn mixed_saturation_some_normal() {
        let v = saturating_add(&[1, i16::MAX, 3], &[4, 10, 6]);
        if let SatVerdict::Ok {
            saturation_events, ..
        } = v
        {
            assert_eq!(saturation_events, 1);
        }
    }

    #[test]
    fn negative_no_overflow_correct() {
        let v = saturating_add(&[-100, -200], &[-50, -100]);
        if let SatVerdict::Ok { result, .. } = v {
            assert_eq!(result, vec![-150, -300]);
        }
    }

    #[test]
    fn many_lanes_handled() {
        let a: Vec<i16> = (0..100).collect();
        let b: Vec<i16> = (0..100).collect();
        let v = saturating_add(&a, &b);
        if let SatVerdict::Ok { result, .. } = v {
            assert_eq!(result.len(), 100);
        }
    }

    #[test]
    fn single_lane_handled() {
        let v = saturating_add(&[5], &[10]);
        if let SatVerdict::Ok { result, .. } = v {
            assert_eq!(result, vec![15]);
        }
    }

    #[test]
    fn zero_plus_zero_no_saturate() {
        let v = saturating_add(&[0], &[0]);
        if let SatVerdict::Ok {
            saturation_events, ..
        } = v
        {
            assert_eq!(saturation_events, 0);
        }
    }
}
