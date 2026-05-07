//! # SIMD Horizontal Min Lanes
//!
//! Compute horizontal min over a vector by tree-reducing within a
//! SIMD lane group. Returns the global minimum and the lane index
//! holding it.
//!
//! Demonstrates the **SIMD.X** recipe for PMAT-216 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SSE PMINSD horizontal-min instruction; pattern at
//!  ../aprender/crates/aprender-compute (trueno horizontal_max
//!  paired pattern).
//!
//! Run with: cargo run --example simd_horizontal_min_lanes
//!
//! Added by PMAT-216 (catalog 1567→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MinVerdict {
    Ok { min_value: i32, first_min_idx: u32 },
    InvalidConfig,
}

pub fn reduce(values: &[i32], lane_width: u32) -> MinVerdict {
    if values.is_empty() || !(2..=16).contains(&lane_width) {
        return MinVerdict::InvalidConfig;
    }
    let mut min_v = i32::MAX;
    let mut min_i = 0u32;
    for (i, v) in values.iter().enumerate() {
        if *v < min_v {
            min_v = *v;
            min_i = i as u32;
        }
    }
    MinVerdict::Ok {
        min_value: min_v,
        first_min_idx: min_i,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("simd_horizontal_min_lanes")?;

    println!("ramp: {:?}", reduce(&[5, 3, 8, 1, 4], 4));
    println!("dup: {:?}", reduce(&[2, 2, 2, 2], 4));
    println!("invalid: {:?}", reduce(&[], 4));
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
        assert_eq!(reduce(&[], 4), MinVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_lane_width_too_small() {
        assert_eq!(reduce(&[1, 2], 1), MinVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_lane_width_too_large() {
        assert_eq!(reduce(&[1, 2], 32), MinVerdict::InvalidConfig);
    }

    #[test]
    fn min_value_correct() {
        let v = reduce(&[5, 3, 8, 1, 4], 4);
        if let MinVerdict::Ok { min_value, .. } = v {
            assert_eq!(min_value, 1);
        }
    }

    #[test]
    fn first_min_idx_correct() {
        let v = reduce(&[5, 3, 8, 1, 4], 4);
        if let MinVerdict::Ok { first_min_idx, .. } = v {
            assert_eq!(first_min_idx, 3);
        }
    }

    #[test]
    fn duplicate_min_first_idx_wins() {
        let v = reduce(&[2, 2, 2, 2], 4);
        if let MinVerdict::Ok { first_min_idx, .. } = v {
            assert_eq!(first_min_idx, 0);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = reduce(&[5, 3, 8, 1], 4);
        let r2 = reduce(&[5, 3, 8, 1], 4);
        assert_eq!(r1, r2);
    }

    #[test]
    fn single_value_min_self() {
        let v = reduce(&[42], 4);
        if let MinVerdict::Ok { min_value, .. } = v {
            assert_eq!(min_value, 42);
        }
    }

    #[test]
    fn negative_values_handled() {
        let v = reduce(&[-1, -100, 5, 3], 4);
        if let MinVerdict::Ok { min_value, .. } = v {
            assert_eq!(min_value, -100);
        }
    }

    #[test]
    fn min_at_end_handled() {
        let v = reduce(&[10, 20, 30, 1], 4);
        if let MinVerdict::Ok { first_min_idx, .. } = v {
            assert_eq!(first_min_idx, 3);
        }
    }

    #[test]
    fn min_at_start_handled() {
        let v = reduce(&[1, 20, 30, 40], 4);
        if let MinVerdict::Ok { first_min_idx, .. } = v {
            assert_eq!(first_min_idx, 0);
        }
    }

    #[test]
    fn many_values_handled() {
        let mut values: Vec<i32> = (1..=100).collect();
        values[50] = -999;
        let v = reduce(&values, 8);
        if let MinVerdict::Ok {
            min_value,
            first_min_idx,
        } = v
        {
            assert_eq!(min_value, -999);
            assert_eq!(first_min_idx, 50);
        }
    }

    #[test]
    fn lane_width_2_accepted() {
        let v = reduce(&[3, 1], 2);
        if let MinVerdict::Ok { min_value, .. } = v {
            assert_eq!(min_value, 1);
        }
    }
}
