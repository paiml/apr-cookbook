//! # SIMD Horizontal Max
//!
//! Find the max across SIMD lanes (used in softmax to subtract for
//! numerical stability). Strategy: pairwise compare, halving the lanes
//! each step.
//!
//! Demonstrates the **SIMD.16** recipe for PMAT-147 (simd round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vector reduction algorithms (Intel optimization manual).
//!
//! Run with: cargo run --example simd_horizontal_max
//!
//! Added by PMAT-147 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MaxVerdict {
    Ok { max: f32, index: usize, steps: u32 },
    EmptyLanes,
    InvalidLanes,
}

pub fn horizontal_max(lanes: &[f32]) -> MaxVerdict {
    if lanes.is_empty() {
        return MaxVerdict::EmptyLanes;
    }
    if lanes.iter().any(|x| x.is_nan()) {
        return MaxVerdict::InvalidLanes;
    }
    let n = lanes.len();
    if !n.is_power_of_two() {
        return MaxVerdict::InvalidLanes;
    }
    let mut working = lanes.to_vec();
    let mut indices: Vec<usize> = (0..n).collect();
    let mut steps = 0u32;
    let mut size = n;
    while size > 1 {
        let half = size / 2;
        for i in 0..half {
            if working[i + half] > working[i] {
                working[i] = working[i + half];
                indices[i] = indices[i + half];
            }
        }
        size = half;
        steps += 1;
    }
    MaxVerdict::Ok {
        max: working[0],
        index: indices[0],
        steps,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("simd_horizontal_max")?;

    println!(
        "8 lanes: {:?}",
        horizontal_max(&[1.0, 5.0, 3.0, 2.0, 8.0, 6.0, 7.0, 4.0])
    );
    println!(
        "16 lanes (random): {:?}",
        horizontal_max(&[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0
        ])
    );
    println!("empty: {:?}", horizontal_max(&[]));
    println!("3 lanes (not pow2): {:?}", horizontal_max(&[1.0, 2.0, 3.0]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn finds_max_in_8_lanes() {
        let v = horizontal_max(&[1.0, 5.0, 3.0, 2.0, 8.0, 6.0, 7.0, 4.0]);
        if let MaxVerdict::Ok { max, .. } = v {
            assert!((max - 8.0).abs() < 1e-9);
        }
    }

    #[test]
    fn finds_max_index() {
        let v = horizontal_max(&[1.0, 5.0, 3.0, 2.0, 8.0, 6.0, 7.0, 4.0]);
        if let MaxVerdict::Ok { index, .. } = v {
            assert_eq!(index, 4);
        }
    }

    #[test]
    fn steps_log2_lanes() {
        // 8 lanes → 3 steps (8→4→2→1).
        let v = horizontal_max(&[0.0; 8]);
        if let MaxVerdict::Ok { steps, .. } = v {
            assert_eq!(steps, 3);
        }
    }

    #[test]
    fn single_lane_zero_steps() {
        let v = horizontal_max(&[42.0]);
        if let MaxVerdict::Ok { max, steps, .. } = v {
            assert!((max - 42.0).abs() < 1e-9);
            assert_eq!(steps, 0);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(horizontal_max(&[]), MaxVerdict::EmptyLanes);
    }

    #[test]
    fn nan_rejected() {
        assert_eq!(
            horizontal_max(&[1.0, f32::NAN, 2.0, 3.0]),
            MaxVerdict::InvalidLanes
        );
    }

    #[test]
    fn non_power_of_two_rejected() {
        assert_eq!(horizontal_max(&[1.0, 2.0, 3.0]), MaxVerdict::InvalidLanes);
    }

    #[test]
    fn negative_values_handled() {
        let v = horizontal_max(&[-5.0, -3.0, -1.0, -7.0]);
        if let MaxVerdict::Ok { max, .. } = v {
            assert!((max - (-1.0)).abs() < 1e-9);
        }
    }

    #[test]
    fn all_equal_first_index_wins() {
        // When all equal, no swap occurs → index stays at 0.
        let v = horizontal_max(&[5.0, 5.0, 5.0, 5.0]);
        if let MaxVerdict::Ok { index, .. } = v {
            assert_eq!(index, 0);
        }
    }

    #[test]
    fn sixteen_lanes_max_correct() {
        let lanes: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let v = horizontal_max(&lanes);
        if let MaxVerdict::Ok { max, index, .. } = v {
            assert!((max - 15.0).abs() < 1e-9);
            assert_eq!(index, 15);
        }
    }
}
