//! # SIMD Per-Lane Count Trailing Zeros
//!
//! Compute trailing-zero count per-lane for u32 lanes. Returns
//! per-lane CTZ values and total trailing zeros across all lanes.
//!
//! Demonstrates the **SIMD.X** recipe for PMAT-222 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AVX-512 VTPCNT (vector trailing population count); ARM
//!  NEON RBIT+CLZ trick; pattern in trueno bit-manipulation kernels.
//!
//! Run with: cargo run --example simd_count_trailing_zeros_lane
//!
//! Added by PMAT-222 (catalog 1621→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CtzVerdict {
    Ok {
        per_lane_ctz: Vec<u32>,
        total_ctz: u32,
    },
    InvalidConfig,
}

pub fn compute(lanes: &[u32]) -> CtzVerdict {
    if lanes.is_empty() {
        return CtzVerdict::InvalidConfig;
    }
    let per_lane: Vec<u32> = lanes.iter().map(|v| v.trailing_zeros()).collect();
    let total: u32 = per_lane.iter().sum();
    CtzVerdict::Ok {
        per_lane_ctz: per_lane,
        total_ctz: total,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("simd_count_trailing_zeros_lane")?;

    println!("powers: {:?}", compute(&[1, 2, 4, 8]));
    println!("invalid: {:?}", compute(&[]));
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
        assert_eq!(compute(&[]), CtzVerdict::InvalidConfig);
    }

    #[test]
    fn single_bit_lanes_correct_ctz() {
        let v = compute(&[1, 2, 4, 8]);
        if let CtzVerdict::Ok { per_lane_ctz, .. } = v {
            assert_eq!(per_lane_ctz, vec![0, 1, 2, 3]);
        }
    }

    #[test]
    fn zero_lane_max_ctz() {
        let v = compute(&[0]);
        if let CtzVerdict::Ok { per_lane_ctz, .. } = v {
            assert_eq!(per_lane_ctz, vec![32]);
        }
    }

    #[test]
    fn odd_value_ctz_zero() {
        let v = compute(&[3, 5, 7]);
        if let CtzVerdict::Ok { per_lane_ctz, .. } = v {
            assert_eq!(per_lane_ctz, vec![0, 0, 0]);
        }
    }

    #[test]
    fn high_bit_ctz_31() {
        let v = compute(&[1 << 31]);
        if let CtzVerdict::Ok { per_lane_ctz, .. } = v {
            assert_eq!(per_lane_ctz, vec![31]);
        }
    }

    #[test]
    fn total_ctz_correct() {
        let v = compute(&[1, 2, 4]);
        if let CtzVerdict::Ok { total_ctz, .. } = v {
            assert_eq!(total_ctz, 0 + 1 + 2);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = compute(&[1]);
        let r2 = compute(&[1]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn lane_count_matches_input() {
        let v = compute(&[1, 2, 3, 4, 5]);
        if let CtzVerdict::Ok { per_lane_ctz, .. } = v {
            assert_eq!(per_lane_ctz.len(), 5);
        }
    }

    #[test]
    fn many_lanes_handled() {
        let lanes: Vec<u32> = (1..=100).collect();
        let v = compute(&lanes);
        if let CtzVerdict::Ok { per_lane_ctz, .. } = v {
            assert_eq!(per_lane_ctz.len(), 100);
        }
    }

    #[test]
    fn high_value_handled() {
        let v = compute(&[u32::MAX]);
        if let CtzVerdict::Ok { per_lane_ctz, .. } = v {
            // u32::MAX = all 1s → ctz = 0
            assert_eq!(per_lane_ctz, vec![0]);
        }
    }

    #[test]
    fn alternating_bits_handled() {
        let v = compute(&[0xAAAA_AAAA, 0x5555_5555]);
        if let CtzVerdict::Ok { per_lane_ctz, .. } = v {
            // 0xAA = ...10101010 → ctz=1; 0x55 = ...01010101 → ctz=0.
            assert_eq!(per_lane_ctz, vec![1, 0]);
        }
    }

    #[test]
    fn power_of_two_ctz_log2() {
        let v = compute(&[1, 2, 4, 8, 16, 32, 64, 128]);
        if let CtzVerdict::Ok { per_lane_ctz, .. } = v {
            assert_eq!(per_lane_ctz, vec![0, 1, 2, 3, 4, 5, 6, 7]);
        }
    }
}
