//! # SIMD Per-Lane Population Count
//!
//! Compute population count (number of set bits) per-lane for u32
//! lanes. Returns popcount for each lane and the total bits set.
//!
//! Demonstrates the **SIMD.X** recipe for PMAT-220 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AVX-512 VPOPCNTDQ; ARM NEON CNT; pattern in trueno
//!  (../aprender/crates/aprender-compute) bit-manipulation kernels.
//!
//! Run with: cargo run --example simd_pop_count_lane
//!
//! Added by PMAT-220 (catalog 1603→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PopCountVerdict {
    Ok { per_lane: Vec<u32>, total_bits: u32 },
    InvalidConfig,
}

pub fn compute(lanes: &[u32]) -> PopCountVerdict {
    if lanes.is_empty() {
        return PopCountVerdict::InvalidConfig;
    }
    let per_lane: Vec<u32> = lanes.iter().map(|v| v.count_ones()).collect();
    let total: u32 = per_lane.iter().sum();
    PopCountVerdict::Ok {
        per_lane,
        total_bits: total,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("simd_pop_count_lane")?;

    println!("powers: {:?}", compute(&[0b0001, 0b0011, 0b0111, 0b1111]));
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
        assert_eq!(compute(&[]), PopCountVerdict::InvalidConfig);
    }

    #[test]
    fn single_bit_per_lane() {
        let v = compute(&[0b0001, 0b0010, 0b0100]);
        if let PopCountVerdict::Ok { per_lane, .. } = v {
            assert_eq!(per_lane, vec![1, 1, 1]);
        }
    }

    #[test]
    fn varying_bits_per_lane() {
        let v = compute(&[0b0001, 0b0011, 0b0111]);
        if let PopCountVerdict::Ok { per_lane, .. } = v {
            assert_eq!(per_lane, vec![1, 2, 3]);
        }
    }

    #[test]
    fn zero_lane_zero_count() {
        let v = compute(&[0]);
        if let PopCountVerdict::Ok { per_lane, .. } = v {
            assert_eq!(per_lane, vec![0]);
        }
    }

    #[test]
    fn all_ones_max_count() {
        let v = compute(&[u32::MAX]);
        if let PopCountVerdict::Ok { per_lane, .. } = v {
            assert_eq!(per_lane, vec![32]);
        }
    }

    #[test]
    fn total_bits_correct() {
        let v = compute(&[0b0001, 0b0011, 0b0111]);
        if let PopCountVerdict::Ok { total_bits, .. } = v {
            assert_eq!(total_bits, 6);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = compute(&[0b0001]);
        let r2 = compute(&[0b0001]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn lane_count_matches_input() {
        let v = compute(&[1, 2, 3, 4, 5]);
        if let PopCountVerdict::Ok { per_lane, .. } = v {
            assert_eq!(per_lane.len(), 5);
        }
    }

    #[test]
    fn many_lanes_handled() {
        let lanes: Vec<u32> = (0..100).collect();
        let v = compute(&lanes);
        if let PopCountVerdict::Ok { per_lane, .. } = v {
            assert_eq!(per_lane.len(), 100);
        }
    }

    #[test]
    fn high_value_handled() {
        let v = compute(&[0xFFFF_FFFF, 0]);
        if let PopCountVerdict::Ok {
            per_lane,
            total_bits,
        } = v
        {
            assert_eq!(per_lane, vec![32, 0]);
            assert_eq!(total_bits, 32);
        }
    }

    #[test]
    fn alternating_bits_handled() {
        let v = compute(&[0xAAAA_AAAA, 0x5555_5555]);
        if let PopCountVerdict::Ok { per_lane, .. } = v {
            assert_eq!(per_lane, vec![16, 16]);
        }
    }

    #[test]
    fn total_le_lanes_times_32() {
        let lanes = vec![u32::MAX; 5];
        let v = compute(&lanes);
        if let PopCountVerdict::Ok { total_bits, .. } = v {
            assert_eq!(total_bits, 32 * 5);
        }
    }
}
