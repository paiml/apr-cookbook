//! # SIMD Lane Swizzle Validate
//!
//! Validate a swizzle index pattern: each index < lane_count, no
//! duplicates if requested (true permutation). Returns categorical
//! verdict.
//!
//! Demonstrates the **SIMD.X** recipe for PMAT-220 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: WebAssembly SIMD i8x16.shuffle imm validation; SSE
//!  PSHUFB index masks.
//!
//! Run with: cargo run --example simd_lane_swizzle_validate
//!
//! Added by PMAT-220 (catalog 1603→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum SwizzleVerdict {
    Valid,
    OutOfRange { lane: u32, index: u32 },
    Duplicate { index: u32 },
    InvalidConfig,
}

pub fn validate(indices: &[u32], lane_count: u32, require_permutation: bool) -> SwizzleVerdict {
    if indices.is_empty() || !(2..=64).contains(&lane_count) {
        return SwizzleVerdict::InvalidConfig;
    }
    if indices.len() != lane_count as usize {
        return SwizzleVerdict::InvalidConfig;
    }
    for (i, idx) in indices.iter().enumerate() {
        if *idx >= lane_count {
            return SwizzleVerdict::OutOfRange {
                lane: i as u32,
                index: *idx,
            };
        }
    }
    if require_permutation {
        let unique: BTreeSet<u32> = indices.iter().copied().collect();
        if unique.len() != indices.len() {
            // Find first duplicate.
            let mut seen: BTreeSet<u32> = BTreeSet::new();
            for idx in indices {
                if !seen.insert(*idx) {
                    return SwizzleVerdict::Duplicate { index: *idx };
                }
            }
        }
    }
    SwizzleVerdict::Valid
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("simd_lane_swizzle_validate")?;

    println!("valid: {:?}", validate(&[0, 1, 2, 3], 4, true));
    println!("oob: {:?}", validate(&[0, 1, 2, 5], 4, true));
    println!("dup: {:?}", validate(&[0, 1, 2, 0], 4, true));
    println!("dup-allowed: {:?}", validate(&[0, 0, 1, 1], 4, false));
    println!("invalid: {:?}", validate(&[], 4, true));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn valid_permutation_accepted() {
        assert_eq!(validate(&[0, 1, 2, 3], 4, true), SwizzleVerdict::Valid);
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(validate(&[], 4, true), SwizzleVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_lane_too_small() {
        assert_eq!(validate(&[0], 1, true), SwizzleVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_lane_too_large() {
        assert_eq!(validate(&[0; 65], 65, true), SwizzleVerdict::InvalidConfig);
    }

    #[test]
    fn length_mismatch_rejected() {
        assert_eq!(validate(&[0, 1], 4, true), SwizzleVerdict::InvalidConfig);
    }

    #[test]
    fn out_of_range_index_rejected() {
        let v = validate(&[0, 1, 2, 5], 4, true);
        if let SwizzleVerdict::OutOfRange { lane, index } = v {
            assert_eq!(lane, 3);
            assert_eq!(index, 5);
        }
    }

    #[test]
    fn duplicate_in_permutation_rejected() {
        let v = validate(&[0, 1, 2, 0], 4, true);
        if let SwizzleVerdict::Duplicate { index } = v {
            assert_eq!(index, 0);
        }
    }

    #[test]
    fn duplicates_allowed_when_not_permutation() {
        assert_eq!(validate(&[0, 0, 1, 1], 4, false), SwizzleVerdict::Valid);
    }

    #[test]
    fn deterministic() {
        let r1 = validate(&[0, 1, 2, 3], 4, true);
        let r2 = validate(&[0, 1, 2, 3], 4, true);
        assert_eq!(r1, r2);
    }

    #[test]
    fn reverse_permutation_valid() {
        assert_eq!(validate(&[3, 2, 1, 0], 4, true), SwizzleVerdict::Valid);
    }

    #[test]
    fn lane_count_64_accepted() {
        let indices: Vec<u32> = (0..64).collect();
        assert_eq!(validate(&indices, 64, true), SwizzleVerdict::Valid);
    }

    #[test]
    fn lane_count_2_accepted() {
        assert_eq!(validate(&[0, 1], 2, true), SwizzleVerdict::Valid);
    }

    #[test]
    fn last_index_at_max_minus_one_valid() {
        assert_eq!(validate(&[0, 1, 2, 3], 4, false), SwizzleVerdict::Valid);
    }
}
