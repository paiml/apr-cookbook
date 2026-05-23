//! # SIMD Mask Lane Predicate
//!
//! Predicated SIMD load/store: only lanes with mask=1 read/write
//! memory. Used for tail-handling (last partial vector at the end of a
//! tensor). This recipe builds the mask generator + the per-lane apply.
//!
//! Demonstrates the **SIMD.9** recipe for PMAT-134 (simd coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AVX-512 vector mask register `kN` semantics; ARM SVE `pred`.
//!
//! Run with: cargo run --example simd_mask_lane_predicate
//!
//! Added by PMAT-134 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MAX_LANES: usize = 16;

#[derive(Debug, PartialEq)]
pub enum MaskVerdict {
    Ok { mask: u16, active_lanes: u32 },
    LaneCountTooHigh { lanes: usize },
    PrefixWiderThanLanes { prefix: usize, lanes: usize },
}

pub fn prefix_mask(active_prefix: usize, total_lanes: usize) -> MaskVerdict {
    if total_lanes == 0 || total_lanes > MAX_LANES {
        return MaskVerdict::LaneCountTooHigh { lanes: total_lanes };
    }
    if active_prefix > total_lanes {
        return MaskVerdict::PrefixWiderThanLanes {
            prefix: active_prefix,
            lanes: total_lanes,
        };
    }
    let mask = if active_prefix == 0 {
        0u16
    } else {
        ((1u32 << active_prefix) - 1) as u16
    };
    MaskVerdict::Ok {
        mask,
        active_lanes: active_prefix as u32,
    }
}

pub fn apply_mask(values: &[f32], mask: u16) -> Vec<f32> {
    values
        .iter()
        .enumerate()
        .map(|(i, v)| if (mask >> i) & 1 == 1 { *v } else { 0.0 })
        .collect()
}

pub fn tail_lanes_for(total_elems: usize, lane_width: usize) -> usize {
    if lane_width == 0 {
        return 0;
    }
    total_elems % lane_width
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("simd_mask_lane_predicate")?;

    println!("prefix=4 of 8: {:?}", prefix_mask(4, 8));
    println!("prefix=8 of 8 (full): {:?}", prefix_mask(8, 8));
    println!("prefix=0: {:?}", prefix_mask(0, 8));
    println!("invalid prefix>lanes: {:?}", prefix_mask(9, 8));

    let v = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    if let MaskVerdict::Ok { mask, .. } = prefix_mask(5, 8) {
        println!("apply mask 0b11111: {:?}", apply_mask(&v, mask));
    }

    println!("tail of 130 elems / lane 16: {}", tail_lanes_for(130, 16));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn predicate_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn full_prefix_all_ones() {
        let v = prefix_mask(8, 8);
        assert!(matches!(
            v,
            MaskVerdict::Ok {
                mask: 0xFF,
                active_lanes: 8
            }
        ));
    }

    #[test]
    fn empty_prefix_zero_mask() {
        let v = prefix_mask(0, 8);
        assert!(matches!(
            v,
            MaskVerdict::Ok {
                mask: 0,
                active_lanes: 0
            }
        ));
    }

    #[test]
    fn partial_prefix_sets_low_bits() {
        let v = prefix_mask(4, 8);
        assert!(matches!(
            v,
            MaskVerdict::Ok {
                mask: 0b1111,
                active_lanes: 4
            }
        ));
    }

    #[test]
    fn prefix_wider_than_lanes_rejected() {
        assert_eq!(
            prefix_mask(9, 8),
            MaskVerdict::PrefixWiderThanLanes {
                prefix: 9,
                lanes: 8
            }
        );
    }

    #[test]
    fn lane_count_zero_rejected() {
        assert_eq!(
            prefix_mask(0, 0),
            MaskVerdict::LaneCountTooHigh { lanes: 0 }
        );
    }

    #[test]
    fn lane_count_above_max_rejected() {
        assert_eq!(
            prefix_mask(0, 17),
            MaskVerdict::LaneCountTooHigh { lanes: 17 }
        );
    }

    #[test]
    fn apply_mask_zeroes_inactive_lanes() {
        let v = [1.0f32, 2.0, 3.0, 4.0];
        let out = apply_mask(&v, 0b0011);
        assert_eq!(out, vec![1.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn apply_full_mask_passes_through() {
        let v = [1.0f32, 2.0, 3.0, 4.0];
        let out = apply_mask(&v, 0b1111);
        assert_eq!(out, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn tail_lanes_typical() {
        // 130 elems / 16-wide = 8 full vectors + 2 tail.
        assert_eq!(tail_lanes_for(130, 16), 2);
    }

    #[test]
    fn tail_lanes_evenly_divisible_zero() {
        assert_eq!(tail_lanes_for(128, 16), 0);
    }

    #[test]
    fn tail_lanes_zero_lane_width_safe() {
        assert_eq!(tail_lanes_for(100, 0), 0);
    }
}
