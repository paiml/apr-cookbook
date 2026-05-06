//! # Format Alignment Padding Calculator
//!
//! Tensor offsets in mmap-friendly formats must align to 64-byte
//! boundaries (cache-line + AVX-512 friendly). This recipe computes
//! padding bytes per offset + total padding overhead pct.
//!
//! Demonstrates the **FMT.22** recipe for PMAT-133 (format coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Intel architecture manual § cache line alignment.
//!
//! Run with: cargo run --example format_alignment_padding
//!
//! Added by PMAT-133 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const DEFAULT_ALIGN: u64 = 64;

#[derive(Debug, PartialEq)]
pub enum PadVerdict {
    Ok { padded_offset: u64, pad_bytes: u64 },
    InvalidAlignment,
}

pub fn pad_offset(offset: u64, alignment: u64) -> PadVerdict {
    if alignment == 0 || !alignment.is_power_of_two() {
        return PadVerdict::InvalidAlignment;
    }
    let mask = alignment - 1;
    let pad_bytes = (alignment - (offset & mask)) & mask;
    PadVerdict::Ok {
        padded_offset: offset + pad_bytes,
        pad_bytes,
    }
}

pub fn total_overhead_pct(tensor_sizes: &[u64], alignment: u64) -> Option<f64> {
    if tensor_sizes.is_empty() || alignment == 0 || !alignment.is_power_of_two() {
        return None;
    }
    let mut offset = 0u64;
    let mut total_pad = 0u64;
    let mut total_data = 0u64;
    for &size in tensor_sizes {
        if let PadVerdict::Ok {
            padded_offset,
            pad_bytes,
        } = pad_offset(offset, alignment)
        {
            offset = padded_offset + size;
            total_pad += pad_bytes;
            total_data += size;
        }
    }
    if total_data == 0 {
        return Some(0.0);
    }
    Some((total_pad as f64 / total_data as f64) * 100.0)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("format_alignment_padding")?;

    for offset in [0u64, 1, 63, 64, 65, 128, 130] {
        println!("offset={offset}: {:?}", pad_offset(offset, DEFAULT_ALIGN));
    }

    let sizes = [1000u64, 2000, 3000, 100, 50];
    println!("overhead: {:?}%", total_overhead_pct(&sizes, DEFAULT_ALIGN));

    println!("invalid alignment 7: {:?}", pad_offset(0, 7));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calc_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn already_aligned_no_padding() {
        let v = pad_offset(64, 64);
        assert!(matches!(
            v,
            PadVerdict::Ok {
                padded_offset: 64,
                pad_bytes: 0
            }
        ));
    }

    #[test]
    fn one_byte_misalignment_pads_63() {
        let v = pad_offset(1, 64);
        assert!(matches!(
            v,
            PadVerdict::Ok {
                padded_offset: 64,
                pad_bytes: 63
            }
        ));
    }

    #[test]
    fn boundary_just_after_align_pads_to_next() {
        let v = pad_offset(65, 64);
        assert!(matches!(
            v,
            PadVerdict::Ok {
                padded_offset: 128,
                pad_bytes: 63
            }
        ));
    }

    #[test]
    fn zero_offset_no_padding() {
        let v = pad_offset(0, 64);
        assert!(matches!(
            v,
            PadVerdict::Ok {
                padded_offset: 0,
                pad_bytes: 0
            }
        ));
    }

    #[test]
    fn non_power_of_two_alignment_rejected() {
        assert_eq!(pad_offset(0, 7), PadVerdict::InvalidAlignment);
        assert_eq!(pad_offset(0, 100), PadVerdict::InvalidAlignment);
    }

    #[test]
    fn zero_alignment_rejected() {
        assert_eq!(pad_offset(0, 0), PadVerdict::InvalidAlignment);
    }

    #[test]
    fn alignment_4096_pads_to_page_boundary() {
        let v = pad_offset(1, 4096);
        assert!(matches!(
            v,
            PadVerdict::Ok {
                padded_offset: 4096,
                pad_bytes: 4095
            }
        ));
    }

    #[test]
    fn overhead_pct_basic() {
        // 5 tensors of 1000 bytes each, all aligned-friendly →
        // some pad needed only between them.
        let pct = total_overhead_pct(&[1000, 2000, 3000, 100, 50], DEFAULT_ALIGN);
        assert!(pct.is_some());
        assert!(pct.unwrap() < 100.0);
    }

    #[test]
    fn overhead_zero_for_aligned_sizes() {
        // All multiples of 64 → no padding ever needed.
        let pct = total_overhead_pct(&[64, 128, 256], 64).unwrap();
        assert!(pct.abs() < 1e-9);
    }

    #[test]
    fn overhead_invalid_alignment_returns_none() {
        assert!(total_overhead_pct(&[1000], 7).is_none());
        assert!(total_overhead_pct(&[], 64).is_none());
    }
}
