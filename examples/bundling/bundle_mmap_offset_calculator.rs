//! # Bundle mmap-Compatible Offset Calculator
//!
//! For zero-copy mmap, each tensor must start on a page boundary
//! (typically 4096 bytes). This recipe computes per-tensor offsets that
//! satisfy the constraint, plus the total bundle size.
//!
//! Demonstrates the **BUNDLE.17** recipe for PMAT-136 (bundling round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: POSIX mmap(2) page-alignment requirement.
//!
//! Run with: cargo run --example bundle_mmap_offset_calculator
//!
//! Added by PMAT-136 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const DEFAULT_PAGE_BYTES: u64 = 4096;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorPlacement {
    pub name: String,
    pub offset: u64,
    pub size: u64,
}

#[derive(Debug, PartialEq)]
pub enum LayoutVerdict {
    Ok {
        placements: Vec<TensorPlacement>,
        total_size: u64,
        padding_bytes: u64,
    },
    EmptyTensors,
    InvalidPageSize,
    OffsetOverflow,
}

pub fn lay_out(tensors: &[(&str, u64)], page_bytes: u64) -> LayoutVerdict {
    if tensors.is_empty() {
        return LayoutVerdict::EmptyTensors;
    }
    if page_bytes == 0 || !page_bytes.is_power_of_two() {
        return LayoutVerdict::InvalidPageSize;
    }
    let mask = page_bytes - 1;
    let mut placements = Vec::with_capacity(tensors.len());
    let mut offset = 0u64;
    let mut padding = 0u64;
    for (name, size) in tensors {
        let aligned = (offset + mask) & !mask;
        let pad = aligned - offset;
        padding += pad;
        let new_end = aligned.checked_add(*size);
        if new_end.is_none() {
            return LayoutVerdict::OffsetOverflow;
        }
        placements.push(TensorPlacement {
            name: (*name).to_string(),
            offset: aligned,
            size: *size,
        });
        offset = new_end.unwrap();
    }
    LayoutVerdict::Ok {
        placements,
        total_size: offset,
        padding_bytes: padding,
    }
}

pub fn lay_out_default(tensors: &[(&str, u64)]) -> LayoutVerdict {
    lay_out(tensors, DEFAULT_PAGE_BYTES)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("bundle_mmap_offset_calculator")?;

    let tensors = [("embed", 1500u64), ("layer.0", 2000), ("layer.1", 5000)];
    println!("4 KiB pages: {:?}", lay_out_default(&tensors));
    println!("64 KiB pages: {:?}", lay_out(&tensors, 65_536));
    println!("empty: {:?}", lay_out_default(&[]));
    println!("non-pow2 page: {:?}", lay_out(&tensors, 100));
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
    fn first_tensor_starts_at_zero() {
        let v = lay_out_default(&[("a", 1000)]);
        if let LayoutVerdict::Ok { placements, .. } = v {
            assert_eq!(placements[0].offset, 0);
        }
    }

    #[test]
    fn second_tensor_aligned_to_next_page() {
        // First tensor 1500 bytes → next page boundary is 4096.
        let v = lay_out_default(&[("a", 1500), ("b", 1000)]);
        if let LayoutVerdict::Ok { placements, .. } = v {
            assert_eq!(placements[1].offset, 4096);
        }
    }

    #[test]
    fn exact_page_size_no_padding() {
        // First tensor exactly 4096 → second at 4096.
        let v = lay_out_default(&[("a", 4096), ("b", 1000)]);
        if let LayoutVerdict::Ok { placements, .. } = v {
            assert_eq!(placements[1].offset, 4096);
        }
    }

    #[test]
    fn empty_tensor_list_rejected() {
        assert_eq!(lay_out_default(&[]), LayoutVerdict::EmptyTensors);
    }

    #[test]
    fn non_power_of_two_page_rejected() {
        assert_eq!(lay_out(&[("a", 100)], 100), LayoutVerdict::InvalidPageSize);
    }

    #[test]
    fn zero_page_rejected() {
        assert_eq!(lay_out(&[("a", 100)], 0), LayoutVerdict::InvalidPageSize);
    }

    #[test]
    fn offsets_strictly_increasing() {
        let v = lay_out_default(&[("a", 1000), ("b", 5000), ("c", 200)]);
        if let LayoutVerdict::Ok { placements, .. } = v {
            for w in placements.windows(2) {
                assert!(w[1].offset > w[0].offset);
            }
        }
    }

    #[test]
    fn total_size_includes_data() {
        let v = lay_out_default(&[("a", 1500), ("b", 2000)]);
        if let LayoutVerdict::Ok {
            placements,
            total_size,
            ..
        } = v
        {
            // Last placement offset + size = total_size.
            let last = placements.last().unwrap();
            assert_eq!(total_size, last.offset + last.size);
        }
    }

    #[test]
    fn padding_reported_correctly() {
        // tensor "a" (1500) + pad (2596) + tensor "b" → padding == 2596.
        let v = lay_out_default(&[("a", 1500), ("b", 1000)]);
        if let LayoutVerdict::Ok { padding_bytes, .. } = v {
            assert_eq!(padding_bytes, 2596);
        }
    }

    #[test]
    fn offsets_aligned_to_page() {
        let v = lay_out_default(&[("a", 1500), ("b", 7777), ("c", 33)]);
        if let LayoutVerdict::Ok { placements, .. } = v {
            for p in placements {
                assert_eq!(p.offset % DEFAULT_PAGE_BYTES, 0);
            }
        }
    }

    #[test]
    fn custom_page_size_works() {
        // 64 KiB page → first tensor 1500, second at 65536.
        let v = lay_out(&[("a", 1500), ("b", 1000)], 65_536);
        if let LayoutVerdict::Ok { placements, .. } = v {
            assert_eq!(placements[1].offset, 65_536);
        }
    }
}
