//! # Conversion NCHW ↔ NHWC Layout Transposer
//!
//! Vision tensors come in two memory layouts: NCHW (PyTorch default,
//! channel-first) and NHWC (TensorFlow default, channel-last). Cross-
//! framework conversion requires a 4D index permutation. This recipe
//! builds the permutation planner + the per-element index translator.
//!
//! Demonstrates the **CONV.10** recipe for PMAT-133 (conversion coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: PyTorch and TensorFlow tensor layout docs.
//!
//! Run with: cargo run --example convert_layout_transposer
//!
//! Added by PMAT-133 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    Nchw,
    Nhwc,
}

#[derive(Debug, PartialEq)]
pub enum TransposeVerdict {
    Ok {
        perm: [usize; 4],
        new_shape: [u32; 4],
    },
    SameLayout,
    InvalidShape,
}

pub fn plan(from: Layout, to: Layout, shape: [u32; 4]) -> TransposeVerdict {
    if shape.contains(&0) {
        return TransposeVerdict::InvalidShape;
    }
    if from == to {
        return TransposeVerdict::SameLayout;
    }
    match (from, to) {
        (Layout::Nchw, Layout::Nhwc) => TransposeVerdict::Ok {
            // NCHW [0=N,1=C,2=H,3=W] → NHWC [0=N,2=H,3=W,1=C].
            perm: [0, 2, 3, 1],
            new_shape: [shape[0], shape[2], shape[3], shape[1]],
        },
        (Layout::Nhwc, Layout::Nchw) => TransposeVerdict::Ok {
            perm: [0, 3, 1, 2],
            new_shape: [shape[0], shape[3], shape[1], shape[2]],
        },
        _ => TransposeVerdict::SameLayout,
    }
}

pub fn translate_index(perm: [usize; 4], src_idx: [u32; 4]) -> [u32; 4] {
    [
        src_idx[perm[0]],
        src_idx[perm[1]],
        src_idx[perm[2]],
        src_idx[perm[3]],
    ]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("convert_layout_transposer")?;

    let nchw = [1u32, 3, 224, 224];
    let nhwc = [1u32, 224, 224, 3];
    println!("NCHW → NHWC: {:?}", plan(Layout::Nchw, Layout::Nhwc, nchw));
    println!("NHWC → NCHW: {:?}", plan(Layout::Nhwc, Layout::Nchw, nhwc));
    println!("same: {:?}", plan(Layout::Nchw, Layout::Nchw, nchw));
    println!(
        "zero dim: {:?}",
        plan(Layout::Nchw, Layout::Nhwc, [1, 0, 224, 224])
    );

    let perm = [0usize, 2, 3, 1];
    let translated = translate_index(perm, [0, 5, 10, 20]);
    println!("translate [0,5,10,20] perm [0,2,3,1] → {translated:?}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transposer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn nchw_to_nhwc_perm_correct() {
        // [N=1, C=3, H=224, W=224] → [1, 224, 224, 3].
        let v = plan(Layout::Nchw, Layout::Nhwc, [1, 3, 224, 224]);
        assert!(matches!(
            v,
            TransposeVerdict::Ok {
                perm: [0, 2, 3, 1],
                new_shape: [1, 224, 224, 3]
            }
        ));
    }

    #[test]
    fn nhwc_to_nchw_perm_correct() {
        let v = plan(Layout::Nhwc, Layout::Nchw, [1, 224, 224, 3]);
        assert!(matches!(
            v,
            TransposeVerdict::Ok {
                perm: [0, 3, 1, 2],
                new_shape: [1, 3, 224, 224]
            }
        ));
    }

    #[test]
    fn same_layout_short_circuits() {
        assert_eq!(
            plan(Layout::Nchw, Layout::Nchw, [1, 3, 224, 224]),
            TransposeVerdict::SameLayout
        );
    }

    #[test]
    fn zero_dimension_invalid() {
        assert_eq!(
            plan(Layout::Nchw, Layout::Nhwc, [1, 0, 224, 224]),
            TransposeVerdict::InvalidShape
        );
    }

    #[test]
    fn round_trip_recovers_original_shape() {
        let original = [2u32, 4, 8, 16];
        let v = plan(Layout::Nchw, Layout::Nhwc, original);
        if let TransposeVerdict::Ok { new_shape, .. } = v {
            let v2 = plan(Layout::Nhwc, Layout::Nchw, new_shape);
            if let TransposeVerdict::Ok {
                new_shape: back, ..
            } = v2
            {
                assert_eq!(back, original);
            }
        }
    }

    #[test]
    fn translate_index_basic() {
        // perm [0,2,3,1] means dst[i] = src[perm[i]].
        // src [N=0, C=5, H=10, W=20] under perm [0,2,3,1] → [0, 10, 20, 5].
        let translated = translate_index([0, 2, 3, 1], [0, 5, 10, 20]);
        assert_eq!(translated, [0, 10, 20, 5]);
    }

    #[test]
    fn translate_index_identity_perm() {
        let perm = [0, 1, 2, 3];
        let original = [1u32, 2, 3, 4];
        assert_eq!(translate_index(perm, original), original);
    }

    #[test]
    fn channel_last_to_first_swaps_correctly() {
        // [N, H, W, C] → [N, C, H, W].
        let perm = [0usize, 3, 1, 2];
        let translated = translate_index(perm, [1, 2, 3, 4]);
        assert_eq!(translated, [1, 4, 2, 3]);
    }

    #[test]
    fn batch_dim_unchanged_in_both_perms() {
        // N is index 0 in both NCHW and NHWC.
        if let TransposeVerdict::Ok { perm, .. } =
            plan(Layout::Nchw, Layout::Nhwc, [4, 3, 224, 224])
        {
            assert_eq!(perm[0], 0);
        }
    }
}
