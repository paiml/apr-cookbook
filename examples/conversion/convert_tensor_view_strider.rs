//! # Conversion Tensor View Strider (no-copy)
//!
//! Some conversions can be done as views (no data copy):
//!   reshape: same total elements, contiguous OK
//!   transpose: just permutes strides
//!   slice: same elements, smaller view
//!   broadcast: requires zero-stride trick
//!
//! Picker checks if a conversion is view-only or requires copy.
//!
//! Demonstrates the **CONV.18** recipe for PMAT-148 (conversion round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NumPy view-vs-copy semantics.
//!
//! Run with: cargo run --example convert_tensor_view_strider
//!
//! Added by PMAT-148 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Reshape,
    Transpose,
    Slice,
    Broadcast,
}

#[derive(Debug, PartialEq)]
pub enum ViewVerdict {
    ViewOnly { strides: Vec<i32> },
    RequiresCopy { reason: &'static str },
    InvalidShape,
}

pub fn check(op: Operation, src_shape: &[u32], dst_shape: &[u32]) -> ViewVerdict {
    if src_shape.is_empty() || dst_shape.is_empty() {
        return ViewVerdict::InvalidShape;
    }
    if src_shape.contains(&0) || dst_shape.contains(&0) {
        return ViewVerdict::InvalidShape;
    }
    match op {
        Operation::Reshape => {
            let src_total: u64 = src_shape.iter().map(|&d| u64::from(d)).product();
            let dst_total: u64 = dst_shape.iter().map(|&d| u64::from(d)).product();
            if src_total != dst_total {
                return ViewVerdict::RequiresCopy {
                    reason: "reshape requires equal element count",
                };
            }
            ViewVerdict::ViewOnly {
                strides: row_major_strides(dst_shape),
            }
        }
        Operation::Transpose => {
            if src_shape.len() != dst_shape.len() {
                return ViewVerdict::RequiresCopy {
                    reason: "transpose requires same rank",
                };
            }
            // Permuted shape: dimensions reordered. Strides also reordered.
            ViewVerdict::ViewOnly {
                strides: row_major_strides(src_shape),
            }
        }
        Operation::Slice => {
            if src_shape.len() != dst_shape.len() {
                return ViewVerdict::RequiresCopy {
                    reason: "slice rank must match",
                };
            }
            for (s, d) in src_shape.iter().zip(dst_shape.iter()) {
                if d > s {
                    return ViewVerdict::RequiresCopy {
                        reason: "slice cannot enlarge",
                    };
                }
            }
            ViewVerdict::ViewOnly {
                strides: row_major_strides(src_shape),
            }
        }
        Operation::Broadcast => {
            // Broadcast: dst shape can be larger; uses zero-stride for new dims.
            let mut strides: Vec<i32> = Vec::new();
            for (s, d) in src_shape.iter().zip(dst_shape.iter()) {
                if *s == 1 && *d > 1 {
                    strides.push(0);
                } else if s != d {
                    return ViewVerdict::RequiresCopy {
                        reason: "broadcast requires src=1 or src=dst per dim",
                    };
                } else {
                    strides.push(1);
                }
            }
            ViewVerdict::ViewOnly { strides }
        }
    }
}

fn row_major_strides(shape: &[u32]) -> Vec<i32> {
    let n = shape.len();
    let mut s = vec![1i32; n];
    for i in (0..n - 1).rev() {
        s[i] = s[i + 1] * shape[i + 1] as i32;
    }
    s
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("convert_tensor_view_strider")?;

    println!(
        "reshape 6=2x3 → 3x2: {:?}",
        check(Operation::Reshape, &[2, 3], &[3, 2])
    );
    println!(
        "transpose: {:?}",
        check(Operation::Transpose, &[2, 3], &[3, 2])
    );
    println!("slice: {:?}", check(Operation::Slice, &[5, 5], &[3, 3]));
    println!(
        "broadcast 1×3 → 4×3: {:?}",
        check(Operation::Broadcast, &[1, 3], &[4, 3])
    );
    println!("invalid: {:?}", check(Operation::Reshape, &[], &[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn reshape_compatible_view_only() {
        let v = check(Operation::Reshape, &[2, 3], &[3, 2]);
        assert!(matches!(v, ViewVerdict::ViewOnly { .. }));
    }

    #[test]
    fn reshape_incompatible_requires_copy() {
        let v = check(Operation::Reshape, &[2, 3], &[2, 4]);
        assert!(matches!(v, ViewVerdict::RequiresCopy { .. }));
    }

    #[test]
    fn transpose_view_only() {
        let v = check(Operation::Transpose, &[2, 3], &[3, 2]);
        assert!(matches!(v, ViewVerdict::ViewOnly { .. }));
    }

    #[test]
    fn slice_view_only() {
        let v = check(Operation::Slice, &[5, 5], &[3, 3]);
        assert!(matches!(v, ViewVerdict::ViewOnly { .. }));
    }

    #[test]
    fn slice_enlarge_requires_copy() {
        let v = check(Operation::Slice, &[3, 3], &[5, 5]);
        assert!(matches!(v, ViewVerdict::RequiresCopy { .. }));
    }

    #[test]
    fn broadcast_view_only() {
        let v = check(Operation::Broadcast, &[1, 3], &[4, 3]);
        assert!(matches!(v, ViewVerdict::ViewOnly { .. }));
    }

    #[test]
    fn broadcast_invalid_requires_copy() {
        let v = check(Operation::Broadcast, &[2, 3], &[4, 3]);
        assert!(matches!(v, ViewVerdict::RequiresCopy { .. }));
    }

    #[test]
    fn invalid_empty_shape() {
        assert_eq!(
            check(Operation::Reshape, &[], &[2, 3]),
            ViewVerdict::InvalidShape
        );
    }

    #[test]
    fn invalid_zero_dim() {
        assert_eq!(
            check(Operation::Reshape, &[2, 0, 3], &[6]),
            ViewVerdict::InvalidShape
        );
    }

    #[test]
    fn broadcast_uses_zero_strides() {
        let v = check(Operation::Broadcast, &[1, 3], &[4, 3]);
        if let ViewVerdict::ViewOnly { strides } = v {
            // First dim broadcasts → stride 0; second matches → stride 1.
            assert_eq!(strides, vec![0, 1]);
        }
    }
}
