//! # Format Tensor Stride Encoder
//!
//! Stride = element offset to advance by 1 in a given dimension.
//! Row-major (C-style): stride[d] = product(shape[d+1..]).
//! Column-major (Fortran-style): stride[d] = product(shape[..d]).
//! This recipe builds both encoders + the contiguity check.
//!
//! Demonstrates the **FMT.21** recipe for PMAT-133 (format coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NumPy stride documentation; Fortran array semantics.
//!
//! Run with: cargo run --example format_stride_encoder
//!
//! Added by PMAT-133 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrideOrder {
    RowMajor,
    ColMajor,
}

#[derive(Debug, PartialEq)]
pub enum StrideVerdict {
    Ok(Vec<u64>),
    EmptyShape,
    ZeroDimension { at_index: usize },
}

pub fn encode(shape: &[u32], order: StrideOrder) -> StrideVerdict {
    if shape.is_empty() {
        return StrideVerdict::EmptyShape;
    }
    for (i, &d) in shape.iter().enumerate() {
        if d == 0 {
            return StrideVerdict::ZeroDimension { at_index: i };
        }
    }
    let n = shape.len();
    let mut strides = vec![1u64; n];
    match order {
        StrideOrder::RowMajor => {
            for i in (0..n - 1).rev() {
                strides[i] = strides[i + 1] * u64::from(shape[i + 1]);
            }
        }
        StrideOrder::ColMajor => {
            for i in 1..n {
                strides[i] = strides[i - 1] * u64::from(shape[i - 1]);
            }
        }
    }
    StrideVerdict::Ok(strides)
}

pub fn is_contiguous_row_major(shape: &[u32], strides: &[u64]) -> bool {
    if shape.is_empty() || shape.len() != strides.len() {
        return false;
    }
    let n = shape.len();
    if strides[n - 1] != 1 {
        return false;
    }
    for i in (0..n - 1).rev() {
        if strides[i] != strides[i + 1] * u64::from(shape[i + 1]) {
            return false;
        }
    }
    true
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("format_stride_encoder")?;

    let shape = [2u32, 3, 4];
    println!(
        "row-major {shape:?}: {:?}",
        encode(&shape, StrideOrder::RowMajor)
    );
    println!(
        "col-major {shape:?}: {:?}",
        encode(&shape, StrideOrder::ColMajor)
    );
    println!("empty: {:?}", encode(&[], StrideOrder::RowMajor));
    println!("zero dim: {:?}", encode(&[2, 0, 4], StrideOrder::RowMajor));

    let strides = vec![12, 4, 1];
    println!(
        "is_contig row-major: {}",
        is_contiguous_row_major(&shape, &strides)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn row_major_3d_correct() {
        // Shape [2, 3, 4]: strides = [12, 4, 1].
        let v = encode(&[2, 3, 4], StrideOrder::RowMajor);
        assert_eq!(v, StrideVerdict::Ok(vec![12, 4, 1]));
    }

    #[test]
    fn col_major_3d_correct() {
        // Shape [2, 3, 4]: strides = [1, 2, 6].
        let v = encode(&[2, 3, 4], StrideOrder::ColMajor);
        assert_eq!(v, StrideVerdict::Ok(vec![1, 2, 6]));
    }

    #[test]
    fn empty_shape_rejected() {
        assert_eq!(
            encode(&[], StrideOrder::RowMajor),
            StrideVerdict::EmptyShape
        );
    }

    #[test]
    fn zero_dim_rejected() {
        let v = encode(&[2, 0, 4], StrideOrder::RowMajor);
        assert_eq!(v, StrideVerdict::ZeroDimension { at_index: 1 });
    }

    #[test]
    fn single_dim_strides_one() {
        let v = encode(&[10], StrideOrder::RowMajor);
        assert_eq!(v, StrideVerdict::Ok(vec![1]));
    }

    #[test]
    fn row_major_4d_correct() {
        // [2, 3, 4, 5]: strides = [60, 20, 5, 1].
        let v = encode(&[2, 3, 4, 5], StrideOrder::RowMajor);
        assert_eq!(v, StrideVerdict::Ok(vec![60, 20, 5, 1]));
    }

    #[test]
    fn contiguous_check_true_for_default() {
        let shape = [2u32, 3, 4];
        let strides = vec![12u64, 4, 1];
        assert!(is_contiguous_row_major(&shape, &strides));
    }

    #[test]
    fn contiguous_check_false_for_strided_view() {
        // A subview with stride[0] = 24 (skipping rows) is not contiguous.
        let shape = [2u32, 3, 4];
        let strides = vec![24u64, 4, 1];
        assert!(!is_contiguous_row_major(&shape, &strides));
    }

    #[test]
    fn contiguous_check_false_for_non_unit_innermost() {
        let shape = [2u32, 3];
        let strides = vec![6u64, 2];
        assert!(!is_contiguous_row_major(&shape, &strides));
    }

    #[test]
    fn row_and_col_major_swap_for_2d() {
        // [3, 4] row-major = [4, 1]; col-major = [1, 3].
        let row = encode(&[3, 4], StrideOrder::RowMajor);
        let col = encode(&[3, 4], StrideOrder::ColMajor);
        assert_eq!(row, StrideVerdict::Ok(vec![4, 1]));
        assert_eq!(col, StrideVerdict::Ok(vec![1, 3]));
    }
}
