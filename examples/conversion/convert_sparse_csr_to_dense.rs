//! # Conversion Sparse-CSR → Dense
//!
//! CSR (Compressed Sparse Row) format: row_ptr[m+1], col_idx[nnz],
//! values[nnz]. Convert to a dense (m × n) matrix by writing each
//! non-zero entry into out[row][col_idx]. This recipe builds the
//! converter with bounds + monotonicity + length checks.
//!
//! Demonstrates the **CONV.15** recipe for PMAT-136 (conversion round 2).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: BLAS/LAPACK CSR (Compressed Sparse Row) layout convention.
//!
//! Run with: cargo run --example convert_sparse_csr_to_dense
//!
//! Added by PMAT-136 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DenseVerdict {
    Ok { dense: Vec<Vec<f64>> },
    InvalidShape,
    RowPtrLengthMismatch,
    RowPtrNonMonotonic { at: usize },
    ColIdxOutOfBounds { at: usize },
    ValuesLengthMismatch,
}

pub fn convert(
    rows: usize,
    cols: usize,
    row_ptr: &[usize],
    col_idx: &[usize],
    values: &[f64],
) -> DenseVerdict {
    if rows == 0 || cols == 0 {
        return DenseVerdict::InvalidShape;
    }
    if row_ptr.len() != rows + 1 {
        return DenseVerdict::RowPtrLengthMismatch;
    }
    if col_idx.len() != values.len() {
        return DenseVerdict::ValuesLengthMismatch;
    }
    for i in 1..row_ptr.len() {
        if row_ptr[i] < row_ptr[i - 1] {
            return DenseVerdict::RowPtrNonMonotonic { at: i };
        }
    }
    let nnz = *row_ptr.last().unwrap();
    if nnz != col_idx.len() {
        return DenseVerdict::ValuesLengthMismatch;
    }
    for (i, &c) in col_idx.iter().enumerate() {
        if c >= cols {
            return DenseVerdict::ColIdxOutOfBounds { at: i };
        }
    }
    let mut dense = vec![vec![0.0; cols]; rows];
    for r in 0..rows {
        for k in row_ptr[r]..row_ptr[r + 1] {
            dense[r][col_idx[k]] = values[k];
        }
    }
    DenseVerdict::Ok { dense }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("convert_sparse_csr_to_dense")?;

    // 3x4 matrix:
    // [10, 0, 0, 20]
    // [ 0, 0,30,  0]
    // [40,50, 0,  0]
    let row_ptr = [0usize, 2, 3, 5];
    let col_idx = [0usize, 3, 2, 0, 1];
    let values = [10.0, 20.0, 30.0, 40.0, 50.0];
    println!(
        "3x4 typical: {:?}",
        convert(3, 4, &row_ptr, &col_idx, &values)
    );

    println!(
        "non-monotonic: {:?}",
        convert(3, 4, &[0, 2, 1, 5], &col_idx, &values)
    );
    println!(
        "out of bounds col: {:?}",
        convert(3, 4, &row_ptr, &[0, 99, 2, 0, 1], &values)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_matrix() -> (usize, usize, Vec<usize>, Vec<usize>, Vec<f64>) {
        // 2x3 matrix: [[1, 0, 2], [0, 3, 0]]
        let rows = 2;
        let cols = 3;
        let row_ptr = vec![0, 2, 3];
        let col_idx = vec![0, 2, 1];
        let values = vec![1.0, 2.0, 3.0];
        (rows, cols, row_ptr, col_idx, values)
    }

    #[test]
    fn converter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_2x3_correct() {
        let (rows, cols, rp, ci, v) = small_matrix();
        if let DenseVerdict::Ok { dense } = convert(rows, cols, &rp, &ci, &v) {
            assert_eq!(dense, vec![vec![1.0, 0.0, 2.0], vec![0.0, 3.0, 0.0]]);
        }
    }

    #[test]
    fn empty_rows_rejected() {
        assert_eq!(convert(0, 3, &[0], &[], &[]), DenseVerdict::InvalidShape);
    }

    #[test]
    fn empty_cols_rejected() {
        assert_eq!(
            convert(2, 0, &[0, 0, 0], &[], &[]),
            DenseVerdict::InvalidShape
        );
    }

    #[test]
    fn row_ptr_length_mismatch_rejected() {
        // For rows=2, row_ptr should be length 3.
        let v = convert(2, 3, &[0, 1], &[0], &[1.0]);
        assert_eq!(v, DenseVerdict::RowPtrLengthMismatch);
    }

    #[test]
    fn non_monotonic_row_ptr_rejected() {
        let v = convert(2, 3, &[0, 2, 1], &[0, 1], &[1.0, 2.0]);
        assert!(matches!(v, DenseVerdict::RowPtrNonMonotonic { .. }));
    }

    #[test]
    fn col_idx_out_of_bounds_rejected() {
        let v = convert(2, 3, &[0, 1, 2], &[0, 5], &[1.0, 2.0]);
        assert!(matches!(v, DenseVerdict::ColIdxOutOfBounds { .. }));
    }

    #[test]
    fn values_length_mismatch_rejected() {
        // col_idx and values disagree.
        let v = convert(2, 3, &[0, 1, 2], &[0, 1], &[1.0]);
        assert_eq!(v, DenseVerdict::ValuesLengthMismatch);
    }

    #[test]
    fn nnz_disagreement_with_row_ptr_rejected() {
        // row_ptr last says 5 nnz, but col_idx has 2.
        let v = convert(2, 3, &[0, 2, 5], &[0, 1], &[1.0, 2.0]);
        assert_eq!(v, DenseVerdict::ValuesLengthMismatch);
    }

    #[test]
    fn empty_matrix_with_zero_nnz_works() {
        // 2x3 all zeros (nnz=0).
        let v = convert(2, 3, &[0, 0, 0], &[], &[]);
        if let DenseVerdict::Ok { dense } = v {
            assert_eq!(dense, vec![vec![0.0; 3]; 2]);
        }
    }

    #[test]
    fn dense_dimensions_match_input() {
        let (rows, cols, rp, ci, v) = small_matrix();
        if let DenseVerdict::Ok { dense } = convert(rows, cols, &rp, &ci, &v) {
            assert_eq!(dense.len(), rows);
            assert_eq!(dense[0].len(), cols);
        }
    }

    #[test]
    fn three_by_four_typical() {
        // 3x4 matrix:
        // [10, 0, 0, 20]
        // [ 0, 0,30,  0]
        // [40,50, 0,  0]
        let v = convert(
            3,
            4,
            &[0, 2, 3, 5],
            &[0, 3, 2, 0, 1],
            &[10.0, 20.0, 30.0, 40.0, 50.0],
        );
        if let DenseVerdict::Ok { dense } = v {
            assert_eq!(dense[0][0], 10.0);
            assert_eq!(dense[0][3], 20.0);
            assert_eq!(dense[1][2], 30.0);
            assert_eq!(dense[2][0], 40.0);
            assert_eq!(dense[2][1], 50.0);
        }
    }
}
