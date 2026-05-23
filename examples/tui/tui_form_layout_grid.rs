//! # TUI Form Layout Grid
//!
//! Assign N form fields to a grid (rows × cols). Returns each
//! field's (row, col) position. Uses row-major order.
//!
//! Demonstrates the **TUI.119** recipe for PMAT-199 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: CSS Grid auto-flow row; macOS NSGridView ordering.
//!
//! Run with: cargo run --example tui_form_layout_grid
//!
//! Added by PMAT-199 (catalog 1414→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum LayoutVerdict {
    Ok {
        positions: Vec<(u32, u32)>,
        used_rows: u32,
    },
    InvalidConfig,
}

pub fn layout(fields: u32, cols: u32) -> LayoutVerdict {
    if fields == 0 || cols == 0 {
        return LayoutVerdict::InvalidConfig;
    }
    let mut positions: Vec<(u32, u32)> = Vec::with_capacity(fields as usize);
    for i in 0..fields {
        let row = i / cols;
        let col = i % cols;
        positions.push((row, col));
    }
    let used_rows = fields.div_ceil(cols);
    LayoutVerdict::Ok {
        positions,
        used_rows,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_form_layout_grid")?;

    println!("8 fields, 3 cols: {:?}", layout(8, 3));
    println!("5 fields, 5 cols: {:?}", layout(5, 5));
    println!("invalid: {:?}", layout(0, 3));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn position_count_matches_fields() {
        let v = layout(8, 3);
        if let LayoutVerdict::Ok { positions, .. } = v {
            assert_eq!(positions.len(), 8);
        }
    }

    #[test]
    fn invalid_zero_fields() {
        assert_eq!(layout(0, 3), LayoutVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_cols() {
        assert_eq!(layout(5, 0), LayoutVerdict::InvalidConfig);
    }

    #[test]
    fn first_field_at_origin() {
        let v = layout(5, 3);
        if let LayoutVerdict::Ok { positions, .. } = v {
            assert_eq!(positions[0], (0, 0));
        }
    }

    #[test]
    fn row_major_order() {
        let v = layout(4, 2);
        if let LayoutVerdict::Ok { positions, .. } = v {
            assert_eq!(positions[0], (0, 0));
            assert_eq!(positions[1], (0, 1));
            assert_eq!(positions[2], (1, 0));
            assert_eq!(positions[3], (1, 1));
        }
    }

    #[test]
    fn used_rows_correct() {
        let v = layout(7, 3);
        if let LayoutVerdict::Ok { used_rows, .. } = v {
            // 7 fields / 3 cols = ceil(7/3) = 3 rows.
            assert_eq!(used_rows, 3);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = layout(8, 3);
        let r2 = layout(8, 3);
        assert_eq!(r1, r2);
    }

    #[test]
    fn single_row_works() {
        let v = layout(5, 5);
        if let LayoutVerdict::Ok { used_rows, .. } = v {
            assert_eq!(used_rows, 1);
        }
    }

    #[test]
    fn single_column_works() {
        let v = layout(5, 1);
        if let LayoutVerdict::Ok { used_rows, .. } = v {
            assert_eq!(used_rows, 5);
        }
    }

    #[test]
    fn col_lt_cols() {
        let v = layout(20, 3);
        if let LayoutVerdict::Ok { positions, .. } = v {
            for (_, c) in &positions {
                assert!(*c < 3);
            }
        }
    }

    #[test]
    fn many_fields_handled() {
        let v = layout(100, 10);
        if let LayoutVerdict::Ok {
            positions,
            used_rows,
        } = v
        {
            assert_eq!(positions.len(), 100);
            assert_eq!(used_rows, 10);
        }
    }
}
