//! # TUI Resize Grid Layout
//!
//! Re-flow a fixed grid of widgets when the terminal resizes. Each
//! widget keeps its preferred (w,h); the layout packs them in row-
//! major order and reports overflow rows.
//!
//! Demonstrates the **TUI.32** recipe for PMAT-170 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: CSS flex-wrap row-major reflow.
//!
//! Run with: cargo run --example tui_resize_grid_layout
//!
//! Added by PMAT-170 (catalog 1153→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum LayoutVerdict {
    Ok {
        positions: Vec<(u32, u32)>,
        rows_used: u32,
        overflow: u32,
    },
    InvalidViewport,
}

pub fn layout(
    widget_widths: &[u32],
    cell_height: u32,
    viewport_width: u32,
    viewport_height: u32,
) -> LayoutVerdict {
    if viewport_width == 0 || viewport_height == 0 || cell_height == 0 {
        return LayoutVerdict::InvalidViewport;
    }
    let mut positions = Vec::with_capacity(widget_widths.len());
    let mut row: u32 = 0;
    let mut col_offset: u32 = 0;
    let mut overflow: u32 = 0;
    let max_rows = viewport_height / cell_height;
    for &w in widget_widths {
        if w > viewport_width {
            overflow += 1;
            continue;
        }
        if col_offset + w > viewport_width {
            row += 1;
            col_offset = 0;
        }
        if row >= max_rows {
            overflow += 1;
            continue;
        }
        positions.push((col_offset, row * cell_height));
        col_offset += w;
    }
    let rows_used = if positions.is_empty() { 0 } else { row + 1 };
    LayoutVerdict::Ok {
        positions,
        rows_used,
        overflow,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_resize_grid_layout")?;

    let widths = [10, 10, 10, 10, 10];
    println!("fits: {:?}", layout(&widths, 5, 30, 20));
    println!("wraps: {:?}", layout(&widths, 5, 20, 20));
    println!("invalid: {:?}", layout(&widths, 5, 0, 20));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn fits_in_one_row() {
        let v = layout(&[10, 10, 10], 5, 30, 20);
        if let LayoutVerdict::Ok {
            rows_used,
            positions,
            overflow,
        } = v
        {
            assert_eq!(rows_used, 1);
            assert_eq!(positions.len(), 3);
            assert_eq!(overflow, 0);
        }
    }

    #[test]
    fn wraps_to_second_row() {
        let v = layout(&[10, 10, 10, 10], 5, 25, 20);
        if let LayoutVerdict::Ok {
            rows_used,
            positions,
            ..
        } = v
        {
            assert_eq!(rows_used, 2);
            assert_eq!(positions.len(), 4);
        }
    }

    #[test]
    fn over_viewport_height_overflow() {
        // 3 widgets per row × 2 rows = 6 widgets, anything more overflows.
        let widths = [10; 10];
        let v = layout(&widths, 5, 30, 10);
        if let LayoutVerdict::Ok { overflow, .. } = v {
            assert!(overflow > 0);
        }
    }

    #[test]
    fn over_viewport_width_overflow() {
        let v = layout(&[100], 5, 50, 20);
        if let LayoutVerdict::Ok { overflow, .. } = v {
            assert_eq!(overflow, 1);
        }
    }

    #[test]
    fn invalid_zero_viewport_width() {
        assert_eq!(layout(&[10], 5, 0, 20), LayoutVerdict::InvalidViewport);
    }

    #[test]
    fn invalid_zero_viewport_height() {
        assert_eq!(layout(&[10], 5, 30, 0), LayoutVerdict::InvalidViewport);
    }

    #[test]
    fn invalid_zero_cell_height() {
        assert_eq!(layout(&[10], 0, 30, 20), LayoutVerdict::InvalidViewport);
    }

    #[test]
    fn empty_input_zero_rows() {
        let v = layout(&[], 5, 30, 20);
        if let LayoutVerdict::Ok { rows_used, .. } = v {
            assert_eq!(rows_used, 0);
        }
    }

    #[test]
    fn positions_in_bounds() {
        let v = layout(&[5, 5, 5, 5], 5, 30, 20);
        if let LayoutVerdict::Ok { positions, .. } = v {
            for (x, y) in positions {
                assert!(x < 30);
                assert!(y < 20);
            }
        }
    }

    #[test]
    fn deterministic() {
        let a = layout(&[10, 10, 10], 5, 30, 20);
        let b = layout(&[10, 10, 10], 5, 30, 20);
        assert_eq!(a, b);
    }
}
