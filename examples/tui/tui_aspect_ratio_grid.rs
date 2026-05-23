//! # TUI Aspect Ratio Grid Solver
//!
//! Given total cells `n` and aspect ratio (w_ratio:h_ratio), compute
//! `(cols, rows)` that fits all cells while approximating the
//! requested aspect ratio.
//!
//! Demonstrates the **TUI.81** recipe for PMAT-186 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: CSS Grid auto-flow; rectangle packing problem.
//!
//! Run with: cargo run --example tui_aspect_ratio_grid
//!
//! Added by PMAT-186 (catalog 1297→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GridVerdict {
    Ok {
        cols: u32,
        rows: u32,
        wasted_cells: u32,
    },
    InvalidConfig,
}

pub fn solve(cells: u32, w_ratio: u32, h_ratio: u32) -> GridVerdict {
    if cells == 0 || w_ratio == 0 || h_ratio == 0 {
        return GridVerdict::InvalidConfig;
    }
    // Solve cols * rows >= cells with cols/rows ≈ w_ratio/h_ratio.
    // → cols ≈ sqrt(cells * w_ratio / h_ratio).
    let target = f64::from(cells) * f64::from(w_ratio) / f64::from(h_ratio);
    let cols = (target.sqrt().ceil() as u32).max(1);
    let rows = cells.div_ceil(cols);
    let wasted = cols * rows - cells;
    GridVerdict::Ok {
        cols,
        rows,
        wasted_cells: wasted,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_aspect_ratio_grid")?;

    println!("16:9 / 100: {:?}", solve(100, 16, 9));
    println!("4:3 / 50: {:?}", solve(50, 4, 3));
    println!("1:1 / 100: {:?}", solve(100, 1, 1));
    println!("invalid: {:?}", solve(0, 1, 1));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn square_grid_for_unit_aspect() {
        let v = solve(100, 1, 1);
        if let GridVerdict::Ok { cols, rows, .. } = v {
            // 100 cells with 1:1 → 10×10.
            assert_eq!(cols, 10);
            assert_eq!(rows, 10);
        }
    }

    #[test]
    fn wider_aspect_more_cols() {
        let s = solve(100, 1, 1);
        let w = solve(100, 4, 1);
        if let (GridVerdict::Ok { cols: sc, .. }, GridVerdict::Ok { cols: wc, .. }) = (s, w) {
            assert!(wc > sc);
        }
    }

    #[test]
    fn cells_fit_in_grid() {
        let v = solve(100, 16, 9);
        if let GridVerdict::Ok { cols, rows, .. } = v {
            assert!(cols * rows >= 100);
        }
    }

    #[test]
    fn invalid_zero_cells() {
        assert_eq!(solve(0, 1, 1), GridVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_w_ratio() {
        assert_eq!(solve(100, 0, 1), GridVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_h_ratio() {
        assert_eq!(solve(100, 1, 0), GridVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = solve(100, 16, 9);
        let r2 = solve(100, 16, 9);
        assert_eq!(r1, r2);
    }

    #[test]
    fn wasted_cells_nonneg() {
        let v = solve(100, 16, 9);
        if let GridVerdict::Ok {
            cols,
            rows,
            wasted_cells,
        } = v
        {
            assert_eq!(cols * rows - 100, wasted_cells);
        }
    }

    #[test]
    fn single_cell_one_by_one() {
        let v = solve(1, 1, 1);
        if let GridVerdict::Ok { cols, rows, .. } = v {
            assert_eq!(cols, 1);
            assert_eq!(rows, 1);
        }
    }

    #[test]
    fn perfect_square_no_waste() {
        let v = solve(16, 1, 1);
        if let GridVerdict::Ok { wasted_cells, .. } = v {
            assert_eq!(wasted_cells, 0);
        }
    }

    #[test]
    fn nine_cells_three_by_three() {
        let v = solve(9, 1, 1);
        if let GridVerdict::Ok { cols, rows, .. } = v {
            assert_eq!(cols, 3);
            assert_eq!(rows, 3);
        }
    }

    #[test]
    fn taller_aspect_more_rows() {
        let wide = solve(100, 4, 1);
        let tall = solve(100, 1, 4);
        if let (GridVerdict::Ok { rows: w_r, .. }, GridVerdict::Ok { rows: t_r, .. }) = (wide, tall)
        {
            assert!(t_r > w_r);
        }
    }
}
