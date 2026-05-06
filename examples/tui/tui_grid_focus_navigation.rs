//! # TUI Grid Focus Navigation
//!
//! Compute new (row, col) focus given an arrow-key press in a 2D
//! grid. Wraps at edges (toroidal) or clamps based on `wrap` flag.
//!
//! Demonstrates the **TUI.28** recipe for PMAT-169 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML2 grid-focus / spreadsheet navigation conventions.
//!
//! Run with: cargo run --example tui_grid_focus_navigation
//!
//! Added by PMAT-169 (catalog 1144→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

#[derive(Debug, PartialEq)]
pub enum NavVerdict {
    Ok { row: u32, col: u32 },
    InvalidGrid,
}

pub fn navigate(
    current_row: u32,
    current_col: u32,
    rows: u32,
    cols: u32,
    direction: Direction,
    wrap: bool,
) -> NavVerdict {
    if rows == 0 || cols == 0 {
        return NavVerdict::InvalidGrid;
    }
    let r = current_row.min(rows - 1);
    let c = current_col.min(cols - 1);
    let (new_r, new_c) = match direction {
        Direction::Up => {
            if r == 0 {
                if wrap {
                    (rows - 1, c)
                } else {
                    (0, c)
                }
            } else {
                (r - 1, c)
            }
        }
        Direction::Down => {
            if r + 1 >= rows {
                if wrap {
                    (0, c)
                } else {
                    (rows - 1, c)
                }
            } else {
                (r + 1, c)
            }
        }
        Direction::Left => {
            if c == 0 {
                if wrap {
                    (r, cols - 1)
                } else {
                    (r, 0)
                }
            } else {
                (r, c - 1)
            }
        }
        Direction::Right => {
            if c + 1 >= cols {
                if wrap {
                    (r, 0)
                } else {
                    (r, cols - 1)
                }
            } else {
                (r, c + 1)
            }
        }
    };
    NavVerdict::Ok {
        row: new_r,
        col: new_c,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_grid_focus_navigation")?;

    println!("right: {:?}", navigate(0, 0, 5, 5, Direction::Right, false));
    println!("up clamp: {:?}", navigate(0, 0, 5, 5, Direction::Up, false));
    println!("up wrap: {:?}", navigate(0, 0, 5, 5, Direction::Up, true));
    println!(
        "right at edge wrap: {:?}",
        navigate(2, 4, 5, 5, Direction::Right, true)
    );
    println!(
        "invalid: {:?}",
        navigate(0, 0, 0, 5, Direction::Right, false)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn navigator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn right_advances_col() {
        let v = navigate(0, 0, 5, 5, Direction::Right, false);
        if let NavVerdict::Ok { row, col } = v {
            assert_eq!(row, 0);
            assert_eq!(col, 1);
        }
    }

    #[test]
    fn up_at_top_clamps() {
        let v = navigate(0, 2, 5, 5, Direction::Up, false);
        if let NavVerdict::Ok { row, .. } = v {
            assert_eq!(row, 0);
        }
    }

    #[test]
    fn up_at_top_wraps() {
        let v = navigate(0, 2, 5, 5, Direction::Up, true);
        if let NavVerdict::Ok { row, .. } = v {
            assert_eq!(row, 4);
        }
    }

    #[test]
    fn right_at_edge_wraps() {
        let v = navigate(2, 4, 5, 5, Direction::Right, true);
        if let NavVerdict::Ok { col, .. } = v {
            assert_eq!(col, 0);
        }
    }

    #[test]
    fn down_advances_row() {
        let v = navigate(0, 0, 5, 5, Direction::Down, false);
        if let NavVerdict::Ok { row, .. } = v {
            assert_eq!(row, 1);
        }
    }

    #[test]
    fn left_decreases_col() {
        let v = navigate(0, 2, 5, 5, Direction::Left, false);
        if let NavVerdict::Ok { col, .. } = v {
            assert_eq!(col, 1);
        }
    }

    #[test]
    fn invalid_zero_rows() {
        assert_eq!(
            navigate(0, 0, 0, 5, Direction::Right, false),
            NavVerdict::InvalidGrid
        );
    }

    #[test]
    fn invalid_zero_cols() {
        assert_eq!(
            navigate(0, 0, 5, 0, Direction::Right, false),
            NavVerdict::InvalidGrid
        );
    }

    #[test]
    fn out_of_bounds_clamped() {
        let v = navigate(100, 100, 5, 5, Direction::Right, false);
        if let NavVerdict::Ok { row, col } = v {
            assert_eq!(row, 4);
            assert_eq!(col, 4);
        }
    }

    #[test]
    fn deterministic() {
        let a = navigate(0, 0, 5, 5, Direction::Right, false);
        let b = navigate(0, 0, 5, 5, Direction::Right, false);
        assert_eq!(a, b);
    }
}
