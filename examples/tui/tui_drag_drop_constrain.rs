//! # TUI Drag-Drop Constrain
//!
//! Constrain a dragged item's (x, y) target to (a) snap-grid and
//! (b) window bounds. Returns final position and whether snapping
//! or clamping occurred.
//!
//! Demonstrates the **TUI.98** recipe for PMAT-192 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML Drag-and-Drop API constrainTo conventions; macOS
//!  Cocoa NSEvent locationInWindow.
//!
//! Run with: cargo run --example tui_drag_drop_constrain
//!
//! Added by PMAT-192 (catalog 1351→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DragVerdict {
    Ok {
        x: i32,
        y: i32,
        snapped: bool,
        clamped: bool,
    },
    InvalidConfig,
}

#[allow(clippy::too_many_arguments)]
pub fn constrain(target_x: i32, target_y: i32, win_w: i32, win_h: i32, grid: u32) -> DragVerdict {
    if win_w <= 0 || win_h <= 0 || grid == 0 {
        return DragVerdict::InvalidConfig;
    }
    let g = grid as i32;
    let snap_x = (target_x / g) * g;
    let snap_y = (target_y / g) * g;
    let snapped = snap_x != target_x || snap_y != target_y;
    let final_x = snap_x.clamp(0, win_w - 1);
    let final_y = snap_y.clamp(0, win_h - 1);
    let clamped = final_x != snap_x || final_y != snap_y;
    DragVerdict::Ok {
        x: final_x,
        y: final_y,
        snapped,
        clamped,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_drag_drop_constrain")?;

    println!("snap: {:?}", constrain(13, 27, 100, 100, 10));
    println!("clamp: {:?}", constrain(150, 80, 100, 100, 10));
    println!("invalid: {:?}", constrain(0, 0, 0, 100, 10));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constrainer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn snap_to_grid() {
        let v = constrain(13, 27, 100, 100, 10);
        if let DragVerdict::Ok { x, y, snapped, .. } = v {
            assert_eq!(x, 10);
            assert_eq!(y, 20);
            assert!(snapped);
        }
    }

    #[test]
    fn clamp_to_window() {
        let v = constrain(150, 80, 100, 100, 10);
        if let DragVerdict::Ok { x, clamped, .. } = v {
            assert!(x < 100);
            assert!(clamped);
        }
    }

    #[test]
    fn invalid_zero_window() {
        assert_eq!(constrain(0, 0, 0, 100, 10), DragVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_grid() {
        assert_eq!(constrain(0, 0, 100, 100, 0), DragVerdict::InvalidConfig);
    }

    #[test]
    fn negative_target_clamped_to_zero() {
        let v = constrain(-50, -10, 100, 100, 10);
        if let DragVerdict::Ok { x, y, clamped, .. } = v {
            assert_eq!(x, 0);
            assert_eq!(y, 0);
            assert!(clamped);
        }
    }

    #[test]
    fn on_grid_no_snap() {
        let v = constrain(20, 30, 100, 100, 10);
        if let DragVerdict::Ok { snapped, .. } = v {
            assert!(!snapped);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = constrain(13, 27, 100, 100, 10);
        let r2 = constrain(13, 27, 100, 100, 10);
        assert_eq!(r1, r2);
    }

    #[test]
    fn within_bounds_no_clamp() {
        let v = constrain(10, 20, 100, 100, 10);
        if let DragVerdict::Ok { clamped, .. } = v {
            assert!(!clamped);
        }
    }

    #[test]
    fn grid_one_no_snap() {
        let v = constrain(13, 27, 100, 100, 1);
        if let DragVerdict::Ok { snapped, .. } = v {
            assert!(!snapped);
        }
    }

    #[test]
    fn x_y_in_window_bounds() {
        let v = constrain(50, 50, 100, 100, 10);
        if let DragVerdict::Ok { x, y, .. } = v {
            assert!(x >= 0 && x < 100);
            assert!(y >= 0 && y < 100);
        }
    }

    #[test]
    fn negative_window_rejected() {
        assert_eq!(constrain(0, 0, -1, 100, 10), DragVerdict::InvalidConfig);
    }
}
