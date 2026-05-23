//! # TUI Window Layout Anchor
//!
//! Compute (x, y) positions for 4 anchored panels (top-left, top-right,
//! bottom-left, bottom-right) given window dimensions and panel sizes.
//!
//! Demonstrates the **TUI.89** recipe for PMAT-189 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS Cocoa autolayout anchors; CSS position:absolute.
//!
//! Run with: cargo run --example tui_window_layout_anchor
//!
//! Added by PMAT-189 (catalog 1324→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Anchor {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Center,
}

#[derive(Debug, PartialEq)]
pub enum LayoutVerdict {
    Ok { x: u32, y: u32 },
    OutOfBounds,
    InvalidConfig,
}

pub fn position(
    window_w: u32,
    window_h: u32,
    panel_w: u32,
    panel_h: u32,
    anchor: Anchor,
) -> LayoutVerdict {
    if window_w == 0 || window_h == 0 || panel_w == 0 || panel_h == 0 {
        return LayoutVerdict::InvalidConfig;
    }
    if panel_w > window_w || panel_h > window_h {
        return LayoutVerdict::OutOfBounds;
    }
    let (x, y) = match anchor {
        Anchor::TopLeft => (0, 0),
        Anchor::TopRight => (window_w - panel_w, 0),
        Anchor::BottomLeft => (0, window_h - panel_h),
        Anchor::BottomRight => (window_w - panel_w, window_h - panel_h),
        Anchor::Center => ((window_w - panel_w) / 2, (window_h - panel_h) / 2),
    };
    LayoutVerdict::Ok { x, y }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_window_layout_anchor")?;

    println!("top-left: {:?}", position(80, 24, 20, 5, Anchor::TopLeft));
    println!("center: {:?}", position(80, 24, 20, 5, Anchor::Center));
    println!(
        "bottom-right: {:?}",
        position(80, 24, 20, 5, Anchor::BottomRight)
    );
    println!("oob: {:?}", position(10, 10, 20, 5, Anchor::TopLeft));
    println!("invalid: {:?}", position(0, 24, 20, 5, Anchor::TopLeft));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positioner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn top_left_origin() {
        let v = position(80, 24, 20, 5, Anchor::TopLeft);
        assert_eq!(v, LayoutVerdict::Ok { x: 0, y: 0 });
    }

    #[test]
    fn top_right_x_calculation() {
        let v = position(80, 24, 20, 5, Anchor::TopRight);
        assert_eq!(v, LayoutVerdict::Ok { x: 60, y: 0 });
    }

    #[test]
    fn bottom_left_y_calculation() {
        let v = position(80, 24, 20, 5, Anchor::BottomLeft);
        assert_eq!(v, LayoutVerdict::Ok { x: 0, y: 19 });
    }

    #[test]
    fn bottom_right_both_calculated() {
        let v = position(80, 24, 20, 5, Anchor::BottomRight);
        assert_eq!(v, LayoutVerdict::Ok { x: 60, y: 19 });
    }

    #[test]
    fn center_calculation() {
        let v = position(80, 24, 20, 5, Anchor::Center);
        assert_eq!(v, LayoutVerdict::Ok { x: 30, y: 9 });
    }

    #[test]
    fn invalid_zero_window_w() {
        assert_eq!(
            position(0, 24, 20, 5, Anchor::TopLeft),
            LayoutVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_window_h() {
        assert_eq!(
            position(80, 0, 20, 5, Anchor::TopLeft),
            LayoutVerdict::InvalidConfig
        );
    }

    #[test]
    fn out_of_bounds_panel_too_wide() {
        assert_eq!(
            position(10, 10, 20, 5, Anchor::TopLeft),
            LayoutVerdict::OutOfBounds
        );
    }

    #[test]
    fn out_of_bounds_panel_too_tall() {
        assert_eq!(
            position(20, 5, 10, 10, Anchor::TopLeft),
            LayoutVerdict::OutOfBounds
        );
    }

    #[test]
    fn deterministic() {
        let r1 = position(80, 24, 20, 5, Anchor::Center);
        let r2 = position(80, 24, 20, 5, Anchor::Center);
        assert_eq!(r1, r2);
    }

    #[test]
    fn exact_fit_panel_at_origin() {
        let v = position(20, 5, 20, 5, Anchor::TopLeft);
        assert_eq!(v, LayoutVerdict::Ok { x: 0, y: 0 });
    }

    #[test]
    fn invalid_zero_panel_size() {
        assert_eq!(
            position(80, 24, 0, 5, Anchor::TopLeft),
            LayoutVerdict::InvalidConfig
        );
    }
}
