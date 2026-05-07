//! # TUI Context Menu Show Position
//!
//! Compute where to place a context menu so it stays within the
//! viewport: prefer cursor position, but flip horizontally/vertically
//! if it would clip. Returns final (x, y) and whether each axis flipped.
//!
//! Demonstrates the **TUI.155** recipe for PMAT-211 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GTK GtkPopover edge-collision flip; W3C ARIA-1.2 popup
//!  positioning guidance.
//!
//! Run with: cargo run --example tui_context_menu_show
//!
//! Added by PMAT-211 (catalog 1522→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MenuPosVerdict {
    Ok {
        x: u32,
        y: u32,
        flipped_x: bool,
        flipped_y: bool,
    },
    InvalidConfig,
}

pub fn position(
    cursor_x: u32,
    cursor_y: u32,
    menu_w: u32,
    menu_h: u32,
    viewport_w: u32,
    viewport_h: u32,
) -> MenuPosVerdict {
    if menu_w == 0
        || menu_h == 0
        || viewport_w < menu_w
        || viewport_h < menu_h
        || cursor_x >= viewport_w
        || cursor_y >= viewport_h
    {
        return MenuPosVerdict::InvalidConfig;
    }
    let (x, flipped_x) = if cursor_x + menu_w <= viewport_w {
        (cursor_x, false)
    } else {
        (cursor_x.saturating_sub(menu_w), true)
    };
    let (y, flipped_y) = if cursor_y + menu_h <= viewport_h {
        (cursor_y, false)
    } else {
        (cursor_y.saturating_sub(menu_h), true)
    };
    MenuPosVerdict::Ok {
        x,
        y,
        flipped_x,
        flipped_y,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_context_menu_show")?;

    println!("inside: {:?}", position(10, 10, 20, 10, 100, 50));
    println!("flip-x: {:?}", position(95, 10, 20, 10, 100, 50));
    println!("flip-both: {:?}", position(95, 45, 20, 10, 100, 50));
    println!("invalid: {:?}", position(0, 0, 0, 10, 100, 50));
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
    fn invalid_zero_menu_w() {
        assert_eq!(
            position(10, 10, 0, 10, 100, 50),
            MenuPosVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_menu_h() {
        assert_eq!(
            position(10, 10, 20, 0, 100, 50),
            MenuPosVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_menu_larger_than_viewport() {
        assert_eq!(
            position(10, 10, 200, 100, 100, 50),
            MenuPosVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_cursor_oob() {
        assert_eq!(
            position(150, 10, 20, 10, 100, 50),
            MenuPosVerdict::InvalidConfig
        );
    }

    #[test]
    fn fits_no_flip() {
        let v = position(10, 10, 20, 10, 100, 50);
        assert_eq!(
            v,
            MenuPosVerdict::Ok {
                x: 10,
                y: 10,
                flipped_x: false,
                flipped_y: false,
            }
        );
    }

    #[test]
    fn flip_x_when_too_close_right() {
        let v = position(95, 10, 20, 10, 100, 50);
        if let MenuPosVerdict::Ok { flipped_x, .. } = v {
            assert!(flipped_x);
        }
    }

    #[test]
    fn flip_y_when_too_close_bottom() {
        let v = position(10, 45, 20, 10, 100, 50);
        if let MenuPosVerdict::Ok { flipped_y, .. } = v {
            assert!(flipped_y);
        }
    }

    #[test]
    fn flip_both_axes() {
        let v = position(95, 45, 20, 10, 100, 50);
        if let MenuPosVerdict::Ok {
            flipped_x,
            flipped_y,
            ..
        } = v
        {
            assert!(flipped_x);
            assert!(flipped_y);
        }
    }

    #[test]
    fn flipped_x_within_viewport() {
        let v = position(95, 10, 20, 10, 100, 50);
        if let MenuPosVerdict::Ok { x, .. } = v {
            assert!(x + 20 <= 100);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = position(10, 10, 20, 10, 100, 50);
        let r2 = position(10, 10, 20, 10, 100, 50);
        assert_eq!(r1, r2);
    }

    #[test]
    fn boundary_exact_fit_no_flip() {
        // cursor + menu = viewport edge → no flip needed.
        let v = position(80, 10, 20, 10, 100, 50);
        if let MenuPosVerdict::Ok { flipped_x, .. } = v {
            assert!(!flipped_x);
        }
    }

    #[test]
    fn corner_origin_no_flip() {
        let v = position(0, 0, 20, 10, 100, 50);
        if let MenuPosVerdict::Ok {
            x,
            y,
            flipped_x,
            flipped_y,
        } = v
        {
            assert_eq!((x, y), (0, 0));
            assert!(!flipped_x);
            assert!(!flipped_y);
        }
    }
}
