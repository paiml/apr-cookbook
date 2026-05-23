//! # TUI Overflow Marker Render
//!
//! Render a scrollable list with `▲` (more above) and `▼` (more
//! below) overflow indicators. Returns marker visibility for top/
//! bottom plus the visible item-index range.
//!
//! Demonstrates the **TUI.111** recipe for PMAT-196 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS Cocoa NSScroller; less(1) more-above/below
//!  conventions.
//!
//! Run with: cargo run --example tui_overflow_marker_render
//!
//! Added by PMAT-196 (catalog 1387→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum OverflowVerdict {
    Ok {
        top_marker: bool,
        bottom_marker: bool,
        visible_start: u32,
        visible_end: u32,
    },
    InvalidConfig,
}

pub fn render(total_items: u32, viewport_height: u32, scroll_offset: u32) -> OverflowVerdict {
    if total_items == 0 || viewport_height == 0 {
        return OverflowVerdict::InvalidConfig;
    }
    let visible_start = scroll_offset.min(total_items);
    let visible_end = (scroll_offset + viewport_height).min(total_items);
    if visible_start >= total_items {
        return OverflowVerdict::InvalidConfig;
    }
    let top_marker = visible_start > 0;
    let bottom_marker = visible_end < total_items;
    OverflowVerdict::Ok {
        top_marker,
        bottom_marker,
        visible_start,
        visible_end,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_overflow_marker_render")?;

    println!("at top: {:?}", render(20, 5, 0));
    println!("middle: {:?}", render(20, 5, 10));
    println!("at bottom: {:?}", render(20, 5, 15));
    println!("invalid: {:?}", render(0, 5, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn at_top_no_top_marker() {
        let v = render(20, 5, 0);
        if let OverflowVerdict::Ok {
            top_marker,
            bottom_marker,
            ..
        } = v
        {
            assert!(!top_marker);
            assert!(bottom_marker);
        }
    }

    #[test]
    fn at_bottom_no_bottom_marker() {
        let v = render(20, 5, 15);
        if let OverflowVerdict::Ok {
            top_marker,
            bottom_marker,
            ..
        } = v
        {
            assert!(top_marker);
            assert!(!bottom_marker);
        }
    }

    #[test]
    fn middle_both_markers() {
        let v = render(20, 5, 10);
        if let OverflowVerdict::Ok {
            top_marker,
            bottom_marker,
            ..
        } = v
        {
            assert!(top_marker);
            assert!(bottom_marker);
        }
    }

    #[test]
    fn empty_list_rejected() {
        assert_eq!(render(0, 5, 0), OverflowVerdict::InvalidConfig);
    }

    #[test]
    fn zero_viewport_rejected() {
        assert_eq!(render(20, 0, 0), OverflowVerdict::InvalidConfig);
    }

    #[test]
    fn scroll_past_end_rejected() {
        assert_eq!(render(20, 5, 25), OverflowVerdict::InvalidConfig);
    }

    #[test]
    fn visible_range_correct() {
        let v = render(20, 5, 10);
        if let OverflowVerdict::Ok {
            visible_start,
            visible_end,
            ..
        } = v
        {
            assert_eq!(visible_start, 10);
            assert_eq!(visible_end, 15);
        }
    }

    #[test]
    fn small_list_fits_no_markers() {
        let v = render(3, 5, 0);
        if let OverflowVerdict::Ok {
            top_marker,
            bottom_marker,
            ..
        } = v
        {
            assert!(!top_marker);
            assert!(!bottom_marker);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render(20, 5, 10);
        let r2 = render(20, 5, 10);
        assert_eq!(r1, r2);
    }

    #[test]
    fn visible_end_capped() {
        let v = render(20, 5, 18);
        if let OverflowVerdict::Ok { visible_end, .. } = v {
            assert_eq!(visible_end, 20);
        }
    }

    #[test]
    fn single_item_list_works() {
        let v = render(1, 5, 0);
        if let OverflowVerdict::Ok {
            top_marker,
            bottom_marker,
            visible_end,
            ..
        } = v
        {
            assert!(!top_marker);
            assert!(!bottom_marker);
            assert_eq!(visible_end, 1);
        }
    }

    #[test]
    fn many_items_handled() {
        let v = render(1000, 10, 500);
        if let OverflowVerdict::Ok {
            top_marker,
            bottom_marker,
            ..
        } = v
        {
            assert!(top_marker);
            assert!(bottom_marker);
        }
    }
}
