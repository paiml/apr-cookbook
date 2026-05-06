//! # TUI Scroll Viewport
//!
//! Compute scroll state given total rows, viewport height, current
//! offset, and a desired anchor row. Returns the corrected offset
//! such that the anchor remains visible.
//!
//! Demonstrates the **TUI.18** recipe for PMAT-165 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ratatui ListState scroll behavior.
//!
//! Run with: cargo run --example tui_scroll_state
//!
//! Added by PMAT-165 (catalog 1108→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ScrollVerdict {
    Ok {
        offset: u32,
        first_visible: u32,
        last_visible: u32,
    },
    EmptyTable,
    InvalidViewport,
}

pub fn compute(
    total_rows: u32,
    viewport_height: u32,
    current_offset: u32,
    anchor: u32,
) -> ScrollVerdict {
    if total_rows == 0 {
        return ScrollVerdict::EmptyTable;
    }
    if viewport_height == 0 {
        return ScrollVerdict::InvalidViewport;
    }
    let max_offset = total_rows.saturating_sub(viewport_height);
    let mut offset = current_offset.min(max_offset);
    let last_visible_idx = offset + viewport_height - 1;
    if anchor < offset {
        offset = anchor;
    } else if anchor > last_visible_idx {
        offset = (anchor + 1).saturating_sub(viewport_height);
    }
    offset = offset.min(max_offset);
    let first_visible = offset;
    let last_visible = (offset + viewport_height - 1).min(total_rows - 1);
    ScrollVerdict::Ok {
        offset,
        first_visible,
        last_visible,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_scroll_state")?;

    println!("anchor visible: {:?}", compute(100, 20, 0, 5));
    println!("anchor below: {:?}", compute(100, 20, 0, 50));
    println!("anchor above: {:?}", compute(100, 20, 50, 10));
    println!("anchor at end: {:?}", compute(100, 20, 0, 99));
    println!("empty: {:?}", compute(0, 20, 0, 0));
    println!("zero viewport: {:?}", compute(100, 0, 0, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn anchor_within_viewport_no_scroll() {
        let v = compute(100, 20, 0, 5);
        if let ScrollVerdict::Ok { offset, .. } = v {
            assert_eq!(offset, 0);
        }
    }

    #[test]
    fn anchor_below_scrolls_down() {
        let v = compute(100, 20, 0, 50);
        if let ScrollVerdict::Ok { offset, .. } = v {
            assert!(offset > 0);
        }
    }

    #[test]
    fn anchor_above_scrolls_up() {
        let v = compute(100, 20, 50, 10);
        if let ScrollVerdict::Ok { offset, .. } = v {
            assert_eq!(offset, 10);
        }
    }

    #[test]
    fn anchor_at_end_scrolls_max() {
        let v = compute(100, 20, 0, 99);
        if let ScrollVerdict::Ok {
            offset,
            last_visible,
            ..
        } = v
        {
            assert_eq!(offset, 80);
            assert_eq!(last_visible, 99);
        }
    }

    #[test]
    fn empty_table_rejected() {
        assert_eq!(compute(0, 20, 0, 0), ScrollVerdict::EmptyTable);
    }

    #[test]
    fn zero_viewport_invalid() {
        assert_eq!(compute(100, 0, 0, 0), ScrollVerdict::InvalidViewport);
    }

    #[test]
    fn viewport_larger_than_total() {
        let v = compute(5, 20, 0, 0);
        if let ScrollVerdict::Ok {
            offset,
            last_visible,
            ..
        } = v
        {
            assert_eq!(offset, 0);
            assert_eq!(last_visible, 4);
        }
    }

    #[test]
    fn first_last_consistent() {
        let v = compute(100, 20, 30, 35);
        if let ScrollVerdict::Ok {
            first_visible,
            last_visible,
            ..
        } = v
        {
            assert!(last_visible - first_visible <= 19);
        }
    }

    #[test]
    fn anchor_in_range_unchanged_offset() {
        let v = compute(100, 20, 30, 40);
        if let ScrollVerdict::Ok { offset, .. } = v {
            assert_eq!(offset, 30);
        }
    }

    #[test]
    fn offset_clamped_to_max() {
        let v = compute(100, 20, 1000, 0);
        if let ScrollVerdict::Ok { offset, .. } = v {
            assert!(offset <= 80);
        }
    }

    #[test]
    fn deterministic() {
        let a = compute(100, 20, 0, 50);
        let b = compute(100, 20, 0, 50);
        assert_eq!(a, b);
    }
}
