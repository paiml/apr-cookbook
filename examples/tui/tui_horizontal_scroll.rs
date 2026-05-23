//! # TUI Horizontal Scroll
//!
//! Compute horizontal scroll offset for wide content (e.g. log lines
//! wider than viewport). Scroll-step jumps by half-viewport for
//! readability; clamps to 0..max_offset.
//!
//! Demonstrates the **TUI.22** recipe for PMAT-167 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: less(1) horizontal scroll behavior.
//!
//! Run with: cargo run --example tui_horizontal_scroll
//!
//! Added by PMAT-167 (catalog 1126→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HScrollOp {
    Left,
    Right,
    Home,
    End,
}

#[derive(Debug, PartialEq)]
pub enum HScrollVerdict {
    Ok { offset: u32 },
    InvalidViewport,
}

pub fn apply(
    current_offset: u32,
    content_width: u32,
    viewport_width: u32,
    op: HScrollOp,
) -> HScrollVerdict {
    if viewport_width == 0 {
        return HScrollVerdict::InvalidViewport;
    }
    let max_offset = content_width.saturating_sub(viewport_width);
    let step = (viewport_width / 2).max(1);
    let new_offset = match op {
        HScrollOp::Left => current_offset.saturating_sub(step),
        HScrollOp::Right => current_offset.saturating_add(step).min(max_offset),
        HScrollOp::Home => 0,
        HScrollOp::End => max_offset,
    };
    HScrollVerdict::Ok {
        offset: new_offset.min(max_offset),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_horizontal_scroll")?;

    println!("right: {:?}", apply(0, 200, 80, HScrollOp::Right));
    println!("end: {:?}", apply(0, 200, 80, HScrollOp::End));
    println!("home: {:?}", apply(50, 200, 80, HScrollOp::Home));
    println!("left at start: {:?}", apply(0, 200, 80, HScrollOp::Left));
    println!("invalid: {:?}", apply(0, 200, 0, HScrollOp::Right));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scroller_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn right_advances_by_half_viewport() {
        let v = apply(0, 200, 80, HScrollOp::Right);
        if let HScrollVerdict::Ok { offset } = v {
            assert_eq!(offset, 40);
        }
    }

    #[test]
    fn left_at_start_clamps_zero() {
        let v = apply(0, 200, 80, HScrollOp::Left);
        if let HScrollVerdict::Ok { offset } = v {
            assert_eq!(offset, 0);
        }
    }

    #[test]
    fn end_jumps_to_max() {
        let v = apply(0, 200, 80, HScrollOp::End);
        if let HScrollVerdict::Ok { offset } = v {
            assert_eq!(offset, 120);
        }
    }

    #[test]
    fn home_jumps_to_zero() {
        let v = apply(50, 200, 80, HScrollOp::Home);
        if let HScrollVerdict::Ok { offset } = v {
            assert_eq!(offset, 0);
        }
    }

    #[test]
    fn right_at_end_clamps_max() {
        let v = apply(120, 200, 80, HScrollOp::Right);
        if let HScrollVerdict::Ok { offset } = v {
            assert_eq!(offset, 120);
        }
    }

    #[test]
    fn invalid_zero_viewport() {
        assert_eq!(
            apply(0, 200, 0, HScrollOp::Right),
            HScrollVerdict::InvalidViewport
        );
    }

    #[test]
    fn content_smaller_than_viewport_no_scroll() {
        let v = apply(0, 50, 80, HScrollOp::Right);
        if let HScrollVerdict::Ok { offset } = v {
            assert_eq!(offset, 0);
        }
    }

    #[test]
    fn left_in_middle_steps_back() {
        let v = apply(40, 200, 80, HScrollOp::Left);
        if let HScrollVerdict::Ok { offset } = v {
            assert_eq!(offset, 0);
        }
    }

    #[test]
    fn current_offset_clamped_to_max() {
        // Even bogus current offset clamps to max.
        let v = apply(1000, 200, 80, HScrollOp::Right);
        if let HScrollVerdict::Ok { offset } = v {
            assert!(offset <= 120);
        }
    }

    #[test]
    fn deterministic() {
        let a = apply(40, 200, 80, HScrollOp::Right);
        let b = apply(40, 200, 80, HScrollOp::Right);
        assert_eq!(a, b);
    }
}
