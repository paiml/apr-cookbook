//! # TUI Clip Region to Viewport
//!
//! Clip a (x, y, w, h) region against a viewport rect. Returns the
//! visible portion (or None if fully outside).
//!
//! Demonstrates the **TUI.21** recipe for PMAT-166 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Cohen-Sutherland clipping algorithm.
//!
//! Run with: cargo run --example tui_clip_to_visible
//!
//! Added by PMAT-166 (catalog 1117→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

#[derive(Debug, PartialEq)]
pub enum ClipVerdict {
    Visible { clipped: Rect },
    OffScreen,
    InvalidRect,
}

pub fn clip(region: Rect, viewport: Rect) -> ClipVerdict {
    if region.w == 0 || region.h == 0 || viewport.w == 0 || viewport.h == 0 {
        return ClipVerdict::InvalidRect;
    }
    let r_x2 = region.x + region.w;
    let r_y2 = region.y + region.h;
    let v_x2 = viewport.x + viewport.w;
    let v_y2 = viewport.y + viewport.h;
    if region.x >= v_x2 || r_x2 <= viewport.x || region.y >= v_y2 || r_y2 <= viewport.y {
        return ClipVerdict::OffScreen;
    }
    let cx = region.x.max(viewport.x);
    let cy = region.y.max(viewport.y);
    let cx2 = r_x2.min(v_x2);
    let cy2 = r_y2.min(v_y2);
    ClipVerdict::Visible {
        clipped: Rect {
            x: cx,
            y: cy,
            w: cx2 - cx,
            h: cy2 - cy,
        },
    }
}

fn rect(x: u32, y: u32, w: u32, h: u32) -> Rect {
    Rect { x, y, w, h }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_clip_to_visible")?;

    let viewport = rect(0, 0, 80, 24);
    println!("fully inside: {:?}", clip(rect(10, 10, 5, 5), viewport));
    println!("partial: {:?}", clip(rect(75, 20, 10, 5), viewport));
    println!("off screen: {:?}", clip(rect(100, 30, 5, 5), viewport));
    println!("invalid: {:?}", clip(rect(0, 0, 0, 5), viewport));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vp() -> Rect {
        rect(0, 0, 80, 24)
    }

    #[test]
    fn clipper_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn fully_inside_unchanged() {
        let r = rect(10, 10, 5, 5);
        if let ClipVerdict::Visible { clipped } = clip(r, vp()) {
            assert_eq!(clipped, r);
        }
    }

    #[test]
    fn partial_right_clipped() {
        let r = rect(75, 10, 10, 5);
        if let ClipVerdict::Visible { clipped } = clip(r, vp()) {
            // Width clipped from 10 to 5.
            assert_eq!(clipped.w, 5);
        }
    }

    #[test]
    fn partial_bottom_clipped() {
        let r = rect(10, 20, 5, 10);
        if let ClipVerdict::Visible { clipped } = clip(r, vp()) {
            assert_eq!(clipped.h, 4);
        }
    }

    #[test]
    fn off_screen_right() {
        let r = rect(100, 10, 5, 5);
        assert_eq!(clip(r, vp()), ClipVerdict::OffScreen);
    }

    #[test]
    fn off_screen_bottom() {
        let r = rect(10, 30, 5, 5);
        assert_eq!(clip(r, vp()), ClipVerdict::OffScreen);
    }

    #[test]
    fn at_left_edge_clipped() {
        let viewport = rect(10, 0, 80, 24);
        let r = rect(0, 0, 20, 5);
        if let ClipVerdict::Visible { clipped } = clip(r, viewport) {
            assert_eq!(clipped.x, 10);
            assert_eq!(clipped.w, 10);
        }
    }

    #[test]
    fn invalid_zero_w() {
        let r = rect(0, 0, 0, 5);
        assert_eq!(clip(r, vp()), ClipVerdict::InvalidRect);
    }

    #[test]
    fn invalid_zero_h() {
        let r = rect(0, 0, 5, 0);
        assert_eq!(clip(r, vp()), ClipVerdict::InvalidRect);
    }

    #[test]
    fn invalid_zero_viewport() {
        let r = rect(0, 0, 5, 5);
        let bad = rect(0, 0, 0, 0);
        assert_eq!(clip(r, bad), ClipVerdict::InvalidRect);
    }

    #[test]
    fn touching_edge_off_screen() {
        // r_x2 == viewport.x → fully off-screen.
        let viewport = rect(10, 0, 80, 24);
        let r = rect(0, 0, 10, 5);
        assert_eq!(clip(r, viewport), ClipVerdict::OffScreen);
    }

    #[test]
    fn deterministic() {
        let r = rect(10, 10, 5, 5);
        let a = clip(r, vp());
        let b = clip(r, vp());
        assert_eq!(a, b);
    }
}
