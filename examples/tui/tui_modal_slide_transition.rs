//! # TUI Modal Slide Transition
//!
//! Compute the y-offset for a modal sliding in from above. Uses a
//! cubic-ease-out interpolation from off-screen to target row over
//! `total_frames`.
//!
//! Demonstrates the **TUI.37** recipe for PMAT-172 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: cubic ease-out (Penner easing functions).
//!
//! Run with: cargo run --example tui_modal_slide_transition
//!
//! Added by PMAT-172 (catalog 1171→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SlideVerdict {
    Pre,
    InProgress { y_offset: i32, progress_pct: f64 },
    Settled { y_offset: i32 },
    InvalidConfig,
}

pub fn frame(current_frame: u32, total_frames: u32, start_y: i32, target_y: i32) -> SlideVerdict {
    if total_frames == 0 {
        return SlideVerdict::InvalidConfig;
    }
    if current_frame == 0 {
        return SlideVerdict::Pre;
    }
    if current_frame >= total_frames {
        return SlideVerdict::Settled { y_offset: target_y };
    }
    let t = f64::from(current_frame) / f64::from(total_frames);
    let eased = 1.0 - (1.0 - t).powi(3);
    let span = (target_y - start_y) as f64;
    let y = start_y as f64 + span * eased;
    SlideVerdict::InProgress {
        y_offset: y.round() as i32,
        progress_pct: eased * 100.0,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_modal_slide_transition")?;

    println!("frame 0: {:?}", frame(0, 30, -10, 5));
    println!("frame 15: {:?}", frame(15, 30, -10, 5));
    println!("frame 30: {:?}", frame(30, 30, -10, 5));
    println!("invalid: {:?}", frame(0, 0, -10, 5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn animator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn frame_zero_pre() {
        assert_eq!(frame(0, 30, -10, 5), SlideVerdict::Pre);
    }

    #[test]
    fn frame_at_end_settled() {
        let v = frame(30, 30, -10, 5);
        if let SlideVerdict::Settled { y_offset } = v {
            assert_eq!(y_offset, 5);
        }
    }

    #[test]
    fn beyond_end_settled() {
        let v = frame(100, 30, -10, 5);
        assert!(matches!(v, SlideVerdict::Settled { .. }));
    }

    #[test]
    fn middle_in_progress() {
        let v = frame(15, 30, -10, 5);
        assert!(matches!(v, SlideVerdict::InProgress { .. }));
    }

    #[test]
    fn invalid_zero_total() {
        assert_eq!(frame(0, 0, -10, 5), SlideVerdict::InvalidConfig);
    }

    #[test]
    fn ease_out_more_than_half_at_50_pct() {
        // Cubic ease-out is faster than linear early.
        let v = frame(15, 30, 0, 100);
        if let SlideVerdict::InProgress { progress_pct, .. } = v {
            assert!(progress_pct > 50.0);
        }
    }

    #[test]
    fn negative_to_positive_span() {
        let v = frame(15, 30, -10, 10);
        if let SlideVerdict::InProgress { y_offset, .. } = v {
            assert!(y_offset > -10 && y_offset <= 10);
        }
    }

    #[test]
    fn equal_start_target_no_motion() {
        let v = frame(15, 30, 5, 5);
        if let SlideVerdict::InProgress { y_offset, .. } = v {
            assert_eq!(y_offset, 5);
        }
    }

    #[test]
    fn progress_in_unit_range() {
        let v = frame(10, 30, 0, 100);
        if let SlideVerdict::InProgress { progress_pct, .. } = v {
            assert!((0.0..=100.0).contains(&progress_pct));
        }
    }

    #[test]
    fn deterministic() {
        let a = frame(15, 30, -10, 5);
        let b = frame(15, 30, -10, 5);
        assert_eq!(a, b);
    }
}
