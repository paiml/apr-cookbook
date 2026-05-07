//! # TUI Loading Dots Animation
//!
//! Compute current loading-dots frame given elapsed time and frame
//! interval. Cycles through "", ".", "..", "..." patterns.
//!
//! Demonstrates the **TUI.166** recipe for PMAT-217 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS terminal status indicators; npm progress dots.
//!
//! Run with: cargo run --example tui_loading_dots_animation
//!
//! Added by PMAT-217 (catalog 1576→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DotsVerdict {
    Ok { dots: String, frame: u32 },
    InvalidConfig,
}

pub fn frame(elapsed_ms: u32, interval_ms: u32, label: &str) -> DotsVerdict {
    if interval_ms == 0 || label.is_empty() {
        return DotsVerdict::InvalidConfig;
    }
    let frame_idx = (elapsed_ms / interval_ms) % 4;
    let dots_str = ".".repeat(frame_idx as usize);
    let rendered = format!("{label}{dots_str}");
    DotsVerdict::Ok {
        dots: rendered,
        frame: frame_idx,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_loading_dots_animation")?;

    println!("frame 0: {:?}", frame(0, 250, "Loading"));
    println!("frame 2: {:?}", frame(500, 250, "Loading"));
    println!("invalid: {:?}", frame(0, 0, "x"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn invalid_zero_interval() {
        assert_eq!(frame(0, 0, "L"), DotsVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_empty_label() {
        assert_eq!(frame(0, 250, ""), DotsVerdict::InvalidConfig);
    }

    #[test]
    fn frame_zero_no_dots() {
        let v = frame(0, 250, "Loading");
        if let DotsVerdict::Ok { dots, frame } = v {
            assert_eq!(frame, 0);
            assert_eq!(dots, "Loading");
        }
    }

    #[test]
    fn frame_one_one_dot() {
        let v = frame(250, 250, "Loading");
        if let DotsVerdict::Ok { frame, .. } = v {
            assert_eq!(frame, 1);
        }
    }

    #[test]
    fn frame_two_two_dots() {
        let v = frame(500, 250, "Loading");
        if let DotsVerdict::Ok { dots, frame } = v {
            assert_eq!(frame, 2);
            assert_eq!(dots, "Loading..");
        }
    }

    #[test]
    fn frame_three_three_dots() {
        let v = frame(750, 250, "Loading");
        if let DotsVerdict::Ok { dots, frame } = v {
            assert_eq!(frame, 3);
            assert_eq!(dots, "Loading...");
        }
    }

    #[test]
    fn cycles_back_to_zero() {
        let v = frame(1000, 250, "Loading");
        if let DotsVerdict::Ok { frame, .. } = v {
            assert_eq!(frame, 0);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = frame(500, 250, "x");
        let r2 = frame(500, 250, "x");
        assert_eq!(r1, r2);
    }

    #[test]
    fn unicode_label_supported() {
        let v = frame(0, 250, "café");
        if let DotsVerdict::Ok { dots, .. } = v {
            assert!(dots.starts_with("café"));
        }
    }

    #[test]
    fn many_cycles_handled() {
        let v = frame(1_000_000, 250, "x");
        assert!(matches!(v, DotsVerdict::Ok { .. }));
    }

    #[test]
    fn long_interval_handled() {
        let v = frame(0, 60_000, "x");
        if let DotsVerdict::Ok { frame, .. } = v {
            assert_eq!(frame, 0);
        }
    }

    #[test]
    fn dots_match_frame_count() {
        let v = frame(750, 250, "x");
        if let DotsVerdict::Ok { dots, .. } = v {
            let dot_count = dots.matches('.').count();
            assert_eq!(dot_count, 3);
        }
    }
}
