//! # TUI Indeterminate Progress Bouncing Bar
//!
//! When total work is unknown, render a "bouncing" segment that
//! traverses the bar back and forth. Returns the segment's [start,
//! end] columns at a given tick.
//!
//! Demonstrates the **TUI.47** recipe for PMAT-175 (catalog crosses 1200).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GTK indeterminate progress widget pattern.
//!
//! Run with: cargo run --example tui_progress_indeterminate
//!
//! Added by PMAT-175 (catalog 1198→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BounceVerdict {
    Ok {
        start_col: u32,
        end_col: u32,
        direction_forward: bool,
    },
    InvalidConfig,
}

pub fn frame(width: u32, segment_width: u32, tick: u64) -> BounceVerdict {
    if width == 0 || segment_width == 0 || segment_width > width {
        return BounceVerdict::InvalidConfig;
    }
    let span = width - segment_width;
    if span == 0 {
        return BounceVerdict::Ok {
            start_col: 0,
            end_col: width,
            direction_forward: true,
        };
    }
    let cycle = u64::from(span) * 2;
    let pos = tick % cycle;
    let (start_col, direction_forward) = if pos < u64::from(span) {
        (pos as u32, true)
    } else {
        (span - (pos - u64::from(span)) as u32, false)
    };
    BounceVerdict::Ok {
        start_col,
        end_col: start_col + segment_width,
        direction_forward,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_progress_indeterminate")?;

    println!("tick 0: {:?}", frame(20, 5, 0));
    println!("tick 5: {:?}", frame(20, 5, 5));
    println!("tick 15 (return): {:?}", frame(20, 5, 15));
    println!("invalid: {:?}", frame(0, 5, 0));
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
    fn segment_width_correct() {
        let v = frame(20, 5, 0);
        if let BounceVerdict::Ok {
            start_col, end_col, ..
        } = v
        {
            assert_eq!(end_col - start_col, 5);
        }
    }

    #[test]
    fn forward_direction_then_reverse() {
        let v_fwd = frame(20, 5, 1);
        let v_back = frame(20, 5, 16);
        if let (
            BounceVerdict::Ok {
                direction_forward: f,
                ..
            },
            BounceVerdict::Ok {
                direction_forward: b,
                ..
            },
        ) = (v_fwd, v_back)
        {
            assert!(f);
            assert!(!b);
        }
    }

    #[test]
    fn end_col_within_width() {
        for tick in [0, 5, 15, 30, 100] {
            let v = frame(20, 5, tick);
            if let BounceVerdict::Ok { end_col, .. } = v {
                assert!(end_col <= 20);
            }
        }
    }

    #[test]
    fn invalid_zero_width() {
        assert_eq!(frame(0, 5, 0), BounceVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_segment_over_width() {
        assert_eq!(frame(5, 10, 0), BounceVerdict::InvalidConfig);
    }

    #[test]
    fn segment_equals_width() {
        let v = frame(10, 10, 0);
        if let BounceVerdict::Ok {
            start_col, end_col, ..
        } = v
        {
            assert_eq!(start_col, 0);
            assert_eq!(end_col, 10);
        }
    }

    #[test]
    fn cycle_completes() {
        // After 2*span ticks, should be back at start.
        let v0 = frame(20, 5, 0);
        let v_cycle = frame(20, 5, 30);
        assert_eq!(v0, v_cycle);
    }

    #[test]
    fn mid_forward_position() {
        // tick=7, span=15 → start=7.
        let v = frame(20, 5, 7);
        if let BounceVerdict::Ok { start_col, .. } = v {
            assert_eq!(start_col, 7);
        }
    }

    #[test]
    fn at_end_of_forward() {
        // span=15, tick=15 → start = span (would be 15) but pos < span is false, switching to reverse.
        let v = frame(20, 5, 15);
        if let BounceVerdict::Ok {
            direction_forward, ..
        } = v
        {
            assert!(!direction_forward);
        }
    }

    #[test]
    fn deterministic() {
        let a = frame(20, 5, 7);
        let b = frame(20, 5, 7);
        assert_eq!(a, b);
    }
}
