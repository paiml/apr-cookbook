//! # TUI Status Pulse Dot
//!
//! Compute pulse-animation glyph at tick t. Three frames `(·, •, ●)`
//! cycling forever; returns the current glyph + frame index.
//!
//! Demonstrates the **TUI.65** recipe for PMAT-181 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ANSI vt100 cursor blink; macOS menubar pulse animation.
//!
//! Run with: cargo run --example tui_status_pulse_dot
//!
//! Added by PMAT-181 (catalog 1252→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PulseVerdict {
    Ok { glyph: char, frame: u32 },
    InvalidConfig,
}

pub fn frame(tick: u64, period: u64) -> PulseVerdict {
    if period == 0 {
        return PulseVerdict::InvalidConfig;
    }
    // 3 distinct sizes; each held for period/3 ticks (rounded up).
    let bucket = (tick / period) % 3;
    let glyph = match bucket {
        0 => '·',
        1 => '•',
        _ => '●',
    };
    PulseVerdict::Ok {
        glyph,
        frame: bucket as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_status_pulse_dot")?;

    println!("tick 0: {:?}", frame(0, 5));
    println!("tick 5: {:?}", frame(5, 5));
    println!("tick 10: {:?}", frame(10, 5));
    println!("tick 15: {:?}", frame(15, 5));
    println!("invalid: {:?}", frame(0, 0));
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
    fn frame_zero_first_glyph() {
        let v = frame(0, 5);
        if let PulseVerdict::Ok { glyph, frame } = v {
            assert_eq!(glyph, '·');
            assert_eq!(frame, 0);
        }
    }

    #[test]
    fn frame_period_advances() {
        let v = frame(5, 5);
        if let PulseVerdict::Ok { frame, .. } = v {
            assert_eq!(frame, 1);
        }
    }

    #[test]
    fn frame_two_period_third_glyph() {
        let v = frame(10, 5);
        if let PulseVerdict::Ok { glyph, .. } = v {
            assert_eq!(glyph, '●');
        }
    }

    #[test]
    fn wraps_after_three_periods() {
        let v0 = frame(0, 5);
        let v15 = frame(15, 5);
        assert_eq!(v0, v15);
    }

    #[test]
    fn invalid_zero_period() {
        assert_eq!(frame(0, 0), PulseVerdict::InvalidConfig);
    }

    #[test]
    fn glyph_one_of_three() {
        let v = frame(7, 3);
        if let PulseVerdict::Ok { glyph, .. } = v {
            assert!(['·', '•', '●'].contains(&glyph));
        }
    }

    #[test]
    fn frame_in_zero_to_two() {
        let v = frame(99, 7);
        if let PulseVerdict::Ok { frame, .. } = v {
            assert!(frame <= 2);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = frame(42, 5);
        let r2 = frame(42, 5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn within_period_holds_glyph() {
        let v0 = frame(0, 5);
        let v4 = frame(4, 5);
        assert_eq!(v0, v4);
    }

    #[test]
    fn period_one_advances_each_tick() {
        let v0 = frame(0, 1);
        let v1 = frame(1, 1);
        let v2 = frame(2, 1);
        assert!(v0 != v1);
        assert!(v1 != v2);
    }

    #[test]
    fn very_large_tick_wraps_correctly() {
        let v_max = frame(u64::MAX - 1, 1);
        assert!(matches!(v_max, PulseVerdict::Ok { .. }));
    }
}
