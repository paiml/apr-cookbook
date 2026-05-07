//! # TUI Cursor Blink Phase
//!
//! Compute cursor visibility (on/off) at a given time `t_ms` using a
//! configurable blink period and on/off duty ratio. Returns the
//! visibility flag and remaining ms until the next phase change.
//!
//! Demonstrates the **TUI.139** recipe for PMAT-206 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: VT100 cursor blink rate (530 ms standard); macOS Terminal
//!  `Insertion Point Blink` preferences.
//!
//! Run with: cargo run --example tui_cursor_blink_phase
//!
//! Added by PMAT-206 (catalog 1477→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BlinkVerdict {
    Ok { visible: bool, ms_until_change: u32 },
    InvalidConfig,
}

pub fn phase(t_ms: u32, period_ms: u32, on_ratio_pct: u32) -> BlinkVerdict {
    if period_ms == 0 || on_ratio_pct == 0 || on_ratio_pct >= 100 {
        return BlinkVerdict::InvalidConfig;
    }
    let on_dur = period_ms * on_ratio_pct / 100;
    let phase_t = t_ms % period_ms;
    if phase_t < on_dur {
        BlinkVerdict::Ok {
            visible: true,
            ms_until_change: on_dur - phase_t,
        }
    } else {
        BlinkVerdict::Ok {
            visible: false,
            ms_until_change: period_ms - phase_t,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_cursor_blink_phase")?;

    println!("at 0ms: {:?}", phase(0, 1000, 50));
    println!("at 600ms: {:?}", phase(600, 1000, 50));
    println!("invalid: {:?}", phase(0, 0, 50));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phaser_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn visible_at_phase_start() {
        let v = phase(0, 1000, 50);
        if let BlinkVerdict::Ok { visible, .. } = v {
            assert!(visible);
        }
    }

    #[test]
    fn invisible_after_on_duration() {
        let v = phase(600, 1000, 50);
        if let BlinkVerdict::Ok { visible, .. } = v {
            assert!(!visible);
        }
    }

    #[test]
    fn invalid_zero_period() {
        assert_eq!(phase(0, 0, 50), BlinkVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_ratio() {
        assert_eq!(phase(0, 1000, 0), BlinkVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_full_ratio() {
        assert_eq!(phase(0, 1000, 100), BlinkVerdict::InvalidConfig);
    }

    #[test]
    fn ms_until_change_correct_visible() {
        let v = phase(100, 1000, 50);
        if let BlinkVerdict::Ok {
            ms_until_change, ..
        } = v
        {
            // on for 500ms; t=100 → 400ms left until invisible.
            assert_eq!(ms_until_change, 400);
        }
    }

    #[test]
    fn ms_until_change_correct_invisible() {
        let v = phase(700, 1000, 50);
        if let BlinkVerdict::Ok {
            ms_until_change, ..
        } = v
        {
            // off from 500-1000; t=700 → 300ms until next on cycle.
            assert_eq!(ms_until_change, 300);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = phase(500, 1000, 50);
        let r2 = phase(500, 1000, 50);
        assert_eq!(r1, r2);
    }

    #[test]
    fn period_wraps() {
        // t=1100ms with period 1000 → phase=100ms → visible.
        let v = phase(1100, 1000, 50);
        if let BlinkVerdict::Ok { visible, .. } = v {
            assert!(visible);
        }
    }

    #[test]
    fn small_ratio_brief_visible() {
        // 10% on, 90% off
        let v = phase(0, 1000, 10);
        if let BlinkVerdict::Ok {
            visible,
            ms_until_change,
        } = v
        {
            assert!(visible);
            assert_eq!(ms_until_change, 100);
        }
    }

    #[test]
    fn large_ratio_brief_invisible() {
        // 90% on
        let v = phase(950, 1000, 90);
        if let BlinkVerdict::Ok { visible, .. } = v {
            assert!(!visible);
        }
    }

    #[test]
    fn high_t_handled() {
        let v = phase(1_000_000, 530, 50);
        assert!(matches!(v, BlinkVerdict::Ok { .. }));
    }
}
