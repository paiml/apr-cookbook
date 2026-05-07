//! # TUI Marquee Speed Control
//!
//! Adjust marquee scroll speed up/down via key inputs (e.g., +/-).
//! Returns new tick interval clamped to `[min_ms, max_ms]`.
//!
//! Demonstrates the **TUI.118** recipe for PMAT-199 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vt100 keyboard repeat rate; mpv playback speed control
//!  conventions.
//!
//! Run with: cargo run --example tui_marquee_speed_control
//!
//! Added by PMAT-199 (catalog 1414→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum SpeedAction {
    Faster,
    Slower,
    Reset,
}

#[derive(Debug, PartialEq)]
pub enum SpeedVerdict {
    Ok { new_interval_ms: u32, clamped: bool },
    InvalidConfig,
}

pub fn adjust(
    current_ms: u32,
    action: SpeedAction,
    default_ms: u32,
    min_ms: u32,
    max_ms: u32,
) -> SpeedVerdict {
    if min_ms >= max_ms || default_ms < min_ms || default_ms > max_ms {
        return SpeedVerdict::InvalidConfig;
    }
    let raw = match action {
        SpeedAction::Faster => current_ms.saturating_sub(current_ms / 4).max(1),
        SpeedAction::Slower => current_ms.saturating_add(current_ms / 4).max(1),
        SpeedAction::Reset => default_ms,
    };
    let clamped_value = raw.clamp(min_ms, max_ms);
    SpeedVerdict::Ok {
        new_interval_ms: clamped_value,
        clamped: clamped_value != raw,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_marquee_speed_control")?;

    println!(
        "faster: {:?}",
        adjust(100, SpeedAction::Faster, 100, 10, 1000)
    );
    println!(
        "slower: {:?}",
        adjust(100, SpeedAction::Slower, 100, 10, 1000)
    );
    println!("reset: {:?}", adjust(50, SpeedAction::Reset, 100, 10, 1000));
    println!(
        "invalid: {:?}",
        adjust(100, SpeedAction::Faster, 100, 1000, 10)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjuster_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn faster_decreases_interval() {
        let v = adjust(100, SpeedAction::Faster, 100, 10, 1000);
        if let SpeedVerdict::Ok {
            new_interval_ms, ..
        } = v
        {
            assert!(new_interval_ms < 100);
        }
    }

    #[test]
    fn slower_increases_interval() {
        let v = adjust(100, SpeedAction::Slower, 100, 10, 1000);
        if let SpeedVerdict::Ok {
            new_interval_ms, ..
        } = v
        {
            assert!(new_interval_ms > 100);
        }
    }

    #[test]
    fn reset_returns_default() {
        let v = adjust(50, SpeedAction::Reset, 100, 10, 1000);
        if let SpeedVerdict::Ok {
            new_interval_ms, ..
        } = v
        {
            assert_eq!(new_interval_ms, 100);
        }
    }

    #[test]
    fn invalid_min_ge_max() {
        assert_eq!(
            adjust(100, SpeedAction::Faster, 100, 1000, 10),
            SpeedVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_default_below_min() {
        assert_eq!(
            adjust(100, SpeedAction::Faster, 5, 10, 1000),
            SpeedVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_default_above_max() {
        assert_eq!(
            adjust(100, SpeedAction::Faster, 5000, 10, 1000),
            SpeedVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let r1 = adjust(100, SpeedAction::Faster, 100, 10, 1000);
        let r2 = adjust(100, SpeedAction::Faster, 100, 10, 1000);
        assert_eq!(r1, r2);
    }

    #[test]
    fn clamped_at_min() {
        let v = adjust(15, SpeedAction::Faster, 100, 10, 1000);
        if let SpeedVerdict::Ok {
            new_interval_ms, ..
        } = v
        {
            assert!(new_interval_ms >= 10);
        }
    }

    #[test]
    fn clamped_at_max() {
        let v = adjust(900, SpeedAction::Slower, 100, 10, 1000);
        if let SpeedVerdict::Ok {
            new_interval_ms, ..
        } = v
        {
            assert!(new_interval_ms <= 1000);
        }
    }

    #[test]
    fn very_fast_clamped() {
        let v = adjust(20, SpeedAction::Faster, 100, 10, 1000);
        if let SpeedVerdict::Ok {
            new_interval_ms, ..
        } = v
        {
            assert!(new_interval_ms >= 10);
        }
    }

    #[test]
    fn reset_within_bounds() {
        let v = adjust(50, SpeedAction::Reset, 100, 10, 1000);
        if let SpeedVerdict::Ok { clamped, .. } = v {
            assert!(!clamped);
        }
    }
}
