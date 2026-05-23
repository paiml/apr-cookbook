//! # TUI FPS Meter Smooth
//!
//! Compute exponentially-smoothed frames-per-second from frame
//! intervals. Returns smoothed FPS and the raw current value.
//!
//! Demonstrates the **TUI.172** recipe for PMAT-221 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: htop CPU sample EMA filter; Chrome DevTools FPS meter
//!  smoothing.
//!
//! Run with: cargo run --example tui_fps_meter_smooth
//!
//! Added by PMAT-221 (catalog 1612→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FpsVerdict {
    Ok { smoothed_fps: u32, raw_fps: u32 },
    InvalidConfig,
}

pub fn compute(frame_interval_ms: u32, prev_smoothed_fps: u32, alpha_pct: u32) -> FpsVerdict {
    if frame_interval_ms == 0 || !(1..=99).contains(&alpha_pct) {
        return FpsVerdict::InvalidConfig;
    }
    let raw = 1000 / frame_interval_ms;
    let alpha = alpha_pct as f64 / 100.0;
    let smoothed = alpha * raw as f64 + (1.0 - alpha) * prev_smoothed_fps as f64;
    FpsVerdict::Ok {
        smoothed_fps: smoothed as u32,
        raw_fps: raw,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_fps_meter_smooth")?;

    println!("16ms (60fps): {:?}", compute(16, 60, 30));
    println!("33ms (30fps): {:?}", compute(33, 60, 30));
    println!("invalid: {:?}", compute(0, 60, 30));
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
    fn invalid_zero_interval() {
        assert_eq!(compute(0, 60, 30), FpsVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_alpha() {
        assert_eq!(compute(16, 60, 0), FpsVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_full_alpha() {
        assert_eq!(compute(16, 60, 100), FpsVerdict::InvalidConfig);
    }

    #[test]
    fn raw_fps_correct_60() {
        let v = compute(16, 60, 30);
        if let FpsVerdict::Ok { raw_fps, .. } = v {
            // 1000/16 = 62 (integer truncate)
            assert_eq!(raw_fps, 62);
        }
    }

    #[test]
    fn raw_fps_correct_30() {
        let v = compute(33, 60, 30);
        if let FpsVerdict::Ok { raw_fps, .. } = v {
            // 1000/33 = 30
            assert_eq!(raw_fps, 30);
        }
    }

    #[test]
    fn smoothed_between_raw_and_prev() {
        let v = compute(33, 60, 50);
        if let FpsVerdict::Ok {
            smoothed_fps,
            raw_fps,
        } = v
        {
            assert!(smoothed_fps >= raw_fps.min(60));
            assert!(smoothed_fps <= raw_fps.max(60));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = compute(16, 60, 30);
        let r2 = compute(16, 60, 30);
        assert_eq!(r1, r2);
    }

    #[test]
    fn high_alpha_more_responsive() {
        // alpha=80 → smoothed closer to raw than alpha=20
        let high_a = compute(33, 60, 80);
        let low_a = compute(33, 60, 20);
        if let (
            FpsVerdict::Ok {
                smoothed_fps: h, ..
            },
            FpsVerdict::Ok {
                smoothed_fps: l, ..
            },
        ) = (high_a, low_a)
        {
            // 33ms = 30fps; smoothing from 60.
            // alpha=80: 0.8*30 + 0.2*60 = 36 (closer to 30)
            // alpha=20: 0.2*30 + 0.8*60 = 54 (closer to 60)
            assert!(h < l);
        }
    }

    #[test]
    fn equal_smoothed_when_raw_eq_prev() {
        let v = compute(16, 62, 50);
        if let FpsVerdict::Ok { smoothed_fps, .. } = v {
            assert!((61..=63).contains(&smoothed_fps));
        }
    }

    #[test]
    fn min_alpha_accepted() {
        let v = compute(16, 60, 1);
        assert!(matches!(v, FpsVerdict::Ok { .. }));
    }

    #[test]
    fn long_interval_low_fps() {
        let v = compute(1000, 60, 50);
        if let FpsVerdict::Ok { raw_fps, .. } = v {
            assert_eq!(raw_fps, 1);
        }
    }

    #[test]
    fn very_short_interval_high_fps() {
        let v = compute(1, 60, 50);
        if let FpsVerdict::Ok { raw_fps, .. } = v {
            assert_eq!(raw_fps, 1000);
        }
    }
}
