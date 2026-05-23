//! # TUI Smooth Scroll Velocity
//!
//! Apply velocity-based smooth scrolling: each tick adds the velocity
//! to position, decays velocity by friction, snapping to integer
//! lines. Returns next position, velocity, and whether motion stopped.
//!
//! Demonstrates the **TUI.160** recipe for PMAT-213 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: iOS UIScrollView momentum decay; Chrome smooth-scroll
//!  velocity model.
//!
//! Run with: cargo run --example tui_smooth_scroll_velocity
//!
//! Added by PMAT-213 (catalog 1540→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ScrollVerdict {
    Ok {
        next_pos: u32,
        next_velocity_x100: i32,
        stopped: bool,
    },
    InvalidConfig,
}

pub fn tick(pos: u32, velocity_x100: i32, friction_pct: u32, max_pos: u32) -> ScrollVerdict {
    if !(1..=99).contains(&friction_pct) {
        return ScrollVerdict::InvalidConfig;
    }
    let friction = friction_pct as f64 / 100.0;
    let v = velocity_x100 as f64 / 100.0;
    let new_pos = pos as f64 + v;
    let clamped = new_pos.max(0.0).min(max_pos as f64);
    let new_v = v * (1.0 - friction);
    let new_v_x100 = (new_v * 100.0) as i32;
    let stopped = new_v_x100.abs() < 5;
    ScrollVerdict::Ok {
        next_pos: clamped as u32,
        next_velocity_x100: if stopped { 0 } else { new_v_x100 },
        stopped,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_smooth_scroll_velocity")?;

    println!("step 1: {:?}", tick(0, 1000, 20, 1000));
    println!("step 2: {:?}", tick(10, 800, 20, 1000));
    println!("invalid: {:?}", tick(0, 100, 0, 1000));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ticker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn invalid_zero_friction() {
        assert_eq!(tick(0, 100, 0, 1000), ScrollVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_full_friction() {
        assert_eq!(tick(0, 100, 100, 1000), ScrollVerdict::InvalidConfig);
    }

    #[test]
    fn position_advances_with_velocity() {
        let v = tick(0, 1000, 20, 1000);
        if let ScrollVerdict::Ok { next_pos, .. } = v {
            assert_eq!(next_pos, 10);
        }
    }

    #[test]
    fn velocity_decays_with_friction() {
        let v = tick(0, 1000, 20, 1000);
        if let ScrollVerdict::Ok {
            next_velocity_x100, ..
        } = v
        {
            // 1000 * (1 - 0.2) = 800
            assert_eq!(next_velocity_x100, 800);
        }
    }

    #[test]
    fn stopped_when_velocity_low() {
        let v = tick(0, 1, 50, 1000);
        if let ScrollVerdict::Ok { stopped, .. } = v {
            assert!(stopped);
        }
    }

    #[test]
    fn max_pos_clamps_position() {
        let v = tick(0, 100_000, 20, 100);
        if let ScrollVerdict::Ok { next_pos, .. } = v {
            assert_eq!(next_pos, 100);
        }
    }

    #[test]
    fn negative_velocity_decreases_pos() {
        let v = tick(50, -1000, 20, 1000);
        if let ScrollVerdict::Ok { next_pos, .. } = v {
            assert_eq!(next_pos, 40);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = tick(0, 1000, 20, 1000);
        let r2 = tick(0, 1000, 20, 1000);
        assert_eq!(r1, r2);
    }

    #[test]
    fn lower_pos_clamps_to_zero() {
        let v = tick(5, -10000, 20, 1000);
        if let ScrollVerdict::Ok { next_pos, .. } = v {
            assert_eq!(next_pos, 0);
        }
    }

    #[test]
    fn high_friction_fast_decay() {
        let low_fric = tick(0, 1000, 10, 1000);
        let high_fric = tick(0, 1000, 80, 1000);
        if let (
            ScrollVerdict::Ok {
                next_velocity_x100: l,
                ..
            },
            ScrollVerdict::Ok {
                next_velocity_x100: h,
                ..
            },
        ) = (low_fric, high_fric)
        {
            assert!(h.abs() < l.abs());
        }
    }

    #[test]
    fn velocity_zero_when_stopped() {
        let v = tick(0, 1, 50, 1000);
        if let ScrollVerdict::Ok {
            next_velocity_x100,
            stopped,
            ..
        } = v
        {
            assert!(stopped);
            assert_eq!(next_velocity_x100, 0);
        }
    }

    #[test]
    fn min_friction_accepted() {
        let v = tick(0, 1000, 1, 1000);
        assert!(matches!(v, ScrollVerdict::Ok { .. }));
    }
}
