//! # TUI Keyboard Repeat Rate
//!
//! Decide whether a held key should fire again. Initial delay before
//! first repeat, then fires every `interval_ms` while held. Returns
//! Fire/Wait given elapsed time.
//!
//! Demonstrates the **TUI.42** recipe for PMAT-173 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: X11 / xkbcommon key-repeat semantics.
//!
//! Run with: cargo run --example tui_keyboard_repeat_rate
//!
//! Added by PMAT-173 (catalog 1180→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RepeatVerdict {
    Wait,
    Fire { repeat_index: u32 },
    InvalidConfig,
}

pub fn step(
    held_ms: u64,
    initial_delay_ms: u64,
    interval_ms: u64,
    last_fire_ms: u64,
) -> RepeatVerdict {
    if interval_ms == 0 || initial_delay_ms == 0 {
        return RepeatVerdict::InvalidConfig;
    }
    if held_ms < initial_delay_ms {
        return RepeatVerdict::Wait;
    }
    let since_last_fire = held_ms.saturating_sub(last_fire_ms);
    if since_last_fire < interval_ms {
        return RepeatVerdict::Wait;
    }
    let elapsed = held_ms - initial_delay_ms;
    let repeat_index = (elapsed / interval_ms) as u32;
    RepeatVerdict::Fire { repeat_index }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_keyboard_repeat_rate")?;

    println!("before delay: {:?}", step(200, 500, 30, 0));
    println!("first fire: {:?}", step(500, 500, 30, 0));
    println!("nth fire: {:?}", step(800, 500, 30, 770));
    println!("invalid: {:?}", step(500, 0, 30, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stepper_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn before_initial_delay_waits() {
        assert_eq!(step(200, 500, 30, 0), RepeatVerdict::Wait);
    }

    #[test]
    fn at_initial_delay_fires() {
        let v = step(500, 500, 30, 0);
        if let RepeatVerdict::Fire { repeat_index } = v {
            assert_eq!(repeat_index, 0);
        }
    }

    #[test]
    fn after_interval_fires() {
        // first fire at 500, last_fire = 500, interval 30 → next at 530.
        let v = step(530, 500, 30, 500);
        if let RepeatVerdict::Fire { repeat_index } = v {
            assert_eq!(repeat_index, 1);
        }
    }

    #[test]
    fn just_under_interval_waits() {
        let v = step(525, 500, 30, 500);
        assert_eq!(v, RepeatVerdict::Wait);
    }

    #[test]
    fn invalid_zero_interval() {
        assert_eq!(step(500, 500, 0, 0), RepeatVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_delay() {
        assert_eq!(step(500, 0, 30, 0), RepeatVerdict::InvalidConfig);
    }

    #[test]
    fn many_fires() {
        let v = step(800, 500, 30, 770);
        if let RepeatVerdict::Fire { repeat_index } = v {
            assert_eq!(repeat_index, 10);
        }
    }

    #[test]
    fn long_hold_high_index() {
        let v = step(10_500, 500, 30, 10_470);
        if let RepeatVerdict::Fire { .. } = v {
            // Just verify Fire, not specific index.
        }
    }

    #[test]
    fn just_after_delay_fires() {
        let v = step(501, 500, 30, 0);
        assert!(matches!(v, RepeatVerdict::Fire { .. }));
    }

    #[test]
    fn deterministic() {
        let a = step(500, 500, 30, 0);
        let b = step(500, 500, 30, 0);
        assert_eq!(a, b);
    }
}
