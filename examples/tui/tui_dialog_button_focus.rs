//! # TUI Dialog Button Focus
//!
//! Tab navigation among dialog buttons (e.g. Yes / No / Cancel).
//! Returns the new focused index on Tab (forward) or Shift-Tab
//! (backward), with wrap-around.
//!
//! Demonstrates the **TUI.54** recipe for PMAT-177 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML dialog focus-trap conventions.
//!
//! Run with: cargo run --example tui_dialog_button_focus
//!
//! Added by PMAT-177 (catalog 1216→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusDir {
    Forward,
    Backward,
}

#[derive(Debug, PartialEq)]
pub enum DialogFocusVerdict {
    Ok { index: u32 },
    InvalidConfig,
}

pub fn step(current: u32, button_count: u32, dir: FocusDir) -> DialogFocusVerdict {
    if button_count == 0 {
        return DialogFocusVerdict::InvalidConfig;
    }
    let cur = current.min(button_count - 1);
    let new_idx = match dir {
        FocusDir::Forward => (cur + 1) % button_count,
        FocusDir::Backward => {
            if cur == 0 {
                button_count - 1
            } else {
                cur - 1
            }
        }
    };
    DialogFocusVerdict::Ok { index: new_idx }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_dialog_button_focus")?;

    println!("forward: {:?}", step(0, 3, FocusDir::Forward));
    println!("forward wrap: {:?}", step(2, 3, FocusDir::Forward));
    println!("backward: {:?}", step(1, 3, FocusDir::Backward));
    println!("backward wrap: {:?}", step(0, 3, FocusDir::Backward));
    println!("invalid: {:?}", step(0, 0, FocusDir::Forward));
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
    fn forward_advances() {
        let v = step(0, 3, FocusDir::Forward);
        if let DialogFocusVerdict::Ok { index } = v {
            assert_eq!(index, 1);
        }
    }

    #[test]
    fn forward_wraps_at_end() {
        let v = step(2, 3, FocusDir::Forward);
        if let DialogFocusVerdict::Ok { index } = v {
            assert_eq!(index, 0);
        }
    }

    #[test]
    fn backward_decrements() {
        let v = step(1, 3, FocusDir::Backward);
        if let DialogFocusVerdict::Ok { index } = v {
            assert_eq!(index, 0);
        }
    }

    #[test]
    fn backward_wraps_at_start() {
        let v = step(0, 3, FocusDir::Backward);
        if let DialogFocusVerdict::Ok { index } = v {
            assert_eq!(index, 2);
        }
    }

    #[test]
    fn invalid_zero_buttons() {
        assert_eq!(
            step(0, 0, FocusDir::Forward),
            DialogFocusVerdict::InvalidConfig
        );
    }

    #[test]
    fn out_of_bounds_clamps() {
        let v = step(100, 3, FocusDir::Forward);
        if let DialogFocusVerdict::Ok { index } = v {
            assert_eq!(index, 0);
        }
    }

    #[test]
    fn single_button_loops_to_self() {
        let v = step(0, 1, FocusDir::Forward);
        if let DialogFocusVerdict::Ok { index } = v {
            assert_eq!(index, 0);
        }
    }

    #[test]
    fn two_buttons_alternate() {
        let v = step(0, 2, FocusDir::Forward);
        if let DialogFocusVerdict::Ok { index } = v {
            assert_eq!(index, 1);
        }
    }

    #[test]
    fn deterministic() {
        let a = step(0, 3, FocusDir::Forward);
        let b = step(0, 3, FocusDir::Forward);
        assert_eq!(a, b);
    }

    #[test]
    fn many_buttons() {
        let v = step(0, 100, FocusDir::Forward);
        if let DialogFocusVerdict::Ok { index } = v {
            assert_eq!(index, 1);
        }
    }
}
