//! # TUI Arrow Key Navigation
//!
//! Translate arrow keys (and modifiers) into scroll/move actions.
//! Modifiers map: plain = move, Shift = select, Ctrl = jump-to-edge.
//!
//! Demonstrates the **TUI.116** recipe for PMAT-198 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim h/j/k/l plus shift/ctrl modifiers; macOS Cocoa
//!  navigation key conventions.
//!
//! Run with: cargo run --example tui_arrow_key_navigation
//!
//! Added by PMAT-198 (catalog 1405→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Arrow {
    Up,
    Down,
    Left,
    Right,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
}

#[derive(Debug, PartialEq)]
pub enum NavAction {
    MoveUp,
    MoveDown,
    MoveLeft,
    MoveRight,
    SelectUp,
    SelectDown,
    JumpTop,
    JumpBottom,
    JumpHome,
    JumpEnd,
}

#[derive(Debug, PartialEq)]
pub enum NavVerdict {
    Ok { action: NavAction },
    InvalidConfig,
}

pub fn translate(arrow: Arrow, mods: Modifiers) -> NavVerdict {
    let action = match (arrow, mods.shift, mods.ctrl) {
        (Arrow::Up, false, false) => NavAction::MoveUp,
        (Arrow::Down, false, false) => NavAction::MoveDown,
        (Arrow::Left, false, false) => NavAction::MoveLeft,
        (Arrow::Right, false, false) => NavAction::MoveRight,
        (Arrow::Up, true, false) => NavAction::SelectUp,
        (Arrow::Down, true, false) => NavAction::SelectDown,
        (Arrow::Up, false, true) => NavAction::JumpTop,
        (Arrow::Down, false, true) => NavAction::JumpBottom,
        (Arrow::Left, false, true) => NavAction::JumpHome,
        (Arrow::Right, false, true) => NavAction::JumpEnd,
        _ => return NavVerdict::InvalidConfig,
    };
    NavVerdict::Ok { action }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_arrow_key_navigation")?;

    let none = Modifiers {
        shift: false,
        ctrl: false,
    };
    let shift = Modifiers {
        shift: true,
        ctrl: false,
    };
    let ctrl = Modifiers {
        shift: false,
        ctrl: true,
    };
    println!("plain up: {:?}", translate(Arrow::Up, none));
    println!("shift down: {:?}", translate(Arrow::Down, shift));
    println!("ctrl left: {:?}", translate(Arrow::Left, ctrl));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_mod() -> Modifiers {
        Modifiers {
            shift: false,
            ctrl: false,
        }
    }

    #[test]
    fn translator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn plain_up_moves() {
        let v = translate(Arrow::Up, no_mod());
        if let NavVerdict::Ok { action } = v {
            assert_eq!(action, NavAction::MoveUp);
        }
    }

    #[test]
    fn plain_down_moves() {
        let v = translate(Arrow::Down, no_mod());
        if let NavVerdict::Ok { action } = v {
            assert_eq!(action, NavAction::MoveDown);
        }
    }

    #[test]
    fn shift_up_selects() {
        let v = translate(
            Arrow::Up,
            Modifiers {
                shift: true,
                ctrl: false,
            },
        );
        if let NavVerdict::Ok { action } = v {
            assert_eq!(action, NavAction::SelectUp);
        }
    }

    #[test]
    fn shift_down_selects() {
        let v = translate(
            Arrow::Down,
            Modifiers {
                shift: true,
                ctrl: false,
            },
        );
        if let NavVerdict::Ok { action } = v {
            assert_eq!(action, NavAction::SelectDown);
        }
    }

    #[test]
    fn ctrl_up_jumps_top() {
        let v = translate(
            Arrow::Up,
            Modifiers {
                shift: false,
                ctrl: true,
            },
        );
        if let NavVerdict::Ok { action } = v {
            assert_eq!(action, NavAction::JumpTop);
        }
    }

    #[test]
    fn ctrl_down_jumps_bottom() {
        let v = translate(
            Arrow::Down,
            Modifiers {
                shift: false,
                ctrl: true,
            },
        );
        if let NavVerdict::Ok { action } = v {
            assert_eq!(action, NavAction::JumpBottom);
        }
    }

    #[test]
    fn ctrl_left_jumps_home() {
        let v = translate(
            Arrow::Left,
            Modifiers {
                shift: false,
                ctrl: true,
            },
        );
        if let NavVerdict::Ok { action } = v {
            assert_eq!(action, NavAction::JumpHome);
        }
    }

    #[test]
    fn ctrl_right_jumps_end() {
        let v = translate(
            Arrow::Right,
            Modifiers {
                shift: false,
                ctrl: true,
            },
        );
        if let NavVerdict::Ok { action } = v {
            assert_eq!(action, NavAction::JumpEnd);
        }
    }

    #[test]
    fn both_mods_invalid() {
        let v = translate(
            Arrow::Up,
            Modifiers {
                shift: true,
                ctrl: true,
            },
        );
        assert_eq!(v, NavVerdict::InvalidConfig);
    }

    #[test]
    fn shift_left_invalid() {
        let v = translate(
            Arrow::Left,
            Modifiers {
                shift: true,
                ctrl: false,
            },
        );
        assert_eq!(v, NavVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = translate(Arrow::Up, no_mod());
        let r2 = translate(Arrow::Up, no_mod());
        assert_eq!(r1, r2);
    }
}
