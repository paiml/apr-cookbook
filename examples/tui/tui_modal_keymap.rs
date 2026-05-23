//! # TUI Modal-Mode Keymap Dispatcher
//!
//! Vim-style modal editor: same key produces different actions per
//! mode (Normal / Insert / Visual / Command). Returns the action for
//! `(mode, key)` or Unbound.
//!
//! Demonstrates the **TUI.17** recipe for PMAT-165 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim modal architecture (Bram Moolenaar).
//!
//! Run with: cargo run --example tui_modal_keymap
//!
//! Added by PMAT-165 (catalog 1108→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Normal,
    Insert,
    Visual,
    Command,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    EnterInsert,
    EnterVisual,
    EnterCommand,
    EnterNormal,
    MoveDown,
    MoveUp,
    DeleteSelection,
    InsertChar,
    ExecCommand,
    Unbound,
}

#[derive(Debug, PartialEq)]
pub enum DispatchVerdict {
    Pick { action: Action },
}

pub fn dispatch(mode: Mode, key: &str) -> DispatchVerdict {
    let action = match (mode, key.trim()) {
        (Mode::Normal, "i") => Action::EnterInsert,
        (Mode::Normal, "v") => Action::EnterVisual,
        (Mode::Normal, ":") => Action::EnterCommand,
        (Mode::Normal, "j") => Action::MoveDown,
        (Mode::Normal, "k") => Action::MoveUp,
        (Mode::Insert, "Escape" | "Esc") => Action::EnterNormal,
        (Mode::Insert, _) => Action::InsertChar,
        (Mode::Visual, "d" | "x") => Action::DeleteSelection,
        (Mode::Visual, "Escape" | "Esc") => Action::EnterNormal,
        (Mode::Command, "Enter") => Action::ExecCommand,
        (Mode::Command, "Escape" | "Esc") => Action::EnterNormal,
        _ => Action::Unbound,
    };
    DispatchVerdict::Pick { action }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_modal_keymap")?;

    println!("Normal+i: {:?}", dispatch(Mode::Normal, "i"));
    println!("Normal+j: {:?}", dispatch(Mode::Normal, "j"));
    println!("Insert+a: {:?}", dispatch(Mode::Insert, "a"));
    println!("Insert+Esc: {:?}", dispatch(Mode::Insert, "Escape"));
    println!("Visual+d: {:?}", dispatch(Mode::Visual, "d"));
    println!("Command+Enter: {:?}", dispatch(Mode::Command, "Enter"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn normal_i_enter_insert() {
        let v = dispatch(Mode::Normal, "i");
        if let DispatchVerdict::Pick { action } = v {
            assert_eq!(action, Action::EnterInsert);
        }
    }

    #[test]
    fn normal_v_enter_visual() {
        let v = dispatch(Mode::Normal, "v");
        if let DispatchVerdict::Pick { action } = v {
            assert_eq!(action, Action::EnterVisual);
        }
    }

    #[test]
    fn normal_j_moves_down() {
        let v = dispatch(Mode::Normal, "j");
        if let DispatchVerdict::Pick { action } = v {
            assert_eq!(action, Action::MoveDown);
        }
    }

    #[test]
    fn insert_any_inserts() {
        let v = dispatch(Mode::Insert, "a");
        if let DispatchVerdict::Pick { action } = v {
            assert_eq!(action, Action::InsertChar);
        }
    }

    #[test]
    fn insert_esc_exits() {
        let v = dispatch(Mode::Insert, "Escape");
        if let DispatchVerdict::Pick { action } = v {
            assert_eq!(action, Action::EnterNormal);
        }
    }

    #[test]
    fn visual_d_deletes() {
        let v = dispatch(Mode::Visual, "d");
        if let DispatchVerdict::Pick { action } = v {
            assert_eq!(action, Action::DeleteSelection);
        }
    }

    #[test]
    fn visual_x_also_deletes() {
        let v = dispatch(Mode::Visual, "x");
        if let DispatchVerdict::Pick { action } = v {
            assert_eq!(action, Action::DeleteSelection);
        }
    }

    #[test]
    fn command_enter_execs() {
        let v = dispatch(Mode::Command, "Enter");
        if let DispatchVerdict::Pick { action } = v {
            assert_eq!(action, Action::ExecCommand);
        }
    }

    #[test]
    fn unknown_key_unbound() {
        let v = dispatch(Mode::Normal, "?");
        if let DispatchVerdict::Pick { action } = v {
            assert_eq!(action, Action::Unbound);
        }
    }

    #[test]
    fn whitespace_trimmed() {
        let v = dispatch(Mode::Normal, "  i  ");
        if let DispatchVerdict::Pick { action } = v {
            assert_eq!(action, Action::EnterInsert);
        }
    }

    #[test]
    fn deterministic() {
        let a = dispatch(Mode::Normal, "i");
        let b = dispatch(Mode::Normal, "i");
        assert_eq!(a, b);
    }
}
