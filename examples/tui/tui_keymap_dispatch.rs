//! # TUI Keymap Dispatch
//!
//! Map a key sequence (e.g. "ctrl-c", "j", "shift-tab") to a logical
//! action. Returns Action::Unknown for unmapped keys and Reserved for
//! protected sequences (ctrl-c). Pure function: no terminal IO.
//!
//! Demonstrates the **TUI.03** recipe for PMAT-160 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim modal keybindings + GNU readline conventions.
//!
//! Run with: cargo run --example tui_keymap_dispatch
//!
//! Added by PMAT-160 (catalog 1063→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum Action {
    MoveDown,
    MoveUp,
    Select,
    Cancel,
    Quit,
    NextField,
    PrevField,
    Reserved,
    Unknown,
}

pub fn dispatch(key: &str) -> Action {
    let normalized = key.trim().to_ascii_lowercase();
    match normalized.as_str() {
        "ctrl-c" | "ctrl-d" => Action::Reserved,
        "j" | "down" => Action::MoveDown,
        "k" | "up" => Action::MoveUp,
        "enter" | "space" => Action::Select,
        "esc" | "escape" => Action::Cancel,
        "q" => Action::Quit,
        "tab" => Action::NextField,
        "shift-tab" | "btab" => Action::PrevField,
        _ => Action::Unknown,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_keymap_dispatch")?;

    println!("j: {:?}", dispatch("j"));
    println!("Enter: {:?}", dispatch("Enter"));
    println!("Ctrl-C: {:?}", dispatch("ctrl-c"));
    println!("?: {:?}", dispatch("?"));
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
    fn vim_down() {
        assert_eq!(dispatch("j"), Action::MoveDown);
    }

    #[test]
    fn arrow_down() {
        assert_eq!(dispatch("Down"), Action::MoveDown);
    }

    #[test]
    fn vim_up() {
        assert_eq!(dispatch("k"), Action::MoveUp);
    }

    #[test]
    fn enter_select() {
        assert_eq!(dispatch("Enter"), Action::Select);
    }

    #[test]
    fn space_select() {
        assert_eq!(dispatch("Space"), Action::Select);
    }

    #[test]
    fn esc_cancel() {
        assert_eq!(dispatch("Esc"), Action::Cancel);
    }

    #[test]
    fn q_quit() {
        assert_eq!(dispatch("q"), Action::Quit);
    }

    #[test]
    fn tab_next_field() {
        assert_eq!(dispatch("Tab"), Action::NextField);
    }

    #[test]
    fn shift_tab_prev_field() {
        assert_eq!(dispatch("Shift-Tab"), Action::PrevField);
    }

    #[test]
    fn ctrl_c_reserved() {
        assert_eq!(dispatch("Ctrl-C"), Action::Reserved);
    }

    #[test]
    fn ctrl_d_reserved() {
        assert_eq!(dispatch("ctrl-d"), Action::Reserved);
    }

    #[test]
    fn unknown_key() {
        assert_eq!(dispatch("?"), Action::Unknown);
    }

    #[test]
    fn whitespace_trimmed() {
        assert_eq!(dispatch("  j  "), Action::MoveDown);
    }

    #[test]
    fn empty_unknown() {
        assert_eq!(dispatch(""), Action::Unknown);
    }

    #[test]
    fn case_insensitive() {
        assert_eq!(dispatch("ENTER"), Action::Select);
    }

    #[test]
    fn deterministic() {
        let a = dispatch("j");
        let b = dispatch("j");
        assert_eq!(a, b);
    }
}
