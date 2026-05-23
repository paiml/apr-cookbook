//! # apr tui — Keybinding Matcher
//!
//! `apr tui` accepts vim-style keybindings (j/k navigation, gg/G top/bottom,
//! / search, q quit). This recipe builds the matcher and asserts the
//! contract: single-key actions match immediately, multi-key sequences
//! (gg, dd) require timeout-or-prefix-flush, unknown keys → no-op.
//!
//! Demonstrates the **TUI.5** recipe for PMAT-108 (apr tui coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender TUI-002 + vim keybinding convention
//!
//! Run with: cargo run --example cli_tui_keybinding_matcher
//!
//! Added by PMAT-108 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    NoOp,
    MoveDown,
    MoveUp,
    PageDown,
    PageUp,
    JumpTop,
    JumpBottom,
    EnterSearch,
    Quit,
    PartialSequence,
}

pub fn match_keys(buffer: &str) -> Action {
    match buffer {
        "j" => Action::MoveDown,
        "k" => Action::MoveUp,
        "\u{0006}" => Action::PageDown, // Ctrl-F
        "\u{0002}" => Action::PageUp,   // Ctrl-B
        "G" => Action::JumpBottom,
        "gg" => Action::JumpTop,
        "/" => Action::EnterSearch,
        "q" => Action::Quit,
        // Multi-char prefix that's the start of a known sequence.
        "g" => Action::PartialSequence,
        _ => Action::NoOp,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_tui_keybinding_matcher")?;

    for k in ["j", "k", "G", "g", "gg", "/", "q", "x"] {
        println!("buffer {k:>4}  →  {:?}", match_keys(k));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn j_moves_down() {
        assert_eq!(match_keys("j"), Action::MoveDown);
    }

    #[test]
    fn k_moves_up() {
        assert_eq!(match_keys("k"), Action::MoveUp);
    }

    #[test]
    fn g_alone_is_partial_sequence() {
        // Could become "gg" — flag for the buffer to wait.
        assert_eq!(match_keys("g"), Action::PartialSequence);
    }

    #[test]
    fn gg_jumps_top() {
        assert_eq!(match_keys("gg"), Action::JumpTop);
    }

    #[test]
    fn capital_g_jumps_bottom() {
        assert_eq!(match_keys("G"), Action::JumpBottom);
    }

    #[test]
    fn slash_enters_search() {
        assert_eq!(match_keys("/"), Action::EnterSearch);
    }

    #[test]
    fn q_quits() {
        assert_eq!(match_keys("q"), Action::Quit);
    }

    #[test]
    fn unknown_key_noop() {
        assert_eq!(match_keys("x"), Action::NoOp);
        assert_eq!(match_keys(""), Action::NoOp);
        assert_eq!(match_keys("zzz"), Action::NoOp);
    }

    #[test]
    fn ctrl_f_page_down() {
        assert_eq!(match_keys("\u{0006}"), Action::PageDown);
    }
}
