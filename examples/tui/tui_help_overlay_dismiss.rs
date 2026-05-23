//! # TUI Help Overlay Dismiss
//!
//! State machine for help overlay: Hidden → Visible (via `?` key)
//! → Hidden (via Escape or `?`). Returns next state given current
//! state + key event.
//!
//! Demonstrates the **TUI.123** recipe for PMAT-200 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim `:help` toggle; tmux `?` keybinding for help.
//!
//! Run with: cargo run --example tui_help_overlay_dismiss
//!
//! Added by PMAT-200 (catalog 1423→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum OverlayState {
    Hidden,
    Visible,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum KeyEvent {
    QuestionMark,
    Escape,
    Other,
}

#[derive(Debug, PartialEq)]
pub enum OverlayVerdict {
    Ok {
        new_state: OverlayState,
        consumed: bool,
    },
    InvalidConfig,
}

pub fn handle(current: OverlayState, key: KeyEvent) -> OverlayVerdict {
    let (new_state, consumed) = match (current, key) {
        (OverlayState::Hidden, KeyEvent::QuestionMark) => (OverlayState::Visible, true),
        (OverlayState::Visible, KeyEvent::QuestionMark) => (OverlayState::Hidden, true),
        (OverlayState::Visible, KeyEvent::Escape) => (OverlayState::Hidden, true),
        (OverlayState::Hidden, KeyEvent::Escape) => (OverlayState::Hidden, false),
        (state, KeyEvent::Other) => (state, false),
    };
    OverlayVerdict::Ok {
        new_state,
        consumed,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_help_overlay_dismiss")?;

    println!(
        "hidden + ?: {:?}",
        handle(OverlayState::Hidden, KeyEvent::QuestionMark)
    );
    println!(
        "visible + Esc: {:?}",
        handle(OverlayState::Visible, KeyEvent::Escape)
    );
    println!(
        "visible + ?: {:?}",
        handle(OverlayState::Visible, KeyEvent::QuestionMark)
    );
    println!(
        "hidden + Other: {:?}",
        handle(OverlayState::Hidden, KeyEvent::Other)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handler_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn hidden_question_mark_shows() {
        let v = handle(OverlayState::Hidden, KeyEvent::QuestionMark);
        if let OverlayVerdict::Ok {
            new_state,
            consumed,
        } = v
        {
            assert_eq!(new_state, OverlayState::Visible);
            assert!(consumed);
        }
    }

    #[test]
    fn visible_question_mark_hides() {
        let v = handle(OverlayState::Visible, KeyEvent::QuestionMark);
        if let OverlayVerdict::Ok { new_state, .. } = v {
            assert_eq!(new_state, OverlayState::Hidden);
        }
    }

    #[test]
    fn visible_escape_hides() {
        let v = handle(OverlayState::Visible, KeyEvent::Escape);
        if let OverlayVerdict::Ok {
            new_state,
            consumed,
        } = v
        {
            assert_eq!(new_state, OverlayState::Hidden);
            assert!(consumed);
        }
    }

    #[test]
    fn hidden_escape_no_consume() {
        let v = handle(OverlayState::Hidden, KeyEvent::Escape);
        if let OverlayVerdict::Ok {
            new_state,
            consumed,
        } = v
        {
            assert_eq!(new_state, OverlayState::Hidden);
            assert!(!consumed);
        }
    }

    #[test]
    fn other_key_no_change_visible() {
        let v = handle(OverlayState::Visible, KeyEvent::Other);
        if let OverlayVerdict::Ok {
            new_state,
            consumed,
        } = v
        {
            assert_eq!(new_state, OverlayState::Visible);
            assert!(!consumed);
        }
    }

    #[test]
    fn other_key_no_change_hidden() {
        let v = handle(OverlayState::Hidden, KeyEvent::Other);
        if let OverlayVerdict::Ok {
            new_state,
            consumed,
        } = v
        {
            assert_eq!(new_state, OverlayState::Hidden);
            assert!(!consumed);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = handle(OverlayState::Hidden, KeyEvent::QuestionMark);
        let r2 = handle(OverlayState::Hidden, KeyEvent::QuestionMark);
        assert_eq!(r1, r2);
    }

    #[test]
    fn toggle_round_trip() {
        let v1 = handle(OverlayState::Hidden, KeyEvent::QuestionMark);
        if let OverlayVerdict::Ok { new_state, .. } = v1 {
            let v2 = handle(new_state, KeyEvent::QuestionMark);
            if let OverlayVerdict::Ok {
                new_state: state2, ..
            } = v2
            {
                assert_eq!(state2, OverlayState::Hidden);
            }
        }
    }

    #[test]
    fn consumed_when_state_changes() {
        let v = handle(OverlayState::Hidden, KeyEvent::QuestionMark);
        if let OverlayVerdict::Ok { consumed, .. } = v {
            assert!(consumed);
        }
    }

    #[test]
    fn other_key_consumed_false() {
        let v = handle(OverlayState::Hidden, KeyEvent::Other);
        if let OverlayVerdict::Ok { consumed, .. } = v {
            assert!(!consumed);
        }
    }

    #[test]
    fn show_hide_three_cycles() {
        let mut state = OverlayState::Hidden;
        for _ in 0..3 {
            let v = handle(state, KeyEvent::QuestionMark);
            if let OverlayVerdict::Ok { new_state, .. } = v {
                state = new_state;
            }
        }
        // 3 toggles: hidden → visible → hidden → visible.
        assert_eq!(state, OverlayState::Visible);
    }
}
