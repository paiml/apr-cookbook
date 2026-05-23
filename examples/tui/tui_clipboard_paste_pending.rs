//! # TUI Clipboard Paste Pending State
//!
//! State machine for clipboard paste: Idle → Input (CSI 2004 ~)
//! → Ready (after Esc end-marker). Returns next state given event.
//!
//! Demonstrates the **TUI.125** recipe for PMAT-201 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: xterm bracketed paste mode (DECSET 2004); vim
//!  `:set paste`.
//!
//! Run with: cargo run --example tui_clipboard_paste_pending
//!
//! Added by PMAT-201 (catalog 1432→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum PasteState {
    Idle,
    Input,
    Ready,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Event {
    StartMarker,
    EndMarker,
    Char,
    Cancel,
}

#[derive(Debug, PartialEq)]
pub enum PasteVerdict {
    Ok { new_state: PasteState },
    InvalidConfig,
}

pub fn transition(current: PasteState, event: Event) -> PasteVerdict {
    let new_state = match (current, event) {
        (PasteState::Idle, Event::StartMarker) => PasteState::Input,
        (PasteState::Input, Event::Char) => PasteState::Input,
        (PasteState::Input, Event::EndMarker) => PasteState::Ready,
        (PasteState::Input, Event::Cancel) => PasteState::Idle,
        (PasteState::Ready, _) => PasteState::Idle,
        (state, _) => state,
    };
    PasteVerdict::Ok { new_state }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_clipboard_paste_pending")?;

    println!(
        "idle + start: {:?}",
        transition(PasteState::Idle, Event::StartMarker)
    );
    println!(
        "input + end: {:?}",
        transition(PasteState::Input, Event::EndMarker)
    );
    println!(
        "ready + char: {:?}",
        transition(PasteState::Ready, Event::Char)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transitioner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn idle_start_to_input() {
        let v = transition(PasteState::Idle, Event::StartMarker);
        if let PasteVerdict::Ok { new_state } = v {
            assert_eq!(new_state, PasteState::Input);
        }
    }

    #[test]
    fn input_char_stays_input() {
        let v = transition(PasteState::Input, Event::Char);
        if let PasteVerdict::Ok { new_state } = v {
            assert_eq!(new_state, PasteState::Input);
        }
    }

    #[test]
    fn input_end_to_ready() {
        let v = transition(PasteState::Input, Event::EndMarker);
        if let PasteVerdict::Ok { new_state } = v {
            assert_eq!(new_state, PasteState::Ready);
        }
    }

    #[test]
    fn input_cancel_to_idle() {
        let v = transition(PasteState::Input, Event::Cancel);
        if let PasteVerdict::Ok { new_state } = v {
            assert_eq!(new_state, PasteState::Idle);
        }
    }

    #[test]
    fn ready_returns_to_idle() {
        let v = transition(PasteState::Ready, Event::Char);
        if let PasteVerdict::Ok { new_state } = v {
            assert_eq!(new_state, PasteState::Idle);
        }
    }

    #[test]
    fn idle_other_stays_idle() {
        let v = transition(PasteState::Idle, Event::Char);
        if let PasteVerdict::Ok { new_state } = v {
            assert_eq!(new_state, PasteState::Idle);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = transition(PasteState::Idle, Event::StartMarker);
        let r2 = transition(PasteState::Idle, Event::StartMarker);
        assert_eq!(r1, r2);
    }

    #[test]
    fn full_paste_cycle() {
        let mut state = PasteState::Idle;
        for ev in [
            Event::StartMarker,
            Event::Char,
            Event::Char,
            Event::EndMarker,
        ] {
            if let PasteVerdict::Ok { new_state } = transition(state, ev) {
                state = new_state;
            }
        }
        assert_eq!(state, PasteState::Ready);
    }

    #[test]
    fn cancel_aborts_paste() {
        let mut state = PasteState::Idle;
        for ev in [Event::StartMarker, Event::Char, Event::Cancel] {
            if let PasteVerdict::Ok { new_state } = transition(state, ev) {
                state = new_state;
            }
        }
        assert_eq!(state, PasteState::Idle);
    }

    #[test]
    fn idle_end_marker_no_change() {
        let v = transition(PasteState::Idle, Event::EndMarker);
        if let PasteVerdict::Ok { new_state } = v {
            assert_eq!(new_state, PasteState::Idle);
        }
    }

    #[test]
    fn ready_anything_to_idle() {
        for ev in [Event::Char, Event::EndMarker, Event::Cancel] {
            let v = transition(PasteState::Ready, ev);
            if let PasteVerdict::Ok { new_state } = v {
                assert_eq!(new_state, PasteState::Idle);
            }
        }
    }
}
