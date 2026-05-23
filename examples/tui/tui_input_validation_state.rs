//! # TUI Input Validation State
//!
//! Track input field state machine: Pristine (untouched) → Typing
//! (during edit) → Valid / Invalid (on blur or submit). Returns
//! current state and validation message.
//!
//! Demonstrates the **TUI.95** recipe for PMAT-191 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML5 form validation pseudo-classes (:valid/:invalid);
//!  Material Design TextField state spec.
//!
//! Run with: cargo run --example tui_input_validation_state
//!
//! Added by PMAT-191 (catalog 1342→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum FieldState {
    Pristine,
    Typing,
    Valid,
    Invalid,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Event {
    Focus,
    Type,
    Blur,
    Submit,
}

#[derive(Debug, PartialEq)]
pub enum TransitionVerdict {
    Ok {
        new_state: FieldState,
        message: String,
    },
    InvalidConfig,
}

pub fn transition(current: FieldState, event: Event, is_valid: bool) -> TransitionVerdict {
    let (new_state, message) = match (current, event) {
        (FieldState::Pristine, Event::Focus) => (FieldState::Typing, "started typing".to_string()),
        (FieldState::Pristine, Event::Type) => (FieldState::Typing, "started typing".to_string()),
        (FieldState::Typing, Event::Type) => (FieldState::Typing, "still typing".to_string()),
        (_, Event::Blur | Event::Submit) => {
            if is_valid {
                (FieldState::Valid, "ok".to_string())
            } else {
                (FieldState::Invalid, "invalid input".to_string())
            }
        }
        (FieldState::Valid | FieldState::Invalid, Event::Type) => {
            (FieldState::Typing, "user editing again".to_string())
        }
        (FieldState::Valid | FieldState::Invalid, Event::Focus) => {
            (current, "refocus, no change".to_string())
        }
        (FieldState::Typing, Event::Focus) => (FieldState::Typing, "already focused".to_string()),
    };
    TransitionVerdict::Ok { new_state, message }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_input_validation_state")?;

    println!(
        "pristine + focus: {:?}",
        transition(FieldState::Pristine, Event::Focus, false)
    );
    println!(
        "typing + blur valid: {:?}",
        transition(FieldState::Typing, Event::Blur, true)
    );
    println!(
        "typing + submit invalid: {:?}",
        transition(FieldState::Typing, Event::Submit, false)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn pristine_focus_to_typing() {
        let v = transition(FieldState::Pristine, Event::Focus, false);
        if let TransitionVerdict::Ok { new_state, .. } = v {
            assert_eq!(new_state, FieldState::Typing);
        }
    }

    #[test]
    fn typing_blur_valid() {
        let v = transition(FieldState::Typing, Event::Blur, true);
        if let TransitionVerdict::Ok { new_state, .. } = v {
            assert_eq!(new_state, FieldState::Valid);
        }
    }

    #[test]
    fn typing_blur_invalid() {
        let v = transition(FieldState::Typing, Event::Blur, false);
        if let TransitionVerdict::Ok { new_state, .. } = v {
            assert_eq!(new_state, FieldState::Invalid);
        }
    }

    #[test]
    fn typing_continues() {
        let v = transition(FieldState::Typing, Event::Type, false);
        if let TransitionVerdict::Ok { new_state, .. } = v {
            assert_eq!(new_state, FieldState::Typing);
        }
    }

    #[test]
    fn valid_focus_no_change() {
        let v = transition(FieldState::Valid, Event::Focus, true);
        if let TransitionVerdict::Ok { new_state, .. } = v {
            assert_eq!(new_state, FieldState::Valid);
        }
    }

    #[test]
    fn valid_type_back_to_typing() {
        let v = transition(FieldState::Valid, Event::Type, true);
        if let TransitionVerdict::Ok { new_state, .. } = v {
            assert_eq!(new_state, FieldState::Typing);
        }
    }

    #[test]
    fn invalid_type_back_to_typing() {
        let v = transition(FieldState::Invalid, Event::Type, false);
        if let TransitionVerdict::Ok { new_state, .. } = v {
            assert_eq!(new_state, FieldState::Typing);
        }
    }

    #[test]
    fn submit_validates() {
        let v = transition(FieldState::Typing, Event::Submit, true);
        if let TransitionVerdict::Ok { new_state, .. } = v {
            assert_eq!(new_state, FieldState::Valid);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = transition(FieldState::Pristine, Event::Focus, false);
        let r2 = transition(FieldState::Pristine, Event::Focus, false);
        assert_eq!(r1, r2);
    }

    #[test]
    fn message_non_empty() {
        let v = transition(FieldState::Pristine, Event::Focus, false);
        if let TransitionVerdict::Ok { message, .. } = v {
            assert!(!message.is_empty());
        }
    }

    #[test]
    fn typing_focus_no_change() {
        let v = transition(FieldState::Typing, Event::Focus, false);
        if let TransitionVerdict::Ok { new_state, .. } = v {
            assert_eq!(new_state, FieldState::Typing);
        }
    }
}
