//! # TUI Popup Dialog Modal State Machine
//!
//! Manage dialog state: opened, focus index, dismissed via Esc/Enter.
//! Returns next-state with whether to redraw and whether the result
//! should be committed.
//!
//! Demonstrates the **TUI.146** recipe for PMAT-208 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GTK GtkDialog modal-state semantics; tmux confirm-before
//!  prompt state.
//!
//! Run with: cargo run --example tui_popup_dialog_modal
//!
//! Added by PMAT-208 (catalog 1495→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DialogVerdict {
    Open { focus_idx: u32 },
    Confirmed { selected_idx: u32 },
    Dismissed,
    InvalidConfig,
}

pub fn step(button_count: u32, current_focus: u32, key: &str, is_open: bool) -> DialogVerdict {
    if button_count == 0 || current_focus >= button_count {
        return DialogVerdict::InvalidConfig;
    }
    if !is_open {
        if key == "OPEN" {
            return DialogVerdict::Open { focus_idx: 0 };
        }
        return DialogVerdict::InvalidConfig;
    }
    match key {
        "Esc" => DialogVerdict::Dismissed,
        "Enter" => DialogVerdict::Confirmed {
            selected_idx: current_focus,
        },
        "Right" | "Tab" => DialogVerdict::Open {
            focus_idx: (current_focus + 1) % button_count,
        },
        "Left" | "ShiftTab" => DialogVerdict::Open {
            focus_idx: (current_focus + button_count - 1) % button_count,
        },
        _ => DialogVerdict::Open {
            focus_idx: current_focus,
        },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_popup_dialog_modal")?;

    println!("open: {:?}", step(2, 0, "OPEN", false));
    println!("right: {:?}", step(2, 0, "Right", true));
    println!("enter: {:?}", step(2, 1, "Enter", true));
    println!("esc: {:?}", step(2, 0, "Esc", true));
    println!("invalid: {:?}", step(0, 0, "Enter", true));
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
    fn open_from_closed() {
        let v = step(2, 0, "OPEN", false);
        assert_eq!(v, DialogVerdict::Open { focus_idx: 0 });
    }

    #[test]
    fn invalid_zero_buttons() {
        assert_eq!(step(0, 0, "Enter", true), DialogVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_focus_oob() {
        assert_eq!(step(2, 5, "Enter", true), DialogVerdict::InvalidConfig);
    }

    #[test]
    fn esc_dismisses() {
        let v = step(2, 0, "Esc", true);
        assert_eq!(v, DialogVerdict::Dismissed);
    }

    #[test]
    fn enter_confirms_at_focus() {
        let v = step(3, 1, "Enter", true);
        assert_eq!(v, DialogVerdict::Confirmed { selected_idx: 1 });
    }

    #[test]
    fn right_advances_focus() {
        let v = step(3, 0, "Right", true);
        assert_eq!(v, DialogVerdict::Open { focus_idx: 1 });
    }

    #[test]
    fn right_wraps_to_zero() {
        let v = step(3, 2, "Right", true);
        assert_eq!(v, DialogVerdict::Open { focus_idx: 0 });
    }

    #[test]
    fn left_decrements_focus() {
        let v = step(3, 1, "Left", true);
        assert_eq!(v, DialogVerdict::Open { focus_idx: 0 });
    }

    #[test]
    fn left_wraps_to_last() {
        let v = step(3, 0, "Left", true);
        assert_eq!(v, DialogVerdict::Open { focus_idx: 2 });
    }

    #[test]
    fn deterministic() {
        let r1 = step(2, 0, "Right", true);
        let r2 = step(2, 0, "Right", true);
        assert_eq!(r1, r2);
    }

    #[test]
    fn unknown_key_keeps_focus() {
        let v = step(2, 0, "X", true);
        assert_eq!(v, DialogVerdict::Open { focus_idx: 0 });
    }

    #[test]
    fn closed_only_open_valid() {
        let v = step(2, 0, "Enter", false);
        assert_eq!(v, DialogVerdict::InvalidConfig);
    }

    #[test]
    fn tab_advances_like_right() {
        assert_eq!(
            step(3, 0, "Tab", true),
            DialogVerdict::Open { focus_idx: 1 }
        );
    }
}
