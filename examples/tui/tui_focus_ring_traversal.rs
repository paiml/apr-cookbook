//! # TUI Focus Ring Traversal
//!
//! Walk through focusable widgets in DOM-order: Tab advances to
//! next, Shift+Tab returns. Skips disabled widgets. Wraps at end.
//! Returns next focused widget id.
//!
//! Demonstrates the **TUI.72** recipe for PMAT-183 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML Living Standard `tabindex` semantics; macOS Cocoa
//!  Focus Ring chapter.
//!
//! Run with: cargo run --example tui_focus_ring_traversal
//!
//! Added by PMAT-183 (catalog 1270→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Direction {
    Forward,
    Backward,
}

#[derive(Debug, PartialEq)]
pub enum FocusVerdict {
    Ok { next_id: String },
    NoFocusable,
    InvalidConfig,
}

pub fn next_focus(
    widgets: &[(&str, bool)], // (id, enabled)
    current_id: &str,
    direction: Direction,
) -> FocusVerdict {
    if widgets.is_empty() {
        return FocusVerdict::InvalidConfig;
    }
    let enabled_indices: Vec<usize> = widgets
        .iter()
        .enumerate()
        .filter_map(|(i, (_, e))| if *e { Some(i) } else { None })
        .collect();
    if enabled_indices.is_empty() {
        return FocusVerdict::NoFocusable;
    }
    let current_idx = widgets.iter().position(|(id, _)| *id == current_id);
    let next_idx = if let Some(idx) = current_idx {
        let pos = enabled_indices.iter().position(|i| *i == idx);
        if let Some(p) = pos {
            // Advance/retreat in enabled-indices.
            let new_p = match direction {
                Direction::Forward => (p + 1) % enabled_indices.len(),
                Direction::Backward => {
                    if p == 0 {
                        enabled_indices.len() - 1
                    } else {
                        p - 1
                    }
                }
            };
            enabled_indices[new_p]
        } else {
            // Current is disabled; jump to first/last enabled.
            match direction {
                Direction::Forward => enabled_indices[0],
                Direction::Backward => enabled_indices[enabled_indices.len() - 1],
            }
        }
    } else {
        // current_id not in list → start from beginning/end.
        match direction {
            Direction::Forward => enabled_indices[0],
            Direction::Backward => enabled_indices[enabled_indices.len() - 1],
        }
    };
    FocusVerdict::Ok {
        next_id: widgets[next_idx].0.to_string(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_focus_ring_traversal")?;

    let widgets = [
        ("input1", true),
        ("button_disabled", false),
        ("input2", true),
        ("submit", true),
    ];
    println!(
        "forward: {:?}",
        next_focus(&widgets, "input1", Direction::Forward)
    );
    println!(
        "wrap: {:?}",
        next_focus(&widgets, "submit", Direction::Forward)
    );
    println!(
        "backward: {:?}",
        next_focus(&widgets, "input1", Direction::Backward)
    );
    println!("invalid: {:?}", next_focus(&[], "x", Direction::Forward));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn focus_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn forward_advances_to_next_enabled() {
        let w = [("a", true), ("b", true)];
        let v = next_focus(&w, "a", Direction::Forward);
        if let FocusVerdict::Ok { next_id } = v {
            assert_eq!(next_id, "b");
        }
    }

    #[test]
    fn forward_skips_disabled() {
        let w = [("a", true), ("b", false), ("c", true)];
        let v = next_focus(&w, "a", Direction::Forward);
        if let FocusVerdict::Ok { next_id } = v {
            assert_eq!(next_id, "c");
        }
    }

    #[test]
    fn forward_wraps_at_end() {
        let w = [("a", true), ("b", true)];
        let v = next_focus(&w, "b", Direction::Forward);
        if let FocusVerdict::Ok { next_id } = v {
            assert_eq!(next_id, "a");
        }
    }

    #[test]
    fn backward_retreats() {
        let w = [("a", true), ("b", true)];
        let v = next_focus(&w, "b", Direction::Backward);
        if let FocusVerdict::Ok { next_id } = v {
            assert_eq!(next_id, "a");
        }
    }

    #[test]
    fn backward_wraps() {
        let w = [("a", true), ("b", true)];
        let v = next_focus(&w, "a", Direction::Backward);
        if let FocusVerdict::Ok { next_id } = v {
            assert_eq!(next_id, "b");
        }
    }

    #[test]
    fn empty_widgets_rejected() {
        assert_eq!(
            next_focus(&[], "x", Direction::Forward),
            FocusVerdict::InvalidConfig
        );
    }

    #[test]
    fn all_disabled_returns_no_focusable() {
        let w = [("a", false), ("b", false)];
        assert_eq!(
            next_focus(&w, "a", Direction::Forward),
            FocusVerdict::NoFocusable
        );
    }

    #[test]
    fn current_not_in_list_starts_from_first() {
        let w = [("a", true), ("b", true)];
        let v = next_focus(&w, "ghost", Direction::Forward);
        if let FocusVerdict::Ok { next_id } = v {
            assert_eq!(next_id, "a");
        }
    }

    #[test]
    fn deterministic() {
        let w = [("a", true), ("b", true)];
        let r1 = next_focus(&w, "a", Direction::Forward);
        let r2 = next_focus(&w, "a", Direction::Forward);
        assert_eq!(r1, r2);
    }

    #[test]
    fn current_disabled_jumps_to_first_enabled() {
        let w = [("a", true), ("disabled", false), ("c", true)];
        let v = next_focus(&w, "disabled", Direction::Forward);
        if let FocusVerdict::Ok { next_id } = v {
            // Expectation: jump to first enabled.
            assert_eq!(next_id, "a");
        }
    }

    #[test]
    fn single_enabled_widget_stays() {
        let w = [("only", true), ("disabled", false)];
        let v = next_focus(&w, "only", Direction::Forward);
        if let FocusVerdict::Ok { next_id } = v {
            assert_eq!(next_id, "only");
        }
    }
}
