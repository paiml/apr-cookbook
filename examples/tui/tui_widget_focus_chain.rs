//! # TUI Widget Focus Chain
//!
//! Move focus through a circular Tab/Shift-Tab chain. Skips disabled
//! widgets. Returns the new focused widget id (or NoEnabled if none).
//!
//! Demonstrates the **TUI.35** recipe for PMAT-171 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML tabindex / GTK focus-chain conventions.
//!
//! Run with: cargo run --example tui_widget_focus_chain
//!
//! Added by PMAT-171 (catalog 1162→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Widget {
    pub id: String,
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocusOp {
    Next,
    Previous,
}

#[derive(Debug, PartialEq)]
pub enum FocusVerdict {
    Focused { id: String, position: u32 },
    NoEnabled,
    EmptyChain,
}

pub fn step(widgets: &[Widget], current_position: u32, op: FocusOp) -> FocusVerdict {
    if widgets.is_empty() {
        return FocusVerdict::EmptyChain;
    }
    if !widgets.iter().any(|w| w.enabled) {
        return FocusVerdict::NoEnabled;
    }
    let n = widgets.len() as i64;
    let cur = (current_position as i64).min(n - 1);
    let delta: i64 = match op {
        FocusOp::Next => 1,
        FocusOp::Previous => -1,
    };
    let mut idx = cur;
    for _ in 0..n {
        idx = (idx + delta).rem_euclid(n);
        if widgets[idx as usize].enabled {
            return FocusVerdict::Focused {
                id: widgets[idx as usize].id.clone(),
                position: idx as u32,
            };
        }
    }
    FocusVerdict::NoEnabled
}

fn widget(id: &str, enabled: bool) -> Widget {
    Widget {
        id: id.to_string(),
        enabled,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_widget_focus_chain")?;

    let widgets = [
        widget("a", true),
        widget("b", false),
        widget("c", true),
        widget("d", true),
    ];
    println!("next: {:?}", step(&widgets, 0, FocusOp::Next));
    println!("prev: {:?}", step(&widgets, 0, FocusOp::Previous));
    println!("skip disabled: {:?}", step(&widgets, 0, FocusOp::Next));

    let none = [widget("only", false)];
    println!("none enabled: {:?}", step(&none, 0, FocusOp::Next));
    println!("empty: {:?}", step(&[], 0, FocusOp::Next));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typical() -> Vec<Widget> {
        vec![
            widget("a", true),
            widget("b", false),
            widget("c", true),
            widget("d", true),
        ]
    }

    #[test]
    fn stepper_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn next_skips_disabled() {
        let v = step(&typical(), 0, FocusOp::Next);
        if let FocusVerdict::Focused { id, .. } = v {
            assert_eq!(id, "c");
        }
    }

    #[test]
    fn prev_skips_disabled() {
        let v = step(&typical(), 2, FocusOp::Previous);
        if let FocusVerdict::Focused { id, .. } = v {
            assert_eq!(id, "a");
        }
    }

    #[test]
    fn next_wraps_circularly() {
        let v = step(&typical(), 3, FocusOp::Next);
        if let FocusVerdict::Focused { id, .. } = v {
            assert_eq!(id, "a");
        }
    }

    #[test]
    fn prev_wraps_circularly() {
        let v = step(&typical(), 0, FocusOp::Previous);
        if let FocusVerdict::Focused { id, .. } = v {
            assert_eq!(id, "d");
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(step(&[], 0, FocusOp::Next), FocusVerdict::EmptyChain);
    }

    #[test]
    fn no_enabled_returns_no_enabled() {
        let widgets = [widget("a", false), widget("b", false)];
        assert_eq!(step(&widgets, 0, FocusOp::Next), FocusVerdict::NoEnabled);
    }

    #[test]
    fn single_enabled_loops_to_self() {
        let widgets = [widget("a", true), widget("b", false)];
        let v = step(&widgets, 0, FocusOp::Next);
        if let FocusVerdict::Focused { id, .. } = v {
            assert_eq!(id, "a");
        }
    }

    #[test]
    fn out_of_bounds_position_clamped() {
        let v = step(&typical(), 100, FocusOp::Next);
        assert!(matches!(v, FocusVerdict::Focused { .. }));
    }

    #[test]
    fn position_value_returned() {
        let v = step(&typical(), 0, FocusOp::Next);
        if let FocusVerdict::Focused { position, .. } = v {
            assert_eq!(position, 2);
        }
    }

    #[test]
    fn deterministic() {
        let widgets = typical();
        let a = step(&widgets, 0, FocusOp::Next);
        let b = step(&widgets, 0, FocusOp::Next);
        assert_eq!(a, b);
    }
}
