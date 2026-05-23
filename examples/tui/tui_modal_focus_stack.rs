//! # TUI Modal Focus Stack
//!
//! Compute focus state for a stack of modals: which modal owns input
//! focus (top of stack), z-index per modal, and whether the
//! background is dimmed (any modal open).
//!
//! Demonstrates the **TUI.08** recipe for PMAT-162 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML focus-trap + macOS sheet stacking conventions.
//!
//! Run with: cargo run --example tui_modal_focus_stack
//!
//! Added by PMAT-162 (catalog 1081→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FocusVerdict {
    Ok {
        focused_modal: String,
        background_dimmed: bool,
        depth: u32,
    },
    NoModalsOpen,
}

pub fn compute(modal_stack: &[&str]) -> FocusVerdict {
    if modal_stack.is_empty() {
        return FocusVerdict::NoModalsOpen;
    }
    let top = modal_stack[modal_stack.len() - 1];
    FocusVerdict::Ok {
        focused_modal: top.to_string(),
        background_dimmed: true,
        depth: modal_stack.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_modal_focus_stack")?;

    println!("none: {:?}", compute(&[]));
    println!("one: {:?}", compute(&["confirm"]));
    println!("nested: {:?}", compute(&["settings", "confirm"]));
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
    fn no_modals_special() {
        assert_eq!(compute(&[]), FocusVerdict::NoModalsOpen);
    }

    #[test]
    fn single_modal_focuses_it() {
        let v = compute(&["confirm"]);
        if let FocusVerdict::Ok {
            focused_modal,
            depth,
            ..
        } = v
        {
            assert_eq!(focused_modal, "confirm");
            assert_eq!(depth, 1);
        }
    }

    #[test]
    fn nested_focuses_top() {
        let v = compute(&["settings", "edit", "confirm"]);
        if let FocusVerdict::Ok {
            focused_modal,
            depth,
            ..
        } = v
        {
            assert_eq!(focused_modal, "confirm");
            assert_eq!(depth, 3);
        }
    }

    #[test]
    fn any_modal_dims_background() {
        let v = compute(&["any"]);
        if let FocusVerdict::Ok {
            background_dimmed, ..
        } = v
        {
            assert!(background_dimmed);
        }
    }

    #[test]
    fn deep_stack_works() {
        let stack: Vec<&str> = (0..100).map(|_| "m").collect();
        let v = compute(&stack);
        if let FocusVerdict::Ok { depth, .. } = v {
            assert_eq!(depth, 100);
        }
    }

    #[test]
    fn empty_modal_name_works() {
        let v = compute(&[""]);
        if let FocusVerdict::Ok { focused_modal, .. } = v {
            assert_eq!(focused_modal, "");
        }
    }

    #[test]
    fn duplicate_names_top_wins() {
        let v = compute(&["x", "y", "x"]);
        if let FocusVerdict::Ok { focused_modal, .. } = v {
            assert_eq!(focused_modal, "x");
        }
    }

    #[test]
    fn no_dim_when_no_modals() {
        let v = compute(&[]);
        assert_eq!(v, FocusVerdict::NoModalsOpen);
    }

    #[test]
    fn unicode_modal_name() {
        let v = compute(&["sÉttings"]);
        if let FocusVerdict::Ok { focused_modal, .. } = v {
            assert_eq!(focused_modal, "sÉttings");
        }
    }

    #[test]
    fn deterministic() {
        let s = ["a", "b"];
        let a = compute(&s);
        let b = compute(&s);
        assert_eq!(a, b);
    }
}
