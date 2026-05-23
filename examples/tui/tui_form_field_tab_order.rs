//! # TUI Form Field Tab Order
//!
//! Compute next form field on Tab key, skipping disabled fields.
//! Wraps to first/last at boundaries.
//!
//! Demonstrates the **TUI.106** recipe for PMAT-195 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML form field `tabindex` semantics; macOS Cocoa
//!  NSWindow makeFirstResponder.
//!
//! Run with: cargo run --example tui_form_field_tab_order
//!
//! Added by PMAT-195 (catalog 1378→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TabVerdict {
    Ok { next_index: u32, wrapped: bool },
    NoFocusable,
    InvalidConfig,
}

pub fn next_tab(
    fields: &[(&str, bool)], // (name, enabled)
    current: u32,
    forward: bool,
) -> TabVerdict {
    if fields.is_empty() {
        return TabVerdict::InvalidConfig;
    }
    if (current as usize) >= fields.len() {
        return TabVerdict::InvalidConfig;
    }
    if !fields.iter().any(|(_, e)| *e) {
        return TabVerdict::NoFocusable;
    }
    let n = fields.len() as u32;
    let mut idx = current;
    let mut steps = 0u32;
    let mut wrapped = false;
    loop {
        if forward {
            let new_idx = (idx + 1) % n;
            if new_idx == 0 && idx == n - 1 {
                wrapped = true;
            }
            idx = new_idx;
        } else {
            let new_idx = if idx == 0 { n - 1 } else { idx - 1 };
            if idx == 0 {
                wrapped = true;
            }
            idx = new_idx;
        }
        steps += 1;
        if fields[idx as usize].1 || steps >= n {
            break;
        }
    }
    TabVerdict::Ok {
        next_index: idx,
        wrapped,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_form_field_tab_order")?;

    let fields = [
        ("name", true),
        ("disabled", false),
        ("email", true),
        ("submit", true),
    ];
    println!("forward: {:?}", next_tab(&fields, 0, true));
    println!("backward: {:?}", next_tab(&fields, 0, false));
    println!("invalid: {:?}", next_tab(&[], 0, true));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tab_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn forward_advances() {
        let fields = [("a", true), ("b", true)];
        let v = next_tab(&fields, 0, true);
        if let TabVerdict::Ok { next_index, .. } = v {
            assert_eq!(next_index, 1);
        }
    }

    #[test]
    fn forward_skips_disabled() {
        let fields = [("a", true), ("dis", false), ("c", true)];
        let v = next_tab(&fields, 0, true);
        if let TabVerdict::Ok { next_index, .. } = v {
            assert_eq!(next_index, 2);
        }
    }

    #[test]
    fn forward_wraps_at_end() {
        let fields = [("a", true), ("b", true)];
        let v = next_tab(&fields, 1, true);
        if let TabVerdict::Ok {
            next_index,
            wrapped,
        } = v
        {
            assert_eq!(next_index, 0);
            assert!(wrapped);
        }
    }

    #[test]
    fn backward_retreats() {
        let fields = [("a", true), ("b", true)];
        let v = next_tab(&fields, 1, false);
        if let TabVerdict::Ok { next_index, .. } = v {
            assert_eq!(next_index, 0);
        }
    }

    #[test]
    fn empty_fields_rejected() {
        assert_eq!(next_tab(&[], 0, true), TabVerdict::InvalidConfig);
    }

    #[test]
    fn out_of_range_current_rejected() {
        let fields = [("a", true)];
        assert_eq!(next_tab(&fields, 5, true), TabVerdict::InvalidConfig);
    }

    #[test]
    fn all_disabled_no_focusable() {
        let fields = [("a", false), ("b", false)];
        assert_eq!(next_tab(&fields, 0, true), TabVerdict::NoFocusable);
    }

    #[test]
    fn deterministic() {
        let fields = [("a", true), ("b", true)];
        let r1 = next_tab(&fields, 0, true);
        let r2 = next_tab(&fields, 0, true);
        assert_eq!(r1, r2);
    }

    #[test]
    fn single_focusable_returns_self() {
        let fields = [("only", true), ("disabled", false)];
        let v = next_tab(&fields, 0, true);
        if let TabVerdict::Ok { next_index, .. } = v {
            assert_eq!(next_index, 0);
        }
    }

    #[test]
    fn backward_wraps_at_zero() {
        let fields = [("a", true), ("b", true)];
        let v = next_tab(&fields, 0, false);
        if let TabVerdict::Ok {
            next_index,
            wrapped,
        } = v
        {
            assert_eq!(next_index, 1);
            assert!(wrapped);
        }
    }

    #[test]
    fn many_fields_handled() {
        let fields: Vec<(&str, bool)> = (0..20).map(|_| ("f", true)).collect();
        let v = next_tab(&fields, 5, true);
        if let TabVerdict::Ok { next_index, .. } = v {
            assert_eq!(next_index, 6);
        }
    }
}
