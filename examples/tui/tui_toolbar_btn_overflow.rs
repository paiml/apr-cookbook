//! # TUI Toolbar Button Overflow
//!
//! Compute which toolbar buttons fit in the available width and which
//! must move to a "More..." overflow menu. Returns visible buttons,
//! overflow-menu items, and the chevron position.
//!
//! Demonstrates the **TUI.151** recipe for PMAT-210 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GTK GtkToolbar overflow handling; Bootstrap navbar
//!  responsive collapse pattern.
//!
//! Run with: cargo run --example tui_toolbar_btn_overflow
//!
//! Added by PMAT-210 (catalog 1513→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum OverflowVerdict {
    Ok {
        visible: Vec<String>,
        overflow: Vec<String>,
        chevron_col: u32,
    },
    InvalidConfig,
}

pub fn layout(buttons: &[(&str, u32)], total_width: u32) -> OverflowVerdict {
    if buttons.is_empty() || total_width < 5 {
        return OverflowVerdict::InvalidConfig;
    }
    // Reserve 6 chars for "[ ▶ ]" chevron when we anticipate overflow.
    let chevron_w = 5u32;
    let mut visible: Vec<String> = Vec::new();
    let mut overflow: Vec<String> = Vec::new();
    let mut used = 0u32;
    let mut all_fit = true;
    for (i, (label, w)) in buttons.iter().enumerate() {
        let remaining_buttons = buttons.len() - i;
        let need_chevron = remaining_buttons > 1 && used + w + chevron_w > total_width;
        if used + w <= total_width && !need_chevron {
            visible.push((*label).to_string());
            used += w;
        } else {
            all_fit = false;
            overflow.push((*label).to_string());
        }
    }
    let chevron_col = if all_fit { 0 } else { used };
    OverflowVerdict::Ok {
        visible,
        overflow,
        chevron_col,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_toolbar_btn_overflow")?;

    let buttons = [("Save", 6), ("Open", 6), ("Run", 5), ("Debug", 7)];
    println!("width 20: {:?}", layout(&buttons, 20));
    println!("width 50: {:?}", layout(&buttons, 50));
    println!("invalid: {:?}", layout(&[], 20));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(layout(&[], 20), OverflowVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_narrow() {
        assert_eq!(layout(&[("a", 5)], 3), OverflowVerdict::InvalidConfig);
    }

    #[test]
    fn all_fit_no_overflow() {
        let v = layout(&[("a", 5), ("b", 5)], 20);
        if let OverflowVerdict::Ok { overflow, .. } = v {
            assert!(overflow.is_empty());
        }
    }

    #[test]
    fn overflow_to_menu() {
        let v = layout(&[("a", 5), ("b", 5), ("c", 5), ("d", 5)], 13);
        if let OverflowVerdict::Ok { overflow, .. } = v {
            assert!(!overflow.is_empty());
        }
    }

    #[test]
    fn chevron_zero_when_all_fit() {
        let v = layout(&[("a", 5)], 20);
        if let OverflowVerdict::Ok { chevron_col, .. } = v {
            assert_eq!(chevron_col, 0);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = layout(&[("a", 5)], 20);
        let r2 = layout(&[("a", 5)], 20);
        assert_eq!(r1, r2);
    }

    #[test]
    fn visible_count_le_buttons() {
        let v = layout(&[("a", 5), ("b", 5)], 20);
        if let OverflowVerdict::Ok { visible, .. } = v {
            assert!(visible.len() <= 2);
        }
    }

    #[test]
    fn first_visible_when_some_fit() {
        // Width=11: first(5)+chevron(5)=10 ≤ 11 → first visible (used=5).
        // second(8) doesn't fit: 5+8=13 > 11 → overflow.
        let v = layout(&[("first", 5), ("second", 8)], 11);
        if let OverflowVerdict::Ok {
            visible, overflow, ..
        } = v
        {
            assert_eq!(visible.first(), Some(&"first".to_string()));
            assert_eq!(overflow, vec!["second".to_string()]);
        }
    }

    #[test]
    fn many_buttons_handled() {
        let buttons: Vec<(&str, u32)> = (0..30).map(|_| ("b", 5)).collect();
        let v = layout(&buttons, 100);
        assert!(matches!(v, OverflowVerdict::Ok { .. }));
    }

    #[test]
    fn unicode_label_supported() {
        let v = layout(&[("café", 5)], 20);
        if let OverflowVerdict::Ok { visible, .. } = v {
            assert_eq!(visible, vec!["café".to_string()]);
        }
    }

    #[test]
    fn last_button_fits_no_chevron_reserved() {
        // For the LAST button, we don't need chevron space → it can fit.
        let v = layout(&[("a", 5), ("b", 14)], 20);
        if let OverflowVerdict::Ok { visible, .. } = v {
            assert_eq!(visible.len(), 2);
        }
    }
}
