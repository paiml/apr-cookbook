//! # TUI Radio Button Select
//!
//! Render a radio-button group with one selected. Returns rendered
//! lines with `(•)` for selected and `( )` for unselected.
//!
//! Demonstrates the **TUI.179** recipe for PMAT-225 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML5 `<input type="radio">` rendering; ARIA
//!  `role="radiogroup"` semantics.
//!
//! Run with: cargo run --example tui_radio_button_select
//!
//! Added by PMAT-225 (catalog 1648→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RadioVerdict {
    Ok {
        lines: Vec<String>,
        selected_idx: u32,
    },
    InvalidConfig,
}

pub fn render(options: &[&str], selected: u32) -> RadioVerdict {
    if options.is_empty() || (selected as usize) >= options.len() {
        return RadioVerdict::InvalidConfig;
    }
    let mut lines: Vec<String> = Vec::with_capacity(options.len());
    for (i, opt) in options.iter().enumerate() {
        let glyph = if (i as u32) == selected {
            "(•)"
        } else {
            "( )"
        };
        lines.push(format!("{glyph} {opt}"));
    }
    RadioVerdict::Ok {
        lines,
        selected_idx: selected,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_radio_button_select")?;

    println!("group: {:?}", render(&["Apple", "Banana", "Cherry"], 1));
    println!("invalid: {:?}", render(&[], 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_options_rejected() {
        assert_eq!(render(&[], 0), RadioVerdict::InvalidConfig);
    }

    #[test]
    fn selected_oob_rejected() {
        assert_eq!(render(&["a"], 5), RadioVerdict::InvalidConfig);
    }

    #[test]
    fn selected_has_filled_dot() {
        let v = render(&["a"], 0);
        if let RadioVerdict::Ok { lines, .. } = v {
            assert!(lines[0].starts_with("(•)"));
        }
    }

    #[test]
    fn unselected_has_empty_paren() {
        let v = render(&["a", "b"], 0);
        if let RadioVerdict::Ok { lines, .. } = v {
            assert!(lines[1].starts_with("( )"));
        }
    }

    #[test]
    fn lines_match_options() {
        let v = render(&["a", "b", "c"], 1);
        if let RadioVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 3);
        }
    }

    #[test]
    fn selected_idx_returned() {
        let v = render(&["a", "b", "c"], 2);
        if let RadioVerdict::Ok { selected_idx, .. } = v {
            assert_eq!(selected_idx, 2);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render(&["a"], 0);
        let r2 = render(&["a"], 0);
        assert_eq!(r1, r2);
    }

    #[test]
    fn option_text_present() {
        let v = render(&["Banana"], 0);
        if let RadioVerdict::Ok { lines, .. } = v {
            assert!(lines[0].contains("Banana"));
        }
    }

    #[test]
    fn middle_selected() {
        let v = render(&["a", "b", "c"], 1);
        if let RadioVerdict::Ok { lines, .. } = v {
            assert!(lines[0].starts_with("( )"));
            assert!(lines[1].starts_with("(•)"));
            assert!(lines[2].starts_with("( )"));
        }
    }

    #[test]
    fn last_selected() {
        let v = render(&["a", "b", "c"], 2);
        if let RadioVerdict::Ok { lines, .. } = v {
            assert!(lines[2].starts_with("(•)"));
        }
    }

    #[test]
    fn unicode_option_supported() {
        let v = render(&["café", "résumé"], 0);
        if let RadioVerdict::Ok { lines, .. } = v {
            assert!(lines[0].contains("café"));
        }
    }

    #[test]
    fn many_options_handled() {
        let options: Vec<&str> = (0..30).map(|_| "opt").collect();
        let v = render(&options, 15);
        if let RadioVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 30);
            assert!(lines[15].starts_with("(•)"));
        }
    }
}
