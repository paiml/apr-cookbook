//! # TUI Menu Render Aligned
//!
//! Render a flat menu with right-aligned shortcut hints. Returns
//! formatted lines and the column width used for the label.
//!
//! Demonstrates the **TUI.142** recipe for PMAT-207 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GTK menu accelerator alignment; Sublime Text command-
//!  palette item layout.
//!
//! Run with: cargo run --example tui_menu_render_aligned
//!
//! Added by PMAT-207 (catalog 1486→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MenuVerdict {
    Ok {
        lines: Vec<String>,
        label_col_width: u32,
    },
    InvalidConfig,
}

pub fn render(items: &[(&str, &str)], total_width: u32) -> MenuVerdict {
    if items.is_empty() || total_width < 10 {
        return MenuVerdict::InvalidConfig;
    }
    let max_label: u32 = items
        .iter()
        .map(|(label, _)| label.chars().count() as u32)
        .max()
        .unwrap_or(0);
    if max_label + 2 >= total_width {
        return MenuVerdict::InvalidConfig;
    }
    let mut lines: Vec<String> = Vec::with_capacity(items.len());
    for (label, hint) in items {
        let label_len = label.chars().count() as u32;
        let hint_len = hint.chars().count() as u32;
        let pad_total = total_width.saturating_sub(label_len + hint_len);
        let line = format!("{label}{}{hint}", " ".repeat(pad_total as usize));
        lines.push(line);
    }
    MenuVerdict::Ok {
        lines,
        label_col_width: max_label,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_menu_render_aligned")?;

    let items = [("Save", "Ctrl+S"), ("Open", "Ctrl+O"), ("Quit", "Ctrl+Q")];
    println!("menu: {:?}", render(&items, 30));
    println!("invalid: {:?}", render(&[], 30));
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
    fn empty_input_rejected() {
        assert_eq!(render(&[], 30), MenuVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_narrow_total() {
        assert_eq!(render(&[("a", "b")], 5), MenuVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_label_overflow() {
        assert_eq!(
            render(&[("longlabelname", "hint")], 12),
            MenuVerdict::InvalidConfig
        );
    }

    #[test]
    fn label_col_width_max() {
        let items = [("a", "x"), ("ABC", "y")];
        let v = render(&items, 20);
        if let MenuVerdict::Ok {
            label_col_width, ..
        } = v
        {
            assert_eq!(label_col_width, 3);
        }
    }

    #[test]
    fn line_count_matches() {
        let items = [("a", "x"), ("b", "y"), ("c", "z")];
        let v = render(&items, 20);
        if let MenuVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 3);
        }
    }

    #[test]
    fn line_contains_label_and_hint() {
        let v = render(&[("Save", "Ctrl+S")], 30);
        if let MenuVerdict::Ok { lines, .. } = v {
            assert!(lines[0].starts_with("Save"));
            assert!(lines[0].ends_with("Ctrl+S"));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render(&[("a", "b")], 20);
        let r2 = render(&[("a", "b")], 20);
        assert_eq!(r1, r2);
    }

    #[test]
    fn lines_have_correct_total_width() {
        let v = render(&[("a", "b")], 20);
        if let MenuVerdict::Ok { lines, .. } = v {
            assert_eq!(lines[0].chars().count(), 20);
        }
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<(&str, &str)> = (0..30).map(|_| ("a", "x")).collect();
        let v = render(&items, 20);
        if let MenuVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 30);
        }
    }

    #[test]
    fn unicode_label_supported() {
        let v = render(&[("café", "Cmd+S")], 20);
        if let MenuVerdict::Ok { lines, .. } = v {
            assert!(lines[0].contains("café"));
        }
    }

    #[test]
    fn min_total_width_accepted() {
        let v = render(&[("a", "b")], 10);
        assert!(matches!(v, MenuVerdict::Ok { .. }));
    }
}
