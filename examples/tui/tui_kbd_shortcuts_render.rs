//! # TUI Keyboard Shortcuts Render
//!
//! Render keyboard shortcut table aligned in two columns
//! `<chord>  <action>`. Returns rendered lines and max chord width
//! for grid alignment.
//!
//! Demonstrates the **TUI.129** recipe for PMAT-202 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim `:help quickref` formatting; tmux key bindings
//!  display.
//!
//! Run with: cargo run --example tui_kbd_shortcuts_render
//!
//! Added by PMAT-202 (catalog 1441→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ShortcutVerdict {
    Ok {
        lines: Vec<String>,
        max_chord_width: u32,
    },
    InvalidConfig,
}

pub fn render(shortcuts: &[(&str, &str)]) -> ShortcutVerdict {
    if shortcuts.is_empty() {
        return ShortcutVerdict::InvalidConfig;
    }
    let max_chord_width = shortcuts
        .iter()
        .map(|(c, _)| c.chars().count() as u32)
        .max()
        .unwrap_or(0);
    let mut lines: Vec<String> = Vec::with_capacity(shortcuts.len());
    for (chord, action) in shortcuts {
        let pad = max_chord_width - chord.chars().count() as u32;
        let line = format!("{chord}{}  {action}", " ".repeat(pad as usize));
        lines.push(line);
    }
    ShortcutVerdict::Ok {
        lines,
        max_chord_width,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_kbd_shortcuts_render")?;

    let shortcuts = [
        ("Ctrl+S", "Save"),
        ("Ctrl+Shift+P", "Command Palette"),
        ("Esc", "Cancel"),
    ];
    println!("rendered: {:?}", render(&shortcuts));
    println!("invalid: {:?}", render(&[]));
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
    fn lines_count_matches_shortcuts() {
        let s = [("a", "x"), ("b", "y")];
        let v = render(&s);
        if let ShortcutVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 2);
        }
    }

    #[test]
    fn chord_aligned_to_max_width() {
        let s = [("a", "x"), ("ABC", "y")];
        let v = render(&s);
        if let ShortcutVerdict::Ok {
            max_chord_width, ..
        } = v
        {
            assert_eq!(max_chord_width, 3);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(render(&[]), ShortcutVerdict::InvalidConfig);
    }

    #[test]
    fn padding_correct() {
        let s = [("a", "x"), ("longer", "y")];
        let v = render(&s);
        if let ShortcutVerdict::Ok { lines, .. } = v {
            // Action column starts at the same position in every line.
            let action_cols: Vec<usize> = lines
                .iter()
                .zip(s.iter())
                .map(|(l, (_, a))| l.find(a).unwrap_or(0))
                .collect();
            assert_eq!(action_cols[0], action_cols[1]);
        }
    }

    #[test]
    fn deterministic() {
        let s = [("a", "x")];
        let r1 = render(&s);
        let r2 = render(&s);
        assert_eq!(r1, r2);
    }

    #[test]
    fn single_shortcut_works() {
        let s = [("Esc", "Cancel")];
        let v = render(&s);
        if let ShortcutVerdict::Ok { lines, .. } = v {
            assert!(lines[0].contains("Esc"));
            assert!(lines[0].contains("Cancel"));
        }
    }

    #[test]
    fn unicode_chord_supported() {
        let s = [("⌘+S", "Save")];
        let v = render(&s);
        if let ShortcutVerdict::Ok { lines, .. } = v {
            assert!(lines[0].contains("⌘+S"));
        }
    }

    #[test]
    fn double_space_separator() {
        let s = [("a", "x")];
        let v = render(&s);
        if let ShortcutVerdict::Ok { lines, .. } = v {
            assert!(lines[0].contains("  "));
        }
    }

    #[test]
    fn many_shortcuts_handled() {
        let s: Vec<(&str, &str)> = (0..20).map(|_| ("a", "x")).collect();
        let v = render(&s);
        if let ShortcutVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 20);
        }
    }

    #[test]
    fn action_text_present() {
        let s = [("Ctrl+S", "Save")];
        let v = render(&s);
        if let ShortcutVerdict::Ok { lines, .. } = v {
            assert!(lines[0].ends_with("Save"));
        }
    }

    #[test]
    fn max_chord_width_finite() {
        let s = [("Ctrl+Shift+P", "Cmd")];
        let v = render(&s);
        if let ShortcutVerdict::Ok {
            max_chord_width, ..
        } = v
        {
            assert_eq!(max_chord_width, 12);
        }
    }
}
