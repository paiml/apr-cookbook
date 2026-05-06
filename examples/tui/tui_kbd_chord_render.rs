//! # TUI Keyboard Chord Render
//!
//! Render a keyboard shortcut chord like `<Ctrl+Shift+K>` from
//! its components. Modifiers ordered Ctrl > Alt > Shift > Cmd.
//!
//! Demonstrates the **TUI.74** recipe for PMAT-184 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: VS Code keybinding doc; tmux command-prefix conventions.
//!
//! Run with: cargo run --example tui_kbd_chord_render
//!
//! Added by PMAT-184 (catalog 1279→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ChordVerdict {
    Ok { rendered: String },
    InvalidConfig,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Modifiers {
    pub ctrl: bool,
    pub alt: bool,
    pub shift: bool,
    pub cmd: bool,
}

pub fn render(modifiers: Modifiers, key: &str) -> ChordVerdict {
    if key.is_empty() {
        return ChordVerdict::InvalidConfig;
    }
    let mut parts: Vec<&str> = Vec::new();
    if modifiers.ctrl {
        parts.push("Ctrl");
    }
    if modifiers.alt {
        parts.push("Alt");
    }
    if modifiers.shift {
        parts.push("Shift");
    }
    if modifiers.cmd {
        parts.push("Cmd");
    }
    parts.push(key);
    let rendered = format!("<{}>", parts.join("+"));
    ChordVerdict::Ok { rendered }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_kbd_chord_render")?;

    let m = Modifiers {
        ctrl: true,
        alt: false,
        shift: true,
        cmd: false,
    };
    println!("ctrl+shift+k: {:?}", render(m, "K"));
    let none = Modifiers {
        ctrl: false,
        alt: false,
        shift: false,
        cmd: false,
    };
    println!("plain key: {:?}", render(none, "Esc"));
    println!("invalid: {:?}", render(none, ""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_mod() -> Modifiers {
        Modifiers {
            ctrl: false,
            alt: false,
            shift: false,
            cmd: false,
        }
    }

    #[test]
    fn renderer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn plain_key_no_modifiers() {
        let v = render(no_mod(), "Esc");
        if let ChordVerdict::Ok { rendered } = v {
            assert_eq!(rendered, "<Esc>");
        }
    }

    #[test]
    fn ctrl_only() {
        let m = Modifiers {
            ctrl: true,
            ..no_mod()
        };
        let v = render(m, "C");
        if let ChordVerdict::Ok { rendered } = v {
            assert_eq!(rendered, "<Ctrl+C>");
        }
    }

    #[test]
    fn ordered_ctrl_alt_shift_cmd() {
        let m = Modifiers {
            ctrl: true,
            alt: true,
            shift: true,
            cmd: true,
        };
        let v = render(m, "K");
        if let ChordVerdict::Ok { rendered } = v {
            assert_eq!(rendered, "<Ctrl+Alt+Shift+Cmd+K>");
        }
    }

    #[test]
    fn empty_key_rejected() {
        assert_eq!(render(no_mod(), ""), ChordVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let m = Modifiers {
            ctrl: true,
            ..no_mod()
        };
        let r1 = render(m, "X");
        let r2 = render(m, "X");
        assert_eq!(r1, r2);
    }

    #[test]
    fn shift_only() {
        let m = Modifiers {
            shift: true,
            ..no_mod()
        };
        let v = render(m, "Tab");
        if let ChordVerdict::Ok { rendered } = v {
            assert_eq!(rendered, "<Shift+Tab>");
        }
    }

    #[test]
    fn cmd_only() {
        let m = Modifiers {
            cmd: true,
            ..no_mod()
        };
        let v = render(m, "C");
        if let ChordVerdict::Ok { rendered } = v {
            assert_eq!(rendered, "<Cmd+C>");
        }
    }

    #[test]
    fn ctrl_shift_skips_alt_and_cmd() {
        let m = Modifiers {
            ctrl: true,
            shift: true,
            ..no_mod()
        };
        let v = render(m, "K");
        if let ChordVerdict::Ok { rendered } = v {
            assert_eq!(rendered, "<Ctrl+Shift+K>");
        }
    }

    #[test]
    fn surrounded_by_angle_brackets() {
        let v = render(no_mod(), "F1");
        if let ChordVerdict::Ok { rendered } = v {
            assert!(rendered.starts_with('<'));
            assert!(rendered.ends_with('>'));
        }
    }

    #[test]
    fn multi_char_key_supported() {
        let v = render(no_mod(), "Enter");
        if let ChordVerdict::Ok { rendered } = v {
            assert_eq!(rendered, "<Enter>");
        }
    }

    #[test]
    fn plus_separator_between_parts() {
        let m = Modifiers {
            ctrl: true,
            alt: true,
            ..no_mod()
        };
        let v = render(m, "K");
        if let ChordVerdict::Ok { rendered } = v {
            assert_eq!(rendered, "<Ctrl+Alt+K>");
        }
    }
}
