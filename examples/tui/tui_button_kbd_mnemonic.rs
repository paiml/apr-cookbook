//! # TUI Button Keyboard Mnemonic
//!
//! Render a button label with a single underlined character serving
//! as a keyboard mnemonic (e.g. `_S_ave` → press `s` to activate).
//! Returns rendered string + activation key.
//!
//! Demonstrates the **TUI.109** recipe for PMAT-196 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: macOS Cocoa "Set Key Equivalent"; Windows Common
//!  Controls Alt+letter access keys.
//!
//! Run with: cargo run --example tui_button_kbd_mnemonic
//!
//! Added by PMAT-196 (catalog 1387→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MnemonicVerdict {
    Ok {
        rendered: String,
        activation_key: char,
    },
    InvalidConfig,
}

pub fn render(label: &str, mnemonic_index: u32) -> MnemonicVerdict {
    if label.is_empty() {
        return MnemonicVerdict::InvalidConfig;
    }
    let chars: Vec<char> = label.chars().collect();
    if (mnemonic_index as usize) >= chars.len() {
        return MnemonicVerdict::InvalidConfig;
    }
    let activation_char = chars[mnemonic_index as usize];
    if !activation_char.is_alphabetic() {
        return MnemonicVerdict::InvalidConfig;
    }
    let mut rendered = String::new();
    for (i, c) in chars.iter().enumerate() {
        if i as u32 == mnemonic_index {
            rendered.push('_');
            rendered.push(*c);
            rendered.push('_');
        } else {
            rendered.push(*c);
        }
    }
    MnemonicVerdict::Ok {
        rendered,
        activation_key: activation_char.to_ascii_lowercase(),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_button_kbd_mnemonic")?;

    println!("Save: {:?}", render("Save", 0));
    println!("Cancel: {:?}", render("Cancel", 0));
    println!("invalid: {:?}", render("", 0));
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
    fn underscore_marks_mnemonic() {
        let v = render("Save", 0);
        if let MnemonicVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("_S_"));
        }
    }

    #[test]
    fn activation_key_lowercase() {
        let v = render("Save", 0);
        if let MnemonicVerdict::Ok { activation_key, .. } = v {
            assert_eq!(activation_key, 's');
        }
    }

    #[test]
    fn empty_label_rejected() {
        assert_eq!(render("", 0), MnemonicVerdict::InvalidConfig);
    }

    #[test]
    fn out_of_range_rejected() {
        assert_eq!(render("ab", 5), MnemonicVerdict::InvalidConfig);
    }

    #[test]
    fn non_alphabetic_rejected() {
        assert_eq!(render("123", 0), MnemonicVerdict::InvalidConfig);
    }

    #[test]
    fn middle_char_mnemonic() {
        let v = render("Cancel", 1);
        if let MnemonicVerdict::Ok {
            rendered,
            activation_key,
        } = v
        {
            assert!(rendered.contains("C_a_"));
            assert_eq!(activation_key, 'a');
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render("Save", 0);
        let r2 = render("Save", 0);
        assert_eq!(r1, r2);
    }

    #[test]
    fn last_char_mnemonic() {
        let v = render("ok", 1);
        if let MnemonicVerdict::Ok {
            rendered,
            activation_key,
        } = v
        {
            assert!(rendered.ends_with('_'));
            assert_eq!(activation_key, 'k');
        }
    }

    #[test]
    fn rendered_includes_underscores() {
        let v = render("Save", 0);
        if let MnemonicVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered.matches('_').count(), 2);
        }
    }

    #[test]
    fn unicode_label_supported() {
        let v = render("Café", 0);
        if let MnemonicVerdict::Ok { activation_key, .. } = v {
            assert_eq!(activation_key, 'c');
        }
    }

    #[test]
    fn label_chars_preserved() {
        let v = render("Save", 0);
        if let MnemonicVerdict::Ok { rendered, .. } = v {
            for c in "Save".chars() {
                assert!(rendered.contains(c));
            }
        }
    }
}
