//! # TUI Password Mask
//!
//! Mask password input with `•` glyphs unless reveal mode is on.
//! Returns the displayed string and the underlying length.
//!
//! Demonstrates the **TUI.48** recipe for PMAT-175 (catalog crosses 1200).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML input[type=password] reveal toggle convention.
//!
//! Run with: cargo run --example tui_password_mask
//!
//! Added by PMAT-175 (catalog 1198→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MaskVerdict {
    Ok { displayed: String, length: u32 },
    InvalidConfig,
}

pub fn mask(password: &str, reveal: bool, max_visible: usize) -> MaskVerdict {
    if max_visible == 0 {
        return MaskVerdict::InvalidConfig;
    }
    let n = password.chars().count();
    let length = n as u32;
    if reveal {
        let truncated: String = if n > max_visible {
            let kept: String = password
                .chars()
                .take(max_visible.saturating_sub(1))
                .collect();
            format!("{kept}…")
        } else {
            password.to_string()
        };
        return MaskVerdict::Ok {
            displayed: truncated,
            length,
        };
    }
    let visible_count = n.min(max_visible);
    let displayed: String = "•".repeat(visible_count);
    MaskVerdict::Ok { displayed, length }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_password_mask")?;

    println!("hidden: {:?}", mask("secret", false, 32));
    println!("revealed: {:?}", mask("secret", true, 32));
    println!("hidden truncated: {:?}", mask("verylongpassword", false, 8));
    println!(
        "revealed truncated: {:?}",
        mask("verylongpassword", true, 8)
    );
    println!("invalid: {:?}", mask("secret", false, 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn masker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn hidden_is_bullets() {
        let v = mask("hello", false, 32);
        if let MaskVerdict::Ok { displayed, length } = v {
            assert_eq!(displayed, "•••••");
            assert_eq!(length, 5);
        }
    }

    #[test]
    fn revealed_shows_text() {
        let v = mask("hello", true, 32);
        if let MaskVerdict::Ok { displayed, .. } = v {
            assert_eq!(displayed, "hello");
        }
    }

    #[test]
    fn hidden_truncated_at_max() {
        let v = mask("verylongpassword", false, 5);
        if let MaskVerdict::Ok { displayed, .. } = v {
            assert_eq!(displayed.chars().count(), 5);
        }
    }

    #[test]
    fn revealed_truncated_with_ellipsis() {
        let v = mask("verylongpassword", true, 5);
        if let MaskVerdict::Ok { displayed, .. } = v {
            assert!(displayed.ends_with('…'));
        }
    }

    #[test]
    fn empty_password() {
        let v = mask("", false, 10);
        if let MaskVerdict::Ok { displayed, length } = v {
            assert_eq!(displayed, "");
            assert_eq!(length, 0);
        }
    }

    #[test]
    fn invalid_zero_max() {
        assert_eq!(mask("x", false, 0), MaskVerdict::InvalidConfig);
    }

    #[test]
    fn unicode_password_hidden() {
        let v = mask("café", false, 10);
        if let MaskVerdict::Ok { displayed, length } = v {
            assert_eq!(displayed.chars().count(), 4);
            assert_eq!(length, 4);
        }
    }

    #[test]
    fn unicode_password_revealed() {
        let v = mask("café", true, 10);
        if let MaskVerdict::Ok { displayed, .. } = v {
            assert_eq!(displayed, "café");
        }
    }

    #[test]
    fn length_preserved_under_truncation() {
        let v = mask("verylongpassword", false, 5);
        if let MaskVerdict::Ok { length, .. } = v {
            assert_eq!(length, 16);
        }
    }

    #[test]
    fn deterministic() {
        let a = mask("hello", false, 32);
        let b = mask("hello", false, 32);
        assert_eq!(a, b);
    }
}
