//! # TUI Password Input Mask
//!
//! Mask password chars with `*` (or other glyph). Optional reveal
//! toggle returns plain text when reveal=true.
//!
//! Demonstrates the **TUI.127** recipe for PMAT-202 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTML `<input type=password>` UI; bash `read -s`.
//!
//! Run with: cargo run --example tui_password_input_mask
//!
//! Added by PMAT-202 (catalog 1441→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum MaskVerdict {
    Ok { display: String, char_count: u32 },
    InvalidConfig,
}

pub fn render(password: &str, mask_glyph: char, reveal: bool) -> MaskVerdict {
    if password.is_empty() {
        return MaskVerdict::InvalidConfig;
    }
    let char_count = password.chars().count() as u32;
    let display = if reveal {
        password.to_string()
    } else {
        std::iter::repeat(mask_glyph)
            .take(char_count as usize)
            .collect()
    };
    MaskVerdict::Ok {
        display,
        char_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_password_input_mask")?;

    println!("masked: {:?}", render("hunter2", '*', false));
    println!("revealed: {:?}", render("hunter2", '*', true));
    println!("invalid: {:?}", render("", '*', false));
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
    fn masked_chars_correct() {
        let v = render("abc", '*', false);
        if let MaskVerdict::Ok { display, .. } = v {
            assert_eq!(display, "***");
        }
    }

    #[test]
    fn reveal_shows_plaintext() {
        let v = render("abc", '*', true);
        if let MaskVerdict::Ok { display, .. } = v {
            assert_eq!(display, "abc");
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(render("", '*', false), MaskVerdict::InvalidConfig);
    }

    #[test]
    fn char_count_matches() {
        let v = render("hello", '*', false);
        if let MaskVerdict::Ok { char_count, .. } = v {
            assert_eq!(char_count, 5);
        }
    }

    #[test]
    fn dot_glyph_works() {
        let v = render("abc", '•', false);
        if let MaskVerdict::Ok { display, .. } = v {
            assert_eq!(display, "•••");
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render("abc", '*', false);
        let r2 = render("abc", '*', false);
        assert_eq!(r1, r2);
    }

    #[test]
    fn unicode_password_handled() {
        let v = render("café", '*', false);
        if let MaskVerdict::Ok {
            display,
            char_count,
        } = v
        {
            assert_eq!(char_count, 4);
            assert_eq!(display.chars().count(), 4);
        }
    }

    #[test]
    fn unicode_reveal() {
        let v = render("café", '*', true);
        if let MaskVerdict::Ok { display, .. } = v {
            assert_eq!(display, "café");
        }
    }

    #[test]
    fn long_password_handled() {
        let v = render("password_with_many_chars", '*', false);
        if let MaskVerdict::Ok {
            display,
            char_count,
        } = v
        {
            assert_eq!(display.chars().count() as u32, char_count);
        }
    }

    #[test]
    fn single_char_works() {
        let v = render("a", '*', false);
        if let MaskVerdict::Ok { display, .. } = v {
            assert_eq!(display, "*");
        }
    }

    #[test]
    fn space_glyph_treated_normally() {
        let v = render("abc", ' ', false);
        if let MaskVerdict::Ok { display, .. } = v {
            assert_eq!(display, "   ");
        }
    }
}
