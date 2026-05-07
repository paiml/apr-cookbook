//! # TUI Inline Emoji Replace
//!
//! Replace shortcode-style emoji (`:smile:`) with Unicode glyphs in a
//! text buffer. Returns rendered text and the replacement count.
//!
//! Demonstrates the **TUI.177** recipe for PMAT-223 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Slack/Discord emoji shortcode rendering; CommonMark
//!  GFM emoji extension.
//!
//! Run with: cargo run --example tui_inline_emoji_replace
//!
//! Added by PMAT-223 (catalog 1630→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum EmojiVerdict {
    Ok { rendered: String, replacements: u32 },
    InvalidConfig,
}

pub fn replace(text: &str) -> EmojiVerdict {
    if text.is_empty() {
        return EmojiVerdict::InvalidConfig;
    }
    let map: &[(&str, &str)] = &[
        (":smile:", "😄"),
        (":heart:", "❤️"),
        (":thumbsup:", "👍"),
        (":fire:", "🔥"),
        (":check:", "✓"),
        (":x:", "✗"),
        (":warning:", "⚠"),
    ];
    let mut result = text.to_string();
    let mut count = 0u32;
    for (code, glyph) in map {
        let occurrences = result.matches(code).count();
        if occurrences > 0 {
            result = result.replace(code, glyph);
            count += occurrences as u32;
        }
    }
    EmojiVerdict::Ok {
        rendered: result,
        replacements: count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_inline_emoji_replace")?;

    println!("smile: {:?}", replace("hello :smile:"));
    println!("multiple: {:?}", replace(":heart: :fire:"));
    println!("none: {:?}", replace("plain text"));
    println!("invalid: {:?}", replace(""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replacer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(replace(""), EmojiVerdict::InvalidConfig);
    }

    #[test]
    fn smile_replaced() {
        let v = replace(":smile:");
        if let EmojiVerdict::Ok { rendered, .. } = v {
            assert_eq!(rendered, "😄");
        }
    }

    #[test]
    fn multiple_codes_replaced() {
        let v = replace(":heart: :fire:");
        if let EmojiVerdict::Ok { replacements, .. } = v {
            assert_eq!(replacements, 2);
        }
    }

    #[test]
    fn no_codes_zero_replacements() {
        let v = replace("plain text");
        if let EmojiVerdict::Ok { replacements, .. } = v {
            assert_eq!(replacements, 0);
        }
    }

    #[test]
    fn surrounding_text_preserved() {
        let v = replace("hello :smile: world");
        if let EmojiVerdict::Ok { rendered, .. } = v {
            assert!(rendered.starts_with("hello"));
            assert!(rendered.ends_with("world"));
        }
    }

    #[test]
    fn unknown_code_unchanged() {
        let v = replace(":unknown:");
        if let EmojiVerdict::Ok {
            rendered,
            replacements,
        } = v
        {
            assert_eq!(rendered, ":unknown:");
            assert_eq!(replacements, 0);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = replace(":smile:");
        let r2 = replace(":smile:");
        assert_eq!(r1, r2);
    }

    #[test]
    fn count_includes_repeats() {
        let v = replace(":fire: :fire: :fire:");
        if let EmojiVerdict::Ok { replacements, .. } = v {
            assert_eq!(replacements, 3);
        }
    }

    #[test]
    fn check_and_x_glyphs() {
        let v = replace(":check: :x:");
        if let EmojiVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains('✓'));
            assert!(rendered.contains('✗'));
        }
    }

    #[test]
    fn warning_replaced() {
        let v = replace(":warning:");
        if let EmojiVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains('⚠'));
        }
    }

    #[test]
    fn long_text_handled() {
        let text = "a very long text with :smile: and :heart: and :fire: scattered around";
        let v = replace(text);
        if let EmojiVerdict::Ok { replacements, .. } = v {
            assert_eq!(replacements, 3);
        }
    }

    #[test]
    fn unicode_in_input_preserved() {
        let v = replace("café :smile:");
        if let EmojiVerdict::Ok { rendered, .. } = v {
            assert!(rendered.contains("café"));
            assert!(rendered.contains('😄'));
        }
    }

    #[test]
    fn partial_code_not_replaced() {
        let v = replace(":smil");
        if let EmojiVerdict::Ok {
            rendered,
            replacements,
        } = v
        {
            assert_eq!(rendered, ":smil");
            assert_eq!(replacements, 0);
        }
    }
}
