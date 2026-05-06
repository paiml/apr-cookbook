//! # TUI Unicode Emoji Width
//!
//! Compute visual width of a string with a heuristic emoji-aware
//! rule: codepoints in standard emoji ranges are width 2; ASCII
//! codepoints are width 1; combining marks are width 0.
//!
//! Demonstrates the **TUI.82** recipe for PMAT-187 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Unicode Standard Annex #11 (East Asian Width); Unicode
//!  Emoji 14.0 (CLDR).
//!
//! Run with: cargo run --example tui_unicode_emoji_width
//!
//! Added by PMAT-187 (catalog 1306→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WidthVerdict {
    Ok { width: u32, char_count: u32 },
    InvalidConfig,
}

pub fn measure(text: &str) -> WidthVerdict {
    if text.is_empty() {
        return WidthVerdict::InvalidConfig;
    }
    let mut width: u32 = 0;
    let mut char_count: u32 = 0;
    for c in text.chars() {
        char_count += 1;
        width += char_width(c);
    }
    WidthVerdict::Ok { width, char_count }
}

fn char_width(c: char) -> u32 {
    let cp = c as u32;
    if c.is_ascii() {
        u32::from(!c.is_control())
    } else if is_emoji(cp) {
        2
    } else {
        u32::from(!is_combining(cp))
    }
}

fn is_emoji(cp: u32) -> bool {
    (0x1F300..=0x1F5FF).contains(&cp) // Misc Symbols & Pictographs
        || (0x1F600..=0x1F64F).contains(&cp) // Emoticons
        || (0x1F680..=0x1F6FF).contains(&cp) // Transport & Map
        || (0x1F900..=0x1F9FF).contains(&cp) // Supplemental Symbols
        || (0x1FA70..=0x1FAFF).contains(&cp) // Symbols & Pictographs Ext-A
        || (0x2600..=0x26FF).contains(&cp) // Misc Symbols (some emoji)
}

fn is_combining(cp: u32) -> bool {
    (0x0300..=0x036F).contains(&cp)
        || (0x1AB0..=0x1AFF).contains(&cp)
        || (0x1DC0..=0x1DFF).contains(&cp)
        || (0x20D0..=0x20FF).contains(&cp)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_unicode_emoji_width")?;

    println!("ascii: {:?}", measure("hello"));
    println!("emoji: {:?}", measure("Hi 🚀"));
    println!("combining: {:?}", measure("é"));
    println!("invalid: {:?}", measure(""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measurer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn ascii_string_width_equals_length() {
        let v = measure("hello");
        if let WidthVerdict::Ok {
            width, char_count, ..
        } = v
        {
            assert_eq!(width, 5);
            assert_eq!(char_count, 5);
        }
    }

    #[test]
    fn rocket_emoji_is_width_two() {
        let v = measure("🚀");
        if let WidthVerdict::Ok { width, .. } = v {
            assert_eq!(width, 2);
        }
    }

    #[test]
    fn ascii_plus_emoji_combined() {
        let v = measure("Hi 🚀");
        if let WidthVerdict::Ok { width, .. } = v {
            // "Hi " = 3, "🚀" = 2 → 5.
            assert_eq!(width, 5);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(measure(""), WidthVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = measure("hello");
        let r2 = measure("hello");
        assert_eq!(r1, r2);
    }

    #[test]
    fn smile_emoji_width_two() {
        let v = measure("😀");
        if let WidthVerdict::Ok { width, .. } = v {
            assert_eq!(width, 2);
        }
    }

    #[test]
    fn control_char_zero_width() {
        let v = measure("a\tb");
        if let WidthVerdict::Ok { width, .. } = v {
            // a=1, tab=0 (control), b=1 → 2.
            assert_eq!(width, 2);
        }
    }

    #[test]
    fn multiple_emojis_sum() {
        let v = measure("🚀🎉");
        if let WidthVerdict::Ok { width, .. } = v {
            assert_eq!(width, 4);
        }
    }

    #[test]
    fn char_count_distinct_from_width() {
        let v = measure("🚀x");
        if let WidthVerdict::Ok {
            width, char_count, ..
        } = v
        {
            assert_eq!(char_count, 2);
            assert_eq!(width, 3);
        }
    }

    #[test]
    fn widget_widths_separate() {
        let v = measure("[a]");
        if let WidthVerdict::Ok { width, .. } = v {
            assert_eq!(width, 3);
        }
    }

    #[test]
    fn unicode_letter_width_one() {
        let v = measure("résumé");
        if let WidthVerdict::Ok { width, .. } = v {
            // 6 letters, none in emoji range → width = 6.
            assert_eq!(width, 6);
        }
    }
}
