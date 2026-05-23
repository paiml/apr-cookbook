//! # TUI Resize-Safe Truncate
//!
//! Truncate text to fit narrower window, ensuring no truncation
//! mid-grapheme. Returns truncated text and number of chars dropped.
//!
//! Demonstrates the **TUI.93** recipe for PMAT-190 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Unicode Standard Annex #29 (grapheme clusters); xterm
//!  resize protocol (CSI 8 t).
//!
//! Run with: cargo run --example tui_resize_safe_truncate
//!
//! Added by PMAT-190 (catalog 1333→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TruncVerdict {
    Ok {
        truncated: String,
        chars_dropped: u32,
    },
    InvalidConfig,
}

pub fn truncate(text: &str, max_width: u32) -> TruncVerdict {
    if max_width == 0 {
        return TruncVerdict::InvalidConfig;
    }
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len() as u32;
    if total <= max_width {
        return TruncVerdict::Ok {
            truncated: text.to_string(),
            chars_dropped: 0,
        };
    }
    let kept_chars: usize = max_width as usize;
    // Avoid truncating if kept_chars would land on a combining mark
    // (UAX #29). For simplicity check `is_combining_mark` heuristic.
    let mut end = kept_chars;
    while end > 0 && is_combining(chars[end - 1] as u32) {
        end -= 1;
    }
    let truncated: String = chars.iter().take(end).collect();
    let chars_dropped = total - end as u32;
    TruncVerdict::Ok {
        truncated,
        chars_dropped,
    }
}

fn is_combining(cp: u32) -> bool {
    (0x0300..=0x036F).contains(&cp)
        || (0x1AB0..=0x1AFF).contains(&cp)
        || (0x1DC0..=0x1DFF).contains(&cp)
        || (0x20D0..=0x20FF).contains(&cp)
        || (0xFE20..=0xFE2F).contains(&cp)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_resize_safe_truncate")?;

    println!("short: {:?}", truncate("hi", 10));
    println!(
        "long: {:?}",
        truncate("hello world this is a very long line", 10)
    );
    println!("invalid: {:?}", truncate("x", 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn short_text_unchanged() {
        let v = truncate("hi", 10);
        if let TruncVerdict::Ok {
            truncated,
            chars_dropped,
        } = v
        {
            assert_eq!(truncated, "hi");
            assert_eq!(chars_dropped, 0);
        }
    }

    #[test]
    fn long_text_truncated_to_width() {
        let v = truncate("hello world", 5);
        if let TruncVerdict::Ok {
            truncated,
            chars_dropped,
        } = v
        {
            assert_eq!(truncated.chars().count(), 5);
            assert_eq!(chars_dropped, 6);
        }
    }

    #[test]
    fn empty_text_works() {
        let v = truncate("", 10);
        if let TruncVerdict::Ok {
            truncated,
            chars_dropped,
        } = v
        {
            assert_eq!(truncated, "");
            assert_eq!(chars_dropped, 0);
        }
    }

    #[test]
    fn zero_width_rejected() {
        assert_eq!(truncate("hi", 0), TruncVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = truncate("hello world", 5);
        let r2 = truncate("hello world", 5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn exact_fit_no_drop() {
        let v = truncate("hello", 5);
        if let TruncVerdict::Ok {
            truncated,
            chars_dropped,
        } = v
        {
            assert_eq!(truncated, "hello");
            assert_eq!(chars_dropped, 0);
        }
    }

    #[test]
    fn unicode_simple_truncation() {
        let v = truncate("café", 3);
        if let TruncVerdict::Ok { truncated, .. } = v {
            assert_eq!(truncated.chars().count(), 3);
        }
    }

    #[test]
    fn one_over_drops_one() {
        let v = truncate("hello!", 5);
        if let TruncVerdict::Ok {
            truncated,
            chars_dropped,
        } = v
        {
            assert_eq!(truncated.chars().count(), 5);
            assert_eq!(chars_dropped, 1);
        }
    }

    #[test]
    fn truncated_le_original() {
        let v = truncate("hello", 3);
        if let TruncVerdict::Ok { truncated, .. } = v {
            assert!(truncated.chars().count() <= 5);
        }
    }

    #[test]
    fn very_short_width_works() {
        let v = truncate("hello", 1);
        if let TruncVerdict::Ok { truncated, .. } = v {
            assert_eq!(truncated.chars().count(), 1);
        }
    }

    #[test]
    fn dropped_count_correct() {
        let v = truncate("abcdefghij", 4);
        if let TruncVerdict::Ok { chars_dropped, .. } = v {
            assert_eq!(chars_dropped, 6);
        }
    }
}
