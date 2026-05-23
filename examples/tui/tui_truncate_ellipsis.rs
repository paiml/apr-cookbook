//! # TUI Truncate with Ellipsis
//!
//! Truncate a string to fit a column width. If too long, replace last
//! 3 chars with `…` and report `truncated: true`. Char-aware (handles
//! ASCII; multibyte preserved at boundary).
//!
//! Demonstrates the **TUI.05** recipe for PMAT-161 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Unicode wcwidth + tui-rs/ratatui truncation conventions.
//!
//! Run with: cargo run --example tui_truncate_ellipsis
//!
//! Added by PMAT-161 (catalog 1072→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TruncVerdict {
    Ok { text: String, truncated: bool },
    InvalidWidth,
}

pub fn truncate(input: &str, width: usize) -> TruncVerdict {
    if width == 0 {
        return TruncVerdict::InvalidWidth;
    }
    let char_count = input.chars().count();
    if char_count <= width {
        return TruncVerdict::Ok {
            text: input.to_string(),
            truncated: false,
        };
    }
    if width < 2 {
        // Not enough room for ellipsis; just truncate.
        let cut: String = input.chars().take(width).collect();
        return TruncVerdict::Ok {
            text: cut,
            truncated: true,
        };
    }
    // Reserve 1 char for the ellipsis.
    let kept: String = input.chars().take(width - 1).collect();
    TruncVerdict::Ok {
        text: format!("{kept}…"),
        truncated: true,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_truncate_ellipsis")?;

    println!("short: {:?}", truncate("hello", 10));
    println!("exact: {:?}", truncate("hello", 5));
    println!("truncated: {:?}", truncate("hello world", 8));
    println!("very narrow: {:?}", truncate("hello", 1));
    println!("invalid: {:?}", truncate("hello", 0));
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
    fn short_string_unchanged() {
        let v = truncate("hello", 10);
        if let TruncVerdict::Ok { text, truncated } = v {
            assert_eq!(text, "hello");
            assert!(!truncated);
        }
    }

    #[test]
    fn exact_width_unchanged() {
        let v = truncate("hello", 5);
        if let TruncVerdict::Ok { text, truncated } = v {
            assert_eq!(text, "hello");
            assert!(!truncated);
        }
    }

    #[test]
    fn long_string_truncated_with_ellipsis() {
        let v = truncate("hello world", 8);
        if let TruncVerdict::Ok { text, truncated } = v {
            assert!(text.ends_with('…'));
            assert!(truncated);
            assert_eq!(text.chars().count(), 8);
        }
    }

    #[test]
    fn width_one_no_ellipsis() {
        let v = truncate("hello", 1);
        if let TruncVerdict::Ok { text, truncated } = v {
            assert_eq!(text, "h");
            assert!(truncated);
        }
    }

    #[test]
    fn zero_width_invalid() {
        assert_eq!(truncate("hello", 0), TruncVerdict::InvalidWidth);
    }

    #[test]
    fn empty_input_unchanged() {
        let v = truncate("", 10);
        if let TruncVerdict::Ok { text, truncated } = v {
            assert_eq!(text, "");
            assert!(!truncated);
        }
    }

    #[test]
    fn unicode_input_truncated() {
        let v = truncate("héllo wörld", 8);
        if let TruncVerdict::Ok { text, truncated } = v {
            assert!(truncated);
            assert!(text.ends_with('…'));
        }
    }

    #[test]
    fn width_two_keeps_one_plus_ellipsis() {
        let v = truncate("hello", 2);
        if let TruncVerdict::Ok { text, truncated } = v {
            assert_eq!(text, "h…");
            assert!(truncated);
        }
    }

    #[test]
    fn just_one_over_truncates() {
        let v = truncate("123456", 5);
        if let TruncVerdict::Ok { text, .. } = v {
            assert_eq!(text, "1234…");
        }
    }

    #[test]
    fn deterministic() {
        let a = truncate("hello world", 8);
        let b = truncate("hello world", 8);
        assert_eq!(a, b);
    }
}
