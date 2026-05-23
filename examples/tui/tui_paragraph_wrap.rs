//! # TUI Paragraph Wrap
//!
//! Wrap a paragraph to fit a target column width. Greedy wrap: each
//! line packs as many whole words as fit; long words are kept whole
//! (overflow allowed) rather than split.
//!
//! Demonstrates the **TUI.15** recipe for PMAT-164 (catalog crosses 1100).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: textwrap crate's "first-fit" algorithm.
//!
//! Run with: cargo run --example tui_paragraph_wrap
//!
//! Added by PMAT-164 (catalog 1099→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WrapVerdict {
    Ok { lines: Vec<String> },
    EmptyText,
    InvalidWidth,
}

pub fn wrap(text: &str, width: usize) -> WrapVerdict {
    if text.trim().is_empty() {
        return WrapVerdict::EmptyText;
    }
    if width == 0 {
        return WrapVerdict::InvalidWidth;
    }
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let word_len = word.chars().count();
        if current.is_empty() {
            current.push_str(word);
            continue;
        }
        let proposed_len = current.chars().count() + 1 + word_len;
        if proposed_len <= width {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(std::mem::take(&mut current));
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    WrapVerdict::Ok { lines }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_paragraph_wrap")?;

    let text = "The quick brown fox jumps over the lazy dog";
    println!("width=20: {:?}", wrap(text, 20));
    println!("width=50: {:?}", wrap(text, 50));
    println!("very narrow: {:?}", wrap(text, 5));
    println!("empty: {:?}", wrap("", 20));
    println!("invalid: {:?}", wrap("hello", 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapper_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn fits_in_one_line() {
        let v = wrap("hello world", 20);
        if let WrapVerdict::Ok { lines } = v {
            assert_eq!(lines, vec!["hello world".to_string()]);
        }
    }

    #[test]
    fn wraps_multiple_lines() {
        let v = wrap("a b c d e", 3);
        if let WrapVerdict::Ok { lines } = v {
            // Each line ≤ 3 chars: "a b" (3), "c d" (3), "e" (1).
            assert_eq!(lines.len(), 3);
        }
    }

    #[test]
    fn long_word_kept_whole() {
        let v = wrap("hi supercalifragilistic bye", 5);
        if let WrapVerdict::Ok { lines } = v {
            // Long word doesn't fit but stays as its own line.
            let has_long = lines.iter().any(|l| l.contains("supercalifragilistic"));
            assert!(has_long);
        }
    }

    #[test]
    fn empty_text_rejected() {
        assert_eq!(wrap("   ", 20), WrapVerdict::EmptyText);
    }

    #[test]
    fn zero_width_invalid() {
        assert_eq!(wrap("hello", 0), WrapVerdict::InvalidWidth);
    }

    #[test]
    fn single_word() {
        let v = wrap("hello", 20);
        if let WrapVerdict::Ok { lines } = v {
            assert_eq!(lines, vec!["hello".to_string()]);
        }
    }

    #[test]
    fn all_lines_within_width_when_possible() {
        let v = wrap("one two three four five", 10);
        if let WrapVerdict::Ok { lines } = v {
            for line in &lines {
                if !line.contains(' ') || line.chars().count() <= 10 {
                    // OK.
                } else {
                    panic!("line over width: {line}");
                }
            }
        }
    }

    #[test]
    fn whitespace_collapsed() {
        let v = wrap("  hello   world  ", 20);
        if let WrapVerdict::Ok { lines } = v {
            assert_eq!(lines, vec!["hello world".to_string()]);
        }
    }

    #[test]
    fn unicode_words() {
        let v = wrap("café résumé naïve", 20);
        if let WrapVerdict::Ok { lines } = v {
            assert!(!lines.is_empty());
        }
    }

    #[test]
    fn deterministic() {
        let a = wrap("hello world foo bar", 10);
        let b = wrap("hello world foo bar", 10);
        assert_eq!(a, b);
    }
}
