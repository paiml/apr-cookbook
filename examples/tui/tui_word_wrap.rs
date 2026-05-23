//! # TUI Word Wrap
//!
//! Wrap a long line at word boundaries to fit `max_width` columns.
//! Returns wrapped lines preserving original word order.
//!
//! Demonstrates the **TUI.85** recipe for PMAT-188 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Knuth & Plass, "Breaking Paragraphs into Lines"
//!  (Software Practice & Experience 1981).
//!
//! Run with: cargo run --example tui_word_wrap
//!
//! Added by PMAT-188 (catalog 1315→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WrapVerdict {
    Ok { lines: Vec<String> },
    InvalidConfig,
}

pub fn wrap(text: &str, max_width: u32) -> WrapVerdict {
    if text.is_empty() || max_width == 0 {
        return WrapVerdict::InvalidConfig;
    }
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        let w_len = word.chars().count() as u32;
        if w_len > max_width {
            // Word longer than width — push as its own line.
            if !current.is_empty() {
                lines.push(std::mem::take(&mut current));
            }
            lines.push(word.to_string());
            continue;
        }
        let cur_len = current.chars().count() as u32;
        let add = if cur_len == 0 { w_len } else { 1 + w_len };
        if cur_len + add > max_width {
            lines.push(std::mem::take(&mut current));
            current = word.to_string();
        } else {
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(word);
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    WrapVerdict::Ok { lines }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_word_wrap")?;

    println!("short width: {:?}", wrap("the quick brown fox jumps", 10));
    println!("invalid: {:?}", wrap("", 10));
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
    fn no_wrap_when_fits() {
        let v = wrap("hello world", 100);
        if let WrapVerdict::Ok { lines } = v {
            assert_eq!(lines, vec!["hello world".to_string()]);
        }
    }

    #[test]
    fn wrap_at_width() {
        let v = wrap("the quick brown fox", 10);
        if let WrapVerdict::Ok { lines } = v {
            for line in &lines {
                assert!(line.chars().count() <= 10);
            }
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(wrap("", 10), WrapVerdict::InvalidConfig);
    }

    #[test]
    fn zero_width_rejected() {
        assert_eq!(wrap("hi", 0), WrapVerdict::InvalidConfig);
    }

    #[test]
    fn long_word_on_own_line() {
        let v = wrap("a verylongwordthatexceeds limit", 10);
        if let WrapVerdict::Ok { lines } = v {
            assert!(lines.iter().any(|l| l == "verylongwordthatexceeds"));
        }
    }

    #[test]
    fn deterministic() {
        let r1 = wrap("hello world", 5);
        let r2 = wrap("hello world", 5);
        assert_eq!(r1, r2);
    }

    #[test]
    fn single_word_works() {
        let v = wrap("hi", 10);
        if let WrapVerdict::Ok { lines } = v {
            assert_eq!(lines, vec!["hi".to_string()]);
        }
    }

    #[test]
    fn whitespace_collapsed() {
        let v = wrap("hello    world", 100);
        if let WrapVerdict::Ok { lines } = v {
            assert_eq!(lines[0], "hello world");
        }
    }

    #[test]
    fn unicode_word_counted_by_char() {
        let v = wrap("café résumé", 5);
        if let WrapVerdict::Ok { lines } = v {
            for line in &lines {
                assert!(line.chars().count() <= 6); // café=4, résumé=6
            }
        }
    }

    #[test]
    fn many_words_split_correctly() {
        let v = wrap("a b c d e f g", 3);
        if let WrapVerdict::Ok { lines } = v {
            assert!(lines.len() > 2);
        }
    }

    #[test]
    fn lines_no_trailing_whitespace() {
        let v = wrap("the quick fox", 8);
        if let WrapVerdict::Ok { lines } = v {
            for line in &lines {
                assert!(!line.ends_with(' '));
            }
        }
    }
}
