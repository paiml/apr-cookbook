//! # TUI Word Count Status Bar
//!
//! Compute word/char/line counts for a buffer suitable for status-bar
//! display. Returns formatted status string and the three counts.
//!
//! Demonstrates the **TUI.143** recipe for PMAT-207 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim `g Ctrl-G` word-count display; Sublime Text status-
//!  bar selection counts.
//!
//! Run with: cargo run --example tui_word_count_status
//!
//! Added by PMAT-207 (catalog 1486→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WordCountVerdict {
    Ok {
        status: String,
        words: u32,
        chars: u32,
        lines: u32,
    },
    InvalidConfig,
}

pub fn compute(buffer: &str) -> WordCountVerdict {
    if buffer.is_empty() {
        return WordCountVerdict::InvalidConfig;
    }
    let chars = buffer.chars().count() as u32;
    let words = buffer.split_whitespace().count() as u32;
    let lines = buffer.split('\n').count() as u32;
    let status = format!("L:{lines} W:{words} C:{chars}");
    WordCountVerdict::Ok {
        status,
        words,
        chars,
        lines,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_word_count_status")?;

    println!("text: {:?}", compute("hello world\nbye"));
    println!("invalid: {:?}", compute(""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn single_word_count() {
        let v = compute("hello");
        if let WordCountVerdict::Ok { words, .. } = v {
            assert_eq!(words, 1);
        }
    }

    #[test]
    fn multiple_words_count() {
        let v = compute("hello world foo");
        if let WordCountVerdict::Ok { words, .. } = v {
            assert_eq!(words, 3);
        }
    }

    #[test]
    fn char_count_correct() {
        let v = compute("hello");
        if let WordCountVerdict::Ok { chars, .. } = v {
            assert_eq!(chars, 5);
        }
    }

    #[test]
    fn line_count_correct() {
        let v = compute("a\nb\nc");
        if let WordCountVerdict::Ok { lines, .. } = v {
            assert_eq!(lines, 3);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(compute(""), WordCountVerdict::InvalidConfig);
    }

    #[test]
    fn single_line_count_one() {
        let v = compute("hello");
        if let WordCountVerdict::Ok { lines, .. } = v {
            assert_eq!(lines, 1);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = compute("hello");
        let r2 = compute("hello");
        assert_eq!(r1, r2);
    }

    #[test]
    fn status_string_formatted() {
        let v = compute("abc");
        if let WordCountVerdict::Ok { status, .. } = v {
            assert!(status.contains("L:1"));
            assert!(status.contains("W:1"));
            assert!(status.contains("C:3"));
        }
    }

    #[test]
    fn unicode_chars_counted() {
        let v = compute("café");
        if let WordCountVerdict::Ok { chars, .. } = v {
            assert_eq!(chars, 4);
        }
    }

    #[test]
    fn whitespace_only_zero_words() {
        let v = compute("   \n  ");
        if let WordCountVerdict::Ok { words, .. } = v {
            assert_eq!(words, 0);
        }
    }

    #[test]
    fn many_lines_handled() {
        let buf: String = (0..30).map(|_| "line\n").collect();
        let v = compute(&buf);
        if let WordCountVerdict::Ok { lines, .. } = v {
            // 30 newlines split → 31 elements (last empty).
            assert_eq!(lines, 31);
        }
    }

    #[test]
    fn tabs_count_as_whitespace() {
        let v = compute("a\tb\tc");
        if let WordCountVerdict::Ok { words, .. } = v {
            assert_eq!(words, 3);
        }
    }
}
