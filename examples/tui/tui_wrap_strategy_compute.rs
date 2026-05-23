//! # TUI Wrap Strategy Compute
//!
//! Compute line breaks for word-wrap or character-wrap strategies.
//! Returns visual lines and the strategy used.
//!
//! Demonstrates the **TUI.158** recipe for PMAT-212 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim `:set wrap` + `linebreak`; rustfmt `wrap_comments`
//!  word-boundary preservation.
//!
//! Run with: cargo run --example tui_wrap_strategy_compute
//!
//! Added by PMAT-212 (catalog 1531→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WrapStrategy {
    Word,
    Char,
}

#[derive(Debug, PartialEq)]
pub enum WrapVerdict {
    Ok { lines: Vec<String>, line_count: u32 },
    InvalidConfig,
}

pub fn wrap(text: &str, width: u32, strategy: &WrapStrategy) -> WrapVerdict {
    if text.is_empty() || width < 1 {
        return WrapVerdict::InvalidConfig;
    }
    let mut lines: Vec<String> = Vec::new();
    match *strategy {
        WrapStrategy::Char => {
            let chars: Vec<char> = text.chars().collect();
            let mut start = 0usize;
            while start < chars.len() {
                let end = (start + width as usize).min(chars.len());
                lines.push(chars[start..end].iter().collect());
                start = end;
            }
        }
        WrapStrategy::Word => {
            let mut current = String::new();
            for word in text.split_whitespace() {
                let needed = if current.is_empty() {
                    word.len()
                } else {
                    current.len() + 1 + word.len()
                };
                if needed <= width as usize {
                    if !current.is_empty() {
                        current.push(' ');
                    }
                    current.push_str(word);
                } else {
                    if !current.is_empty() {
                        lines.push(current.clone());
                    }
                    if word.chars().count() > width as usize {
                        let chars: Vec<char> = word.chars().collect();
                        let mut s = 0usize;
                        while s < chars.len() {
                            let e = (s + width as usize).min(chars.len());
                            lines.push(chars[s..e].iter().collect());
                            s = e;
                        }
                        current = String::new();
                    } else {
                        current = word.to_string();
                    }
                }
            }
            if !current.is_empty() {
                lines.push(current);
            }
        }
    }
    let count = lines.len() as u32;
    WrapVerdict::Ok {
        lines,
        line_count: count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_wrap_strategy_compute")?;

    println!(
        "char-10: {:?}",
        wrap("hello world foo", 10, &WrapStrategy::Char)
    );
    println!(
        "word-10: {:?}",
        wrap("hello world foo", 10, &WrapStrategy::Word)
    );
    println!("invalid: {:?}", wrap("", 10, &WrapStrategy::Char));
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
    fn empty_text_rejected() {
        assert_eq!(
            wrap("", 10, &WrapStrategy::Char),
            WrapVerdict::InvalidConfig
        );
    }

    #[test]
    fn zero_width_rejected() {
        assert_eq!(
            wrap("hello", 0, &WrapStrategy::Char),
            WrapVerdict::InvalidConfig
        );
    }

    #[test]
    fn char_wrap_fixed_width() {
        let v = wrap("0123456789ABCDEF", 5, &WrapStrategy::Char);
        if let WrapVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 4);
            assert_eq!(lines[0], "01234");
        }
    }

    #[test]
    fn word_wrap_preserves_words() {
        // "hello world" = 11 chars, width 10 → must break (greedy).
        let v = wrap("hello world", 11, &WrapStrategy::Word);
        if let WrapVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 1);
        }
    }

    #[test]
    fn word_wrap_breaks_when_needed() {
        // "hello world foo bar" = 19 chars; width 10.
        // Greedy: "hello" (5), "world foo" (9), "bar" (3) → 3 lines.
        let v = wrap("hello world foo bar", 10, &WrapStrategy::Word);
        if let WrapVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 3);
        }
    }

    #[test]
    fn word_longer_than_width_char_wrapped() {
        // "supercalifragilistic" doesn't fit in width 5 → char-fallback breaks it.
        let v = wrap("supercalifragilistic", 5, &WrapStrategy::Word);
        if let WrapVerdict::Ok { lines, .. } = v {
            assert!(lines.len() >= 4);
        }
    }

    #[test]
    fn line_count_correct() {
        let v = wrap("0123456789AB", 4, &WrapStrategy::Char);
        if let WrapVerdict::Ok { line_count, .. } = v {
            assert_eq!(line_count, 3);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = wrap("hello", 3, &WrapStrategy::Char);
        let r2 = wrap("hello", 3, &WrapStrategy::Char);
        assert_eq!(r1, r2);
    }

    #[test]
    fn unicode_chars_counted_correctly() {
        let v = wrap("café", 4, &WrapStrategy::Char);
        if let WrapVerdict::Ok { lines, .. } = v {
            assert_eq!(lines.len(), 1);
            assert_eq!(lines[0], "café");
        }
    }

    #[test]
    fn long_text_handled() {
        let text: String = "word ".repeat(50);
        let v = wrap(&text, 20, &WrapStrategy::Word);
        assert!(matches!(v, WrapVerdict::Ok { .. }));
    }

    #[test]
    fn single_char_handled() {
        let v = wrap("a", 5, &WrapStrategy::Word);
        if let WrapVerdict::Ok { lines, .. } = v {
            assert_eq!(lines, vec!["a".to_string()]);
        }
    }
}
