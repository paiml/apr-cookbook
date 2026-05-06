//! # TUI Markdown Quote Block Render
//!
//! Render a markdown blockquote (`> ...` lines, possibly nested
//! `> > ...`) with a left-margin pipe indicator. Returns each line
//! as `"|".repeat(depth) + " " + content`.
//!
//! Demonstrates the **TUI.61** recipe for PMAT-180 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: CommonMark §5.1 (Block quotes); RFC 1855 §2.1.1.
//!
//! Run with: cargo run --example tui_quote_block_render
//!
//! Added by PMAT-180 (catalog 1243→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum QuoteVerdict {
    Ok { rendered: Vec<String> },
    InvalidConfig,
}

pub fn render(lines: &[&str]) -> QuoteVerdict {
    if lines.is_empty() {
        return QuoteVerdict::InvalidConfig;
    }
    let mut rendered: Vec<String> = Vec::with_capacity(lines.len());
    for line in lines {
        let mut depth = 0u32;
        let mut rest: &str = line;
        loop {
            let trimmed = rest.trim_start();
            if let Some(after) = trimmed.strip_prefix('>') {
                depth += 1;
                rest = after.trim_start();
            } else {
                rest = trimmed;
                break;
            }
        }
        if depth == 0 {
            rendered.push((*line).to_string());
        } else {
            let mut out = String::new();
            for _ in 0..depth {
                out.push('|');
            }
            out.push(' ');
            out.push_str(rest);
            rendered.push(out);
        }
    }
    QuoteVerdict::Ok { rendered }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_quote_block_render")?;

    let lines = [
        "Plain text",
        "> single quote",
        "> > nested quote",
        "> > > triple",
    ];
    println!("rendered: {:?}", render(&lines));
    println!("invalid: {:?}", render(&[]));
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
    fn plain_line_unchanged() {
        let v = render(&["plain"]);
        if let QuoteVerdict::Ok { rendered } = v {
            assert_eq!(rendered, vec!["plain".to_string()]);
        }
    }

    #[test]
    fn single_quote_one_pipe() {
        let v = render(&["> hello"]);
        if let QuoteVerdict::Ok { rendered } = v {
            assert_eq!(rendered, vec!["| hello".to_string()]);
        }
    }

    #[test]
    fn nested_quote_two_pipes() {
        let v = render(&["> > nested"]);
        if let QuoteVerdict::Ok { rendered } = v {
            assert_eq!(rendered, vec!["|| nested".to_string()]);
        }
    }

    #[test]
    fn triple_nested_three_pipes() {
        let v = render(&["> > > triple"]);
        if let QuoteVerdict::Ok { rendered } = v {
            assert_eq!(rendered, vec!["||| triple".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(render(&[]), QuoteVerdict::InvalidConfig);
    }

    #[test]
    fn line_count_preserved() {
        let v = render(&["a", "b", "c"]);
        if let QuoteVerdict::Ok { rendered } = v {
            assert_eq!(rendered.len(), 3);
        }
    }

    #[test]
    fn leading_whitespace_tolerated() {
        let v = render(&["  > hi"]);
        if let QuoteVerdict::Ok { rendered } = v {
            assert_eq!(rendered, vec!["| hi".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = render(&["> a", "> > b"]);
        let r2 = render(&["> a", "> > b"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn empty_after_marker_yields_pipe_space() {
        let v = render(&[">"]);
        if let QuoteVerdict::Ok { rendered } = v {
            assert_eq!(rendered, vec!["| ".to_string()]);
        }
    }

    #[test]
    fn unicode_content_preserved() {
        let v = render(&["> café"]);
        if let QuoteVerdict::Ok { rendered } = v {
            assert_eq!(rendered, vec!["| café".to_string()]);
        }
    }

    #[test]
    fn mixed_lines_preserve_structure() {
        let v = render(&["plain", "> q", "> > nq", "back to plain"]);
        if let QuoteVerdict::Ok { rendered } = v {
            assert_eq!(rendered.len(), 4);
            assert_eq!(rendered[0], "plain");
            assert_eq!(rendered[3], "back to plain");
        }
    }
}
