//! # TUI Trailing Whitespace Highlighter
//!
//! Detect lines containing trailing whitespace (spaces or tabs at end
//! of line, before newline). Returns sorted line numbers and total
//! offending count.
//!
//! Demonstrates the **TUI.134** recipe for PMAT-204 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim `:set list` shows trailing whitespace; rustfmt
//!  `trim_trailing_whitespace` rule; PEP-8 W291.
//!
//! Run with: cargo run --example tui_trailing_ws_highlight
//!
//! Added by PMAT-204 (catalog 1459→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TrailingWsVerdict {
    Ok {
        offending_lines: Vec<u32>,
        offending_count: u32,
    },
    InvalidConfig,
}

pub fn check(buffer: &str) -> TrailingWsVerdict {
    if buffer.is_empty() {
        return TrailingWsVerdict::InvalidConfig;
    }
    let mut offenders: Vec<u32> = Vec::new();
    for (i, line) in buffer.split('\n').enumerate() {
        if line.is_empty() {
            continue;
        }
        let last = line.chars().last();
        if matches!(last, Some(' ' | '\t')) {
            offenders.push((i as u32) + 1);
        }
    }
    let count = offenders.len() as u32;
    TrailingWsVerdict::Ok {
        offending_lines: offenders,
        offending_count: count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_trailing_ws_highlight")?;

    let buf = "fn foo() {  \n    let x = 1;\n}\t\n";
    println!("check: {:?}", check(buf));
    println!("invalid: {:?}", check(""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_trailing_ws_no_offenders() {
        let v = check("clean line\nanother clean");
        if let TrailingWsVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert!(offending_lines.is_empty());
        }
    }

    #[test]
    fn trailing_space_flagged() {
        let v = check("with space  ");
        if let TrailingWsVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert_eq!(offending_lines, vec![1]);
        }
    }

    #[test]
    fn trailing_tab_flagged() {
        let v = check("with tab\t");
        if let TrailingWsVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert_eq!(offending_lines, vec![1]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(""), TrailingWsVerdict::InvalidConfig);
    }

    #[test]
    fn multiple_offenders() {
        let v = check("ok\nbad \nok\nbad\t\n");
        if let TrailingWsVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert_eq!(offending_lines, vec![2, 4]);
        }
    }

    #[test]
    fn count_correct() {
        let v = check("a \nb\t\nc");
        if let TrailingWsVerdict::Ok {
            offending_count, ..
        } = v
        {
            assert_eq!(offending_count, 2);
        }
    }

    #[test]
    fn empty_lines_skipped() {
        let v = check("ok\n\nclean");
        if let TrailingWsVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert!(offending_lines.is_empty());
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check("a \nb");
        let r2 = check("a \nb");
        assert_eq!(r1, r2);
    }

    #[test]
    fn line_numbers_sorted() {
        let v = check("bad \nbad\t\nbad ");
        if let TrailingWsVerdict::Ok {
            offending_lines, ..
        } = v
        {
            for w in offending_lines.windows(2) {
                assert!(w[0] < w[1]);
            }
        }
    }

    #[test]
    fn middle_space_no_flag() {
        let v = check("foo bar baz");
        if let TrailingWsVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert!(offending_lines.is_empty());
        }
    }

    #[test]
    fn many_lines_handled() {
        let buf: String = (0..30).map(|_| "ok \n").collect();
        let v = check(&buf);
        if let TrailingWsVerdict::Ok {
            offending_count, ..
        } = v
        {
            assert_eq!(offending_count, 30);
        }
    }

    #[test]
    fn unicode_line_no_false_positive() {
        let v = check("café");
        if let TrailingWsVerdict::Ok {
            offending_lines, ..
        } = v
        {
            assert!(offending_lines.is_empty());
        }
    }
}
