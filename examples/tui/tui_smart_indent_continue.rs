//! # TUI Smart Indent Continuation
//!
//! Compute next-line indent when user presses Enter — preserves
//! current indent level and adds one tab/space-block when the prior
//! line ends with an opening brace `{`. Returns indent string.
//!
//! Demonstrates the **TUI.130** recipe for PMAT-203 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim `smartindent` and `cindent`; emacs `electric-indent`
//!  conventions.
//!
//! Run with: cargo run --example tui_smart_indent_continue
//!
//! Added by PMAT-203 (catalog 1450→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum IndentVerdict {
    Ok {
        next_indent: String,
        indent_units: u32,
    },
    InvalidConfig,
}

pub fn next_indent(prev_line: &str, tab_width: u32) -> IndentVerdict {
    if tab_width == 0 || tab_width > 16 {
        return IndentVerdict::InvalidConfig;
    }
    let leading_spaces: u32 = prev_line.chars().take_while(|c| *c == ' ').count() as u32;
    let trimmed = prev_line.trim_end();
    let extra = if trimmed.ends_with('{') { tab_width } else { 0 };
    let total = leading_spaces + extra;
    IndentVerdict::Ok {
        next_indent: " ".repeat(total as usize),
        indent_units: total,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_smart_indent_continue")?;

    println!("after fn: {:?}", next_indent("fn foo() {", 4));
    println!("plain: {:?}", next_indent("    let x = 1;", 4));
    println!("invalid: {:?}", next_indent("hi", 0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indenter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn indent_preserved_no_brace() {
        let v = next_indent("    let x = 1;", 4);
        if let IndentVerdict::Ok { indent_units, .. } = v {
            assert_eq!(indent_units, 4);
        }
    }

    #[test]
    fn brace_adds_extra_indent() {
        let v = next_indent("fn foo() {", 4);
        if let IndentVerdict::Ok { indent_units, .. } = v {
            assert_eq!(indent_units, 4);
        }
    }

    #[test]
    fn brace_with_existing_indent() {
        let v = next_indent("    if cond {", 4);
        if let IndentVerdict::Ok { indent_units, .. } = v {
            assert_eq!(indent_units, 8);
        }
    }

    #[test]
    fn empty_line_zero_indent() {
        let v = next_indent("", 4);
        if let IndentVerdict::Ok { indent_units, .. } = v {
            assert_eq!(indent_units, 0);
        }
    }

    #[test]
    fn invalid_zero_tab_width() {
        assert_eq!(next_indent("x", 0), IndentVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_huge_tab_width() {
        assert_eq!(next_indent("x", 100), IndentVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = next_indent("  x", 2);
        let r2 = next_indent("  x", 2);
        assert_eq!(r1, r2);
    }

    #[test]
    fn next_indent_string_correct_length() {
        let v = next_indent("  ", 2);
        if let IndentVerdict::Ok { next_indent, .. } = v {
            assert_eq!(next_indent.len(), 2);
        }
    }

    #[test]
    fn trailing_whitespace_after_brace_handled() {
        let v = next_indent("fn foo() {   ", 4);
        if let IndentVerdict::Ok { indent_units, .. } = v {
            // trim_end strips trailing spaces; brace still detected
            assert_eq!(indent_units, 4);
        }
    }

    #[test]
    fn brace_in_middle_no_extra() {
        // Brace not at end of line → no smart indent
        let v = next_indent("if x { y }", 4);
        if let IndentVerdict::Ok { indent_units, .. } = v {
            assert_eq!(indent_units, 0);
        }
    }

    #[test]
    fn tab_width_two() {
        let v = next_indent("if x {", 2);
        if let IndentVerdict::Ok { indent_units, .. } = v {
            assert_eq!(indent_units, 2);
        }
    }

    #[test]
    fn deeply_indented_brace() {
        let v = next_indent("            block {", 4);
        if let IndentVerdict::Ok { indent_units, .. } = v {
            assert_eq!(indent_units, 16);
        }
    }
}
