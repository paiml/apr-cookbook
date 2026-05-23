//! # TUI Brace Match Highlighter
//!
//! Find the matching brace position for a brace at the given byte
//! index. Supports `()`, `[]`, `{}`. Returns matched position or
//! `Unbalanced` verdict.
//!
//! Demonstrates the **TUI.131** recipe for PMAT-203 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim `matchparen.vim`; emacs `show-paren-mode`.
//!
//! Run with: cargo run --example tui_braces_match_highlight
//!
//! Added by PMAT-203 (catalog 1450→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BraceVerdict {
    Matched { partner_pos: u32 },
    Unbalanced,
    InvalidConfig,
}

pub fn match_brace(line: &str, pos: u32) -> BraceVerdict {
    let bytes = line.as_bytes();
    if (pos as usize) >= bytes.len() {
        return BraceVerdict::InvalidConfig;
    }
    let opener = bytes[pos as usize];
    let (forward, target) = match opener {
        b'(' => (true, b')'),
        b'[' => (true, b']'),
        b'{' => (true, b'}'),
        b')' => (false, b'('),
        b']' => (false, b'['),
        b'}' => (false, b'{'),
        _ => return BraceVerdict::InvalidConfig,
    };
    let mut depth = 1i32;
    if forward {
        let mut i = pos as usize + 1;
        while i < bytes.len() {
            if bytes[i] == opener {
                depth += 1;
            } else if bytes[i] == target {
                depth -= 1;
                if depth == 0 {
                    return BraceVerdict::Matched {
                        partner_pos: i as u32,
                    };
                }
            }
            i += 1;
        }
    } else {
        let mut i = pos as i64 - 1;
        while i >= 0 {
            let b = bytes[i as usize];
            if b == opener {
                depth += 1;
            } else if b == target {
                depth -= 1;
                if depth == 0 {
                    return BraceVerdict::Matched {
                        partner_pos: i as u32,
                    };
                }
            }
            i -= 1;
        }
    }
    BraceVerdict::Unbalanced
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_braces_match_highlight")?;

    println!("paren: {:?}", match_brace("foo()", 3));
    println!("nested: {:?}", match_brace("[a[b]c]", 0));
    println!("invalid: {:?}", match_brace("x", 5));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn paren_forward_match() {
        assert_eq!(
            match_brace("()", 0),
            BraceVerdict::Matched { partner_pos: 1 }
        );
    }

    #[test]
    fn paren_backward_match() {
        assert_eq!(
            match_brace("()", 1),
            BraceVerdict::Matched { partner_pos: 0 }
        );
    }

    #[test]
    fn bracket_match() {
        assert_eq!(
            match_brace("[a]", 0),
            BraceVerdict::Matched { partner_pos: 2 }
        );
    }

    #[test]
    fn brace_match() {
        assert_eq!(
            match_brace("{x}", 0),
            BraceVerdict::Matched { partner_pos: 2 }
        );
    }

    #[test]
    fn nested_paren_outer_match() {
        assert_eq!(
            match_brace("((x))", 0),
            BraceVerdict::Matched { partner_pos: 4 }
        );
    }

    #[test]
    fn nested_paren_inner_match() {
        assert_eq!(
            match_brace("((x))", 1),
            BraceVerdict::Matched { partner_pos: 3 }
        );
    }

    #[test]
    fn unbalanced_returns_unbalanced() {
        assert_eq!(match_brace("(", 0), BraceVerdict::Unbalanced);
    }

    #[test]
    fn out_of_bounds_invalid() {
        assert_eq!(match_brace("x", 10), BraceVerdict::InvalidConfig);
    }

    #[test]
    fn non_brace_char_invalid() {
        assert_eq!(match_brace("foo", 0), BraceVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = match_brace("()", 0);
        let r2 = match_brace("()", 0);
        assert_eq!(r1, r2);
    }

    #[test]
    fn mixed_brace_types_isolate() {
        // Square bracket at 0 ignores parens
        assert_eq!(
            match_brace("[(x)]", 0),
            BraceVerdict::Matched { partner_pos: 4 }
        );
    }

    #[test]
    fn deeply_nested() {
        assert_eq!(
            match_brace("{{{}}}", 0),
            BraceVerdict::Matched { partner_pos: 5 }
        );
    }
}
