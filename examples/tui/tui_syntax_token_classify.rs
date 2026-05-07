//! # TUI Syntax Token Classify
//!
//! Classify a single token from a code line into one of:
//! Keyword, String, Comment, Number, Identifier. Used by syntax
//! highlighters for color-class assignment.
//!
//! Demonstrates the **TUI.157** recipe for PMAT-212 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: vim syntax-region groups; tree-sitter highlight.scm
//!  capture-name conventions.
//!
//! Run with: cargo run --example tui_syntax_token_classify
//!
//! Added by PMAT-212 (catalog 1531→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TokenKind {
    Keyword,
    StringLit,
    Comment,
    Number,
    Identifier,
}

#[derive(Debug, PartialEq)]
pub enum ClassifyVerdict {
    Ok {
        kind: TokenKind,
        color_class: String,
    },
    InvalidConfig,
}

pub fn classify(token: &str) -> ClassifyVerdict {
    if token.is_empty() {
        return ClassifyVerdict::InvalidConfig;
    }
    const KEYWORDS: &[&str] = &[
        "fn", "let", "if", "else", "match", "for", "while", "return", "struct", "enum", "impl",
        "trait", "pub", "mod", "use",
    ];
    let kind = if KEYWORDS.contains(&token) {
        TokenKind::Keyword
    } else if (token.starts_with('"') && token.ends_with('"') && token.len() >= 2)
        || (token.starts_with('\'') && token.ends_with('\'') && token.len() >= 2)
    {
        TokenKind::StringLit
    } else if token.starts_with("//") || token.starts_with("/*") {
        TokenKind::Comment
    } else if token.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        TokenKind::Number
    } else {
        TokenKind::Identifier
    };
    let color_class = match kind {
        TokenKind::Keyword => "fg-blue",
        TokenKind::StringLit => "fg-green",
        TokenKind::Comment => "fg-gray",
        TokenKind::Number => "fg-cyan",
        TokenKind::Identifier => "fg-default",
    }
    .to_string();
    ClassifyVerdict::Ok { kind, color_class }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tui_syntax_token_classify")?;

    println!("fn: {:?}", classify("fn"));
    println!("\"hi\": {:?}", classify("\"hi\""));
    println!("// note: {:?}", classify("// note"));
    println!("42: {:?}", classify("42"));
    println!("invalid: {:?}", classify(""));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn keyword_classified() {
        let v = classify("fn");
        if let ClassifyVerdict::Ok { kind, .. } = v {
            assert_eq!(kind, TokenKind::Keyword);
        }
    }

    #[test]
    fn string_classified() {
        let v = classify("\"hello\"");
        if let ClassifyVerdict::Ok { kind, .. } = v {
            assert_eq!(kind, TokenKind::StringLit);
        }
    }

    #[test]
    fn comment_classified() {
        let v = classify("//");
        if let ClassifyVerdict::Ok { kind, .. } = v {
            assert_eq!(kind, TokenKind::Comment);
        }
    }

    #[test]
    fn number_classified() {
        let v = classify("42");
        if let ClassifyVerdict::Ok { kind, .. } = v {
            assert_eq!(kind, TokenKind::Number);
        }
    }

    #[test]
    fn identifier_default() {
        let v = classify("foo_bar");
        if let ClassifyVerdict::Ok { kind, .. } = v {
            assert_eq!(kind, TokenKind::Identifier);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(classify(""), ClassifyVerdict::InvalidConfig);
    }

    #[test]
    fn color_class_for_keyword_blue() {
        let v = classify("let");
        if let ClassifyVerdict::Ok { color_class, .. } = v {
            assert_eq!(color_class, "fg-blue");
        }
    }

    #[test]
    fn color_class_for_string_green() {
        let v = classify("\"abc\"");
        if let ClassifyVerdict::Ok { color_class, .. } = v {
            assert_eq!(color_class, "fg-green");
        }
    }

    #[test]
    fn deterministic() {
        let r1 = classify("fn");
        let r2 = classify("fn");
        assert_eq!(r1, r2);
    }

    #[test]
    fn block_comment_classified() {
        let v = classify("/*");
        if let ClassifyVerdict::Ok { kind, .. } = v {
            assert_eq!(kind, TokenKind::Comment);
        }
    }

    #[test]
    fn single_quote_string_classified() {
        let v = classify("'a'");
        if let ClassifyVerdict::Ok { kind, .. } = v {
            assert_eq!(kind, TokenKind::StringLit);
        }
    }

    #[test]
    fn unicode_identifier_supported() {
        let v = classify("café");
        if let ClassifyVerdict::Ok { kind, .. } = v {
            assert_eq!(kind, TokenKind::Identifier);
        }
    }
}
