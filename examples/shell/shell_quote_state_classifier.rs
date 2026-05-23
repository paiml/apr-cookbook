//! # Shell Quote State Classifier
//!
//! Tracks current quoting context while scanning a shell line: None,
//! SingleQuote (literal), DoubleQuote (variable expansion + escapes),
//! Backslash (one-char). Mismatched quotes are a parse error. This
//! recipe builds the state machine + final-state validator.
//!
//! Demonstrates the **SHELL.6** recipe for PMAT-126 (shell coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: POSIX 1003.1 §2.2 Quoting.
//!
//! Run with: cargo run --example shell_quote_state_classifier
//!
//! Added by PMAT-126 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuoteState {
    None,
    SingleQuote,
    DoubleQuote,
    Backslash,
    BackslashInDouble,
}

#[derive(Debug, PartialEq)]
pub enum LineVerdict {
    BalancedNeutral,
    EndsInUnclosedSingle,
    EndsInUnclosedDouble,
    EndsInDanglingBackslash,
}

pub fn final_state(line: &str) -> QuoteState {
    let mut state = QuoteState::None;
    for c in line.chars() {
        state = next_state(state, c);
    }
    state
}

fn next_state(state: QuoteState, c: char) -> QuoteState {
    match (state, c) {
        (QuoteState::None, '\'') => QuoteState::SingleQuote,
        (QuoteState::None, '"') => QuoteState::DoubleQuote,
        (QuoteState::None, '\\') => QuoteState::Backslash,
        (QuoteState::None, _) => QuoteState::None,

        (QuoteState::SingleQuote, '\'') => QuoteState::None,
        (QuoteState::SingleQuote, _) => QuoteState::SingleQuote,

        (QuoteState::DoubleQuote, '"') => QuoteState::None,
        (QuoteState::DoubleQuote, '\\') => QuoteState::BackslashInDouble,
        (QuoteState::DoubleQuote, _) => QuoteState::DoubleQuote,

        // After top-level backslash, consume next char then return to None.
        (QuoteState::Backslash, _) => QuoteState::None,
        // After backslash inside DQ, return to DoubleQuote so the closing
        // `"` is recognised. This handles `"escaped \" still inside"`.
        (QuoteState::BackslashInDouble, _) => QuoteState::DoubleQuote,
    }
}

pub fn classify(line: &str) -> LineVerdict {
    match final_state(line) {
        QuoteState::None => LineVerdict::BalancedNeutral,
        QuoteState::SingleQuote => LineVerdict::EndsInUnclosedSingle,
        QuoteState::DoubleQuote | QuoteState::BackslashInDouble => {
            LineVerdict::EndsInUnclosedDouble
        }
        QuoteState::Backslash => LineVerdict::EndsInDanglingBackslash,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("shell_quote_state_classifier")?;

    for line in [
        r#"echo "hello world""#,
        "echo 'literal'",
        r#"echo "unterminated"#,
        "echo 'unterminated",
        r#"echo \"escaped\""#,
        "trailing\\",
    ] {
        println!("{line:<35} → {:?}", classify(line));
    }
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
    fn balanced_double_quotes_neutral() {
        assert_eq!(classify(r#"echo "hello""#), LineVerdict::BalancedNeutral);
    }

    #[test]
    fn balanced_single_quotes_neutral() {
        assert_eq!(classify("echo 'hello'"), LineVerdict::BalancedNeutral);
    }

    #[test]
    fn unclosed_double_detected() {
        assert_eq!(
            classify(r#"echo "hello"#),
            LineVerdict::EndsInUnclosedDouble
        );
    }

    #[test]
    fn unclosed_single_detected() {
        assert_eq!(classify("echo 'hello"), LineVerdict::EndsInUnclosedSingle);
    }

    #[test]
    fn trailing_backslash_dangling() {
        assert_eq!(classify("trailing\\"), LineVerdict::EndsInDanglingBackslash);
    }

    #[test]
    fn nested_quote_inside_other() {
        // 'has "double" inside' is balanced single — embedded " is literal.
        assert_eq!(
            classify("'has \"double\" inside'"),
            LineVerdict::BalancedNeutral
        );
    }

    #[test]
    fn escaped_quote_inside_double() {
        // "escaped \" still inside" is balanced.
        assert_eq!(
            classify(r#""escaped \" still inside""#),
            LineVerdict::BalancedNeutral
        );
    }

    #[test]
    fn empty_line_neutral() {
        assert_eq!(classify(""), LineVerdict::BalancedNeutral);
    }

    #[test]
    fn final_state_helper_works_directly() {
        assert_eq!(final_state(r#""hello"#), QuoteState::DoubleQuote);
        assert_eq!(final_state("'hello"), QuoteState::SingleQuote);
        assert_eq!(final_state("\\"), QuoteState::Backslash);
        assert_eq!(final_state("plain"), QuoteState::None);
    }

    #[test]
    fn alternating_quotes_neutral() {
        let line = r#"echo "a" 'b' "c""#;
        assert_eq!(classify(line), LineVerdict::BalancedNeutral);
    }
}
