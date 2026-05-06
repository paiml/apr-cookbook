//! # apr experiment --hypothesis — Hypothesis Statement Envelope
//!
//! Each experiment registers a hypothesis: text + claimed direction +
//! significance level (α). Constraints: text non-empty + < 280 chars
//! (Twitter-tractable); α ∈ (0, 0.5) (sane statistics); direction
//! must match metric direction. This recipe builds the envelope.
//!
//! Demonstrates the **EXP.5** recipe for PMAT-118 (apr experiment coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender EXP-001 + Fisher 1925 (significance testing)
//!
//! Run with: cargo run --example cli_experiment_hypothesis_envelope
//!
//! Added by PMAT-118 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Increase,
    Decrease,
    Twosided,
}

#[derive(Debug, PartialEq)]
pub enum HypothesisVerdict {
    Ok,
    EmptyStatement,
    StatementTooLong { length: usize, max: usize },
    InvalidAlpha,
    DirectionMismatch,
}

const MAX_STATEMENT_LEN: usize = 280;

pub fn validate(
    statement: &str,
    alpha: f64,
    direction: Direction,
    metric_lower_is_better: bool,
) -> HypothesisVerdict {
    if statement.trim().is_empty() {
        return HypothesisVerdict::EmptyStatement;
    }
    if statement.len() > MAX_STATEMENT_LEN {
        return HypothesisVerdict::StatementTooLong {
            length: statement.len(),
            max: MAX_STATEMENT_LEN,
        };
    }
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 0.5 {
        return HypothesisVerdict::InvalidAlpha;
    }
    let direction_ok = matches!(
        (direction, metric_lower_is_better),
        (Direction::Decrease, true) | (Direction::Increase, false) | (Direction::Twosided, _)
    );
    if !direction_ok {
        return HypothesisVerdict::DirectionMismatch;
    }
    HypothesisVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_experiment_hypothesis_envelope")?;

    let cases = [
        (
            "FlashAttention v3 reduces decode latency by ≥ 20%",
            0.05,
            Direction::Decrease,
            true,
        ),
        ("", 0.05, Direction::Decrease, true),
        ("invalid alpha 0.6", 0.6, Direction::Decrease, true),
        (
            "wrong direction for accuracy",
            0.05,
            Direction::Decrease,
            false,
        ),
    ];
    for (s, a, d, l) in cases {
        let label = if s.len() > 30 { &s[..30] } else { s };
        println!("{label:>30}  →  {:?}", validate(s, a, d, l));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_hypothesis_passes() {
        let v = validate(
            "FlashAttention v3 reduces decode latency",
            0.05,
            Direction::Decrease,
            true,
        );
        assert_eq!(v, HypothesisVerdict::Ok);
    }

    #[test]
    fn empty_statement_rejected() {
        assert_eq!(
            validate("", 0.05, Direction::Decrease, true),
            HypothesisVerdict::EmptyStatement
        );
    }

    #[test]
    fn whitespace_only_rejected() {
        assert_eq!(
            validate("   ", 0.05, Direction::Decrease, true),
            HypothesisVerdict::EmptyStatement
        );
    }

    #[test]
    fn over_280_chars_rejected() {
        let long = "x".repeat(MAX_STATEMENT_LEN + 1);
        let v = validate(&long, 0.05, Direction::Decrease, true);
        assert!(matches!(v, HypothesisVerdict::StatementTooLong { .. }));
    }

    #[test]
    fn at_280_chars_passes() {
        let exactly_max = "x".repeat(MAX_STATEMENT_LEN);
        let v = validate(&exactly_max, 0.05, Direction::Decrease, true);
        assert_eq!(v, HypothesisVerdict::Ok);
    }

    #[test]
    fn invalid_alpha_rejected() {
        // alpha = 0.5 is the upper exclusive bound.
        assert_eq!(
            validate("test", 0.5, Direction::Decrease, true),
            HypothesisVerdict::InvalidAlpha
        );
        // negative
        assert_eq!(
            validate("test", -0.05, Direction::Decrease, true),
            HypothesisVerdict::InvalidAlpha
        );
        // zero
        assert_eq!(
            validate("test", 0.0, Direction::Decrease, true),
            HypothesisVerdict::InvalidAlpha
        );
    }

    #[test]
    fn nan_alpha_rejected() {
        assert_eq!(
            validate("test", f64::NAN, Direction::Decrease, true),
            HypothesisVerdict::InvalidAlpha
        );
    }

    #[test]
    fn direction_mismatch_rejected() {
        // Predicting decrease for an accuracy metric (higher is better).
        assert_eq!(
            validate("test", 0.05, Direction::Decrease, false),
            HypothesisVerdict::DirectionMismatch
        );
        // Predicting increase for a loss metric.
        assert_eq!(
            validate("test", 0.05, Direction::Increase, true),
            HypothesisVerdict::DirectionMismatch
        );
    }

    #[test]
    fn twosided_works_for_either_direction() {
        // Twosided is direction-agnostic.
        assert_eq!(
            validate("test", 0.05, Direction::Twosided, true),
            HypothesisVerdict::Ok
        );
        assert_eq!(
            validate("test", 0.05, Direction::Twosided, false),
            HypothesisVerdict::Ok
        );
    }
}
