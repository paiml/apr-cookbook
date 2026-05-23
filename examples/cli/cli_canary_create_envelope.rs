//! # apr canary create — Test Envelope
//!
//! `apr canary create <NAME>` records a regression-test snapshot: the
//! model identifier, the prompt, the expected first-N tokens, and a
//! tolerance band. This recipe builds the snapshot envelope and asserts
//! the contract: name must be filename-safe, prompt must be non-empty,
//! expected_tokens must contain at least one entry, tolerance > 0.
//!
//! Demonstrates the **CANARY.4** recipe for PMAT-100 (apr canary create coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CANARY-001 + regression-snapshot convention
//!
//! Run with: cargo run --example cli_canary_create_envelope
//!
//! Added by PMAT-100 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq)]
pub struct CanarySnapshot {
    pub name: String,
    pub model: String,
    pub prompt: String,
    pub expected_tokens: Vec<u32>,
    pub tolerance: f64,
}

#[derive(Debug, PartialEq)]
pub enum CanaryVerdict {
    Ok,
    NameNotFilenameSafe,
    EmptyPrompt,
    EmptyExpected,
    TolerancePositive,
}

pub fn validate_snapshot(s: &CanarySnapshot) -> CanaryVerdict {
    if s.name.is_empty()
        || s.name
            .chars()
            .any(|c| !c.is_ascii_alphanumeric() && c != '-' && c != '_')
    {
        return CanaryVerdict::NameNotFilenameSafe;
    }
    if s.prompt.is_empty() {
        return CanaryVerdict::EmptyPrompt;
    }
    if s.expected_tokens.is_empty() {
        return CanaryVerdict::EmptyExpected;
    }
    if !s.tolerance.is_finite() || s.tolerance <= 0.0 {
        return CanaryVerdict::TolerancePositive;
    }
    CanaryVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_canary_create_envelope")?;

    let cases: &[(&str, CanarySnapshot)] = &[
        (
            "happy",
            CanarySnapshot {
                name: "math-2plus2".into(),
                model: "qwen2.5-coder-1.5b".into(),
                prompt: "2+2=".into(),
                expected_tokens: vec![19, 4],
                tolerance: 0.001,
            },
        ),
        (
            "name with slash",
            CanarySnapshot {
                name: "math/2plus2".into(),
                model: "m".into(),
                prompt: "p".into(),
                expected_tokens: vec![1],
                tolerance: 0.1,
            },
        ),
        (
            "empty prompt",
            CanarySnapshot {
                name: "ok".into(),
                model: "m".into(),
                prompt: String::new(),
                expected_tokens: vec![1],
                tolerance: 0.1,
            },
        ),
        (
            "zero tolerance",
            CanarySnapshot {
                name: "ok".into(),
                model: "m".into(),
                prompt: "p".into(),
                expected_tokens: vec![1],
                tolerance: 0.0,
            },
        ),
    ];

    for (label, s) in cases {
        println!("{label:>16}  →  {:?}", validate_snapshot(s));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn happy() -> CanarySnapshot {
        CanarySnapshot {
            name: "ok".into(),
            model: "m".into(),
            prompt: "p".into(),
            expected_tokens: vec![1, 2, 3],
            tolerance: 0.1,
        }
    }

    #[test]
    fn envelope_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_snapshot_passes() {
        assert_eq!(validate_snapshot(&happy()), CanaryVerdict::Ok);
    }

    #[test]
    fn slash_in_name_rejected() {
        // Snapshot names become filenames; slashes break filesystem mapping.
        let mut s = happy();
        s.name = "math/2plus2".into();
        assert_eq!(validate_snapshot(&s), CanaryVerdict::NameNotFilenameSafe);
    }

    #[test]
    fn whitespace_in_name_rejected() {
        let mut s = happy();
        s.name = "math 2plus2".into();
        assert_eq!(validate_snapshot(&s), CanaryVerdict::NameNotFilenameSafe);
    }

    #[test]
    fn underscore_and_hyphen_allowed_in_name() {
        let mut s = happy();
        s.name = "math-2plus2_v1".into();
        assert_eq!(validate_snapshot(&s), CanaryVerdict::Ok);
    }

    #[test]
    fn empty_expected_tokens_rejected() {
        let mut s = happy();
        s.expected_tokens = vec![];
        assert_eq!(validate_snapshot(&s), CanaryVerdict::EmptyExpected);
    }

    #[test]
    fn nan_tolerance_rejected() {
        let mut s = happy();
        s.tolerance = f64::NAN;
        assert_eq!(validate_snapshot(&s), CanaryVerdict::TolerancePositive);
    }

    #[test]
    fn negative_tolerance_rejected() {
        let mut s = happy();
        s.tolerance = -0.1;
        assert_eq!(validate_snapshot(&s), CanaryVerdict::TolerancePositive);
    }
}
