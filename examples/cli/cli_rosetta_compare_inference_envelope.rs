//! # apr rosetta compare-inference — Invocation Envelope
//!
//! `apr rosetta compare-inference <MODEL_A> <MODEL_B> --prompt <P>
//! --max-tokens <N> --temperature <T> --tolerance <TOL>` runs the same
//! prompt through both models and asserts logit divergence stays under
//! tolerance. This recipe models the invocation envelope and asserts the
//! constraint web: temperature must be in [0, 2], max-tokens ≥ 1,
//! tolerance > 0, and same-file-twice is a special "self-comparison" case.
//!
//! Demonstrates the **ROSETTA-CMP.1** recipe for PMAT-096 (apr rosetta compare-inference coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-114 + greedy/sampling decoding parity
//!
//! Run with: cargo run --example cli_rosetta_compare_inference_envelope
//!
//! Added by PMAT-096 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone)]
pub struct CompareInvocation {
    pub model_a: String,
    pub model_b: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f64,
    pub tolerance: f64,
}

#[derive(Debug, PartialEq)]
pub enum CompareVerdict {
    Ok { same_file: bool },
    EmptyPath,
    EmptyPrompt,
    InvalidMaxTokens,
    InvalidTemperature,
    InvalidTolerance,
}

pub fn validate_invocation(inv: &CompareInvocation) -> CompareVerdict {
    if inv.model_a.is_empty() || inv.model_b.is_empty() {
        return CompareVerdict::EmptyPath;
    }
    if inv.prompt.is_empty() {
        return CompareVerdict::EmptyPrompt;
    }
    if inv.max_tokens == 0 {
        return CompareVerdict::InvalidMaxTokens;
    }
    if !(0.0..=2.0).contains(&inv.temperature) || !inv.temperature.is_finite() {
        return CompareVerdict::InvalidTemperature;
    }
    if inv.tolerance <= 0.0 || !inv.tolerance.is_finite() {
        return CompareVerdict::InvalidTolerance;
    }
    CompareVerdict::Ok {
        same_file: inv.model_a == inv.model_b,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_compare_inference_envelope")?;

    let cases: &[(&str, CompareInvocation)] = &[
        (
            "happy",
            CompareInvocation {
                model_a: "ref.gguf".into(),
                model_b: "test.apr".into(),
                prompt: "2+2=".into(),
                max_tokens: 5,
                temperature: 0.0,
                tolerance: 0.1,
            },
        ),
        (
            "self-cmp",
            CompareInvocation {
                model_a: "model.apr".into(),
                model_b: "model.apr".into(),
                prompt: "hi".into(),
                max_tokens: 3,
                temperature: 0.0,
                tolerance: 0.001,
            },
        ),
        (
            "temp too high",
            CompareInvocation {
                model_a: "a".into(),
                model_b: "b".into(),
                prompt: "hi".into(),
                max_tokens: 3,
                temperature: 5.0,
                tolerance: 0.1,
            },
        ),
        (
            "tol zero",
            CompareInvocation {
                model_a: "a".into(),
                model_b: "b".into(),
                prompt: "hi".into(),
                max_tokens: 3,
                temperature: 0.0,
                tolerance: 0.0,
            },
        ),
    ];

    for (label, inv) in cases {
        println!("{label:>16}  →  {:?}", validate_invocation(inv));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn happy() -> CompareInvocation {
        CompareInvocation {
            model_a: "a.gguf".into(),
            model_b: "b.apr".into(),
            prompt: "hello".into(),
            max_tokens: 5,
            temperature: 0.0,
            tolerance: 0.1,
        }
    }

    #[test]
    fn envelope_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_passes_with_same_file_false() {
        assert_eq!(
            validate_invocation(&happy()),
            CompareVerdict::Ok { same_file: false }
        );
    }

    #[test]
    fn self_compare_flagged_in_verdict() {
        // Useful "is the binary deterministic?" smoke test — a valid use case,
        // not an error, but operator wants to know the same file was used twice.
        let mut inv = happy();
        inv.model_a = "x.apr".into();
        inv.model_b = "x.apr".into();
        assert_eq!(
            validate_invocation(&inv),
            CompareVerdict::Ok { same_file: true }
        );
    }

    #[test]
    fn empty_prompt_rejected() {
        let mut inv = happy();
        inv.prompt = String::new();
        assert_eq!(validate_invocation(&inv), CompareVerdict::EmptyPrompt);
    }

    #[test]
    fn temperature_above_two_rejected() {
        let mut inv = happy();
        inv.temperature = 2.5;
        assert_eq!(
            validate_invocation(&inv),
            CompareVerdict::InvalidTemperature
        );
    }

    #[test]
    fn nan_temperature_rejected() {
        let mut inv = happy();
        inv.temperature = f64::NAN;
        assert_eq!(
            validate_invocation(&inv),
            CompareVerdict::InvalidTemperature
        );
    }

    #[test]
    fn zero_or_negative_tolerance_rejected() {
        let mut inv = happy();
        inv.tolerance = 0.0;
        assert_eq!(validate_invocation(&inv), CompareVerdict::InvalidTolerance);
        inv.tolerance = -0.1;
        assert_eq!(validate_invocation(&inv), CompareVerdict::InvalidTolerance);
    }
}
