//! # apr cbtop — Speculative Decoding Envelope
//!
//! `apr cbtop --speculative --speculation-k <K> --draft-model <PATH>`
//! benchmarks the target+draft pair using k-token speculation. This recipe
//! models the invocation envelope and asserts the constraint web: when
//! `--speculative` is set, `--draft-model` is REQUIRED, and `--speculation-k`
//! is bounded by the practical [1, 16] window per PAR-100/099.
//!
//! Demonstrates the **CBTOP.4** recipe for PMAT-094 (apr cbtop coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PAR-100 + Leviathan et al. (2023) speculative decoding
//!
//! Run with: cargo run --example cli_cbtop_speculative_decoding_envelope
//!
//! Added by PMAT-094 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Default, Clone)]
pub struct SpecConfig {
    pub speculative: bool,
    pub speculation_k: u32,
    pub draft_model: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum SpecVerdict {
    Ok,
    MissingDraftModel,
    KOutOfRange { k: u32 },
    SpeculativeFalseWithDraft, // user passed --draft-model but not --speculative
}

pub fn validate_spec(c: &SpecConfig) -> SpecVerdict {
    if !c.speculative && c.draft_model.is_some() {
        return SpecVerdict::SpeculativeFalseWithDraft;
    }
    if !c.speculative {
        return SpecVerdict::Ok;
    }
    if c.draft_model.is_none() {
        return SpecVerdict::MissingDraftModel;
    }
    if !(1..=16).contains(&c.speculation_k) {
        return SpecVerdict::KOutOfRange { k: c.speculation_k };
    }
    SpecVerdict::Ok
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_cbtop_speculative_decoding_envelope")?;

    let cases: &[(&str, SpecConfig)] = &[
        (
            "happy",
            SpecConfig {
                speculative: true,
                speculation_k: 4,
                draft_model: Some("draft.gguf".into()),
            },
        ),
        ("no spec", SpecConfig::default()),
        (
            "missing draft",
            SpecConfig {
                speculative: true,
                speculation_k: 4,
                draft_model: None,
            },
        ),
        (
            "k too high",
            SpecConfig {
                speculative: true,
                speculation_k: 32,
                draft_model: Some("draft.gguf".into()),
            },
        ),
        (
            "draft without spec",
            SpecConfig {
                speculative: false,
                speculation_k: 4,
                draft_model: Some("draft.gguf".into()),
            },
        ),
    ];

    println!("=== Recipe: cli_cbtop_speculative_decoding_envelope ===");
    for (label, c) in cases {
        println!("{label:>20}  →  {:?}", validate_spec(c));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn happy_config_is_ok() {
        let c = SpecConfig {
            speculative: true,
            speculation_k: 4,
            draft_model: Some("draft.gguf".into()),
        };
        assert_eq!(validate_spec(&c), SpecVerdict::Ok);
    }

    #[test]
    fn speculative_without_draft_model_fails() {
        let c = SpecConfig {
            speculative: true,
            speculation_k: 4,
            draft_model: None,
        };
        assert_eq!(validate_spec(&c), SpecVerdict::MissingDraftModel);
    }

    #[test]
    fn k_above_16_fails() {
        let c = SpecConfig {
            speculative: true,
            speculation_k: 32,
            draft_model: Some("draft.gguf".into()),
        };
        assert_eq!(validate_spec(&c), SpecVerdict::KOutOfRange { k: 32 });
    }

    #[test]
    fn k_zero_fails() {
        let c = SpecConfig {
            speculative: true,
            speculation_k: 0,
            draft_model: Some("draft.gguf".into()),
        };
        assert_eq!(validate_spec(&c), SpecVerdict::KOutOfRange { k: 0 });
    }

    #[test]
    fn draft_model_with_speculative_false_is_warned() {
        // User probably forgot --speculative; surface the inconsistency.
        let c = SpecConfig {
            speculative: false,
            speculation_k: 4,
            draft_model: Some("draft.gguf".into()),
        };
        assert_eq!(validate_spec(&c), SpecVerdict::SpeculativeFalseWithDraft);
    }

    #[test]
    fn no_speculation_default_is_ok() {
        assert_eq!(validate_spec(&SpecConfig::default()), SpecVerdict::Ok);
    }
}
