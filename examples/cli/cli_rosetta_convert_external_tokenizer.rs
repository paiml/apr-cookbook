//! # apr rosetta convert — `--tokenizer` for Weights-Only Models
//!
//! `apr rosetta convert <SOURCE> <TARGET> --tokenizer <FILE>` supplies an
//! external `tokenizer.json` for weights-only models (PMAT-232) that lack
//! embedded tokenizer state. This recipe asserts the policy: the
//! tokenizer is REQUIRED only when the source format omits the tokenizer
//! AND the target format expects it (e.g., SafeTensors → APR).
//!
//! Demonstrates the **ROSETTA-CONVERT.3** recipe for PMAT-098 (apr rosetta convert coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-232
//!
//! Run with: cargo run --example cli_rosetta_convert_external_tokenizer
//!
//! Added by PMAT-098 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceCarriesTokenizer {
    Yes,
    No,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetNeedsTokenizer {
    Yes,
    No,
}

#[derive(Debug, PartialEq)]
pub enum TokenizerVerdict {
    Ok,
    MissingExternal,
    Superfluous,
}

pub fn validate_tokenizer(
    source: SourceCarriesTokenizer,
    target: TargetNeedsTokenizer,
    external_provided: bool,
) -> TokenizerVerdict {
    match (source, target, external_provided) {
        // Target doesn't need tokenizer at all → vacuous OK regardless of source/external.
        (_, TargetNeedsTokenizer::No, _) => TokenizerVerdict::Ok,
        // Source has tokenizer, target wants tokenizer, no external → OK (uses source's).
        (SourceCarriesTokenizer::Yes, TargetNeedsTokenizer::Yes, false) => TokenizerVerdict::Ok,
        // External provided when source already has tokenizer → superfluous (warn).
        (SourceCarriesTokenizer::Yes, TargetNeedsTokenizer::Yes, true) => {
            TokenizerVerdict::Superfluous
        }
        // Source lacks tokenizer, target needs it, no external → REQUIRED.
        (SourceCarriesTokenizer::No, TargetNeedsTokenizer::Yes, false) => {
            TokenizerVerdict::MissingExternal
        }
        // Source lacks tokenizer, target needs it, external provided → OK.
        (SourceCarriesTokenizer::No, TargetNeedsTokenizer::Yes, true) => TokenizerVerdict::Ok,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_convert_external_tokenizer")?;

    let cases = [
        (
            "APR (has tok) → APR, no ext",
            SourceCarriesTokenizer::Yes,
            TargetNeedsTokenizer::Yes,
            false,
        ),
        (
            "APR → APR with ext (warn)",
            SourceCarriesTokenizer::Yes,
            TargetNeedsTokenizer::Yes,
            true,
        ),
        (
            "ST (no tok) → APR, no ext (BAD)",
            SourceCarriesTokenizer::No,
            TargetNeedsTokenizer::Yes,
            false,
        ),
        (
            "ST → APR with ext (OK)",
            SourceCarriesTokenizer::No,
            TargetNeedsTokenizer::Yes,
            true,
        ),
        (
            "X → format that doesn't need tok",
            SourceCarriesTokenizer::No,
            TargetNeedsTokenizer::No,
            false,
        ),
    ];

    for (label, src, tgt, ext) in cases {
        println!("{label:>40}  →  {:?}", validate_tokenizer(src, tgt, ext));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn source_with_tokenizer_passes_without_external() {
        // Common case: APR with tokenizer → APR.
        assert_eq!(
            validate_tokenizer(
                SourceCarriesTokenizer::Yes,
                TargetNeedsTokenizer::Yes,
                false
            ),
            TokenizerVerdict::Ok
        );
    }

    #[test]
    fn external_when_source_has_one_is_superfluous() {
        // Operator probably forgot to remove --tokenizer flag; warn rather
        // than silently override.
        assert_eq!(
            validate_tokenizer(SourceCarriesTokenizer::Yes, TargetNeedsTokenizer::Yes, true),
            TokenizerVerdict::Superfluous
        );
    }

    #[test]
    fn missing_external_when_required_rejected() {
        // SafeTensors → APR without --tokenizer must error out.
        assert_eq!(
            validate_tokenizer(SourceCarriesTokenizer::No, TargetNeedsTokenizer::Yes, false),
            TokenizerVerdict::MissingExternal
        );
    }

    #[test]
    fn external_when_required_passes() {
        assert_eq!(
            validate_tokenizer(SourceCarriesTokenizer::No, TargetNeedsTokenizer::Yes, true),
            TokenizerVerdict::Ok
        );
    }

    #[test]
    fn target_without_tokenizer_passes_regardless() {
        // If target doesn't need a tokenizer (e.g., weights-only export),
        // the rule is vacuous regardless of source state.
        for src in [SourceCarriesTokenizer::Yes, SourceCarriesTokenizer::No] {
            for ext in [false, true] {
                assert_eq!(
                    validate_tokenizer(src, TargetNeedsTokenizer::No, ext),
                    TokenizerVerdict::Ok
                );
            }
        }
    }
}
