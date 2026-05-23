//! # apr rosetta validate-stats — Reference vs Fingerprints Source
//!
//! `apr rosetta validate-stats` accepts either `--reference <MODEL>` or
//! `--fingerprints <FILE>` to source the comparison stats. Exactly one
//! must be provided (mutually exclusive). This recipe documents the
//! constraint web and the deterministic fallback rule when neither is
//! provided (refuse to validate — there's nothing to compare against).
//!
//! Demonstrates the **ROSETTA-VALIDATE.2** recipe for PMAT-097 (apr rosetta validate-stats coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-202
//!
//! Run with: cargo run --example cli_rosetta_validate_stats_reference_or_fingerprints
//!
//! Added by PMAT-097 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Default, Clone)]
pub struct ValidateInvocation {
    pub model: String,
    pub reference: Option<String>,
    pub fingerprints: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum SourceVerdict {
    UseReference(String),
    UseFingerprints(String),
    Underspecified,
    Conflicting,
    EmptyModel,
}

pub fn pick_source(inv: &ValidateInvocation) -> SourceVerdict {
    if inv.model.is_empty() {
        return SourceVerdict::EmptyModel;
    }
    match (&inv.reference, &inv.fingerprints) {
        (Some(r), None) => SourceVerdict::UseReference(r.clone()),
        (None, Some(f)) => SourceVerdict::UseFingerprints(f.clone()),
        (Some(_), Some(_)) => SourceVerdict::Conflicting,
        (None, None) => SourceVerdict::Underspecified,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_rosetta_validate_stats_reference_or_fingerprints")?;

    let cases: &[(&str, ValidateInvocation)] = &[
        (
            "ref only",
            ValidateInvocation {
                model: "test.apr".into(),
                reference: Some("ref.gguf".into()),
                fingerprints: None,
            },
        ),
        (
            "fp only",
            ValidateInvocation {
                model: "test.apr".into(),
                reference: None,
                fingerprints: Some("fp.json".into()),
            },
        ),
        (
            "neither",
            ValidateInvocation {
                model: "test.apr".into(),
                ..Default::default()
            },
        ),
        (
            "both",
            ValidateInvocation {
                model: "test.apr".into(),
                reference: Some("ref.gguf".into()),
                fingerprints: Some("fp.json".into()),
            },
        ),
        (
            "no model",
            ValidateInvocation {
                model: String::new(),
                reference: Some("ref.gguf".into()),
                fingerprints: None,
            },
        ),
    ];

    for (label, inv) in cases {
        println!("{label:>10}  →  {:?}", pick_source(inv));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn reference_only_picks_reference() {
        let inv = ValidateInvocation {
            model: "m.apr".into(),
            reference: Some("r.gguf".into()),
            fingerprints: None,
        };
        assert_eq!(
            pick_source(&inv),
            SourceVerdict::UseReference("r.gguf".into())
        );
    }

    #[test]
    fn fingerprints_only_picks_fingerprints() {
        let inv = ValidateInvocation {
            model: "m.apr".into(),
            reference: None,
            fingerprints: Some("fp.json".into()),
        };
        assert_eq!(
            pick_source(&inv),
            SourceVerdict::UseFingerprints("fp.json".into())
        );
    }

    #[test]
    fn neither_is_underspecified_not_default() {
        // Critical: don't silently assume a default — operator must pick.
        let inv = ValidateInvocation {
            model: "m.apr".into(),
            ..Default::default()
        };
        assert_eq!(pick_source(&inv), SourceVerdict::Underspecified);
    }

    #[test]
    fn both_provided_is_conflicting() {
        // Mutually exclusive: passing both indicates operator confusion.
        let inv = ValidateInvocation {
            model: "m.apr".into(),
            reference: Some("r".into()),
            fingerprints: Some("f".into()),
        };
        assert_eq!(pick_source(&inv), SourceVerdict::Conflicting);
    }

    #[test]
    fn empty_model_short_circuits_before_source_check() {
        // Empty model is the priority error — surface that first.
        let inv = ValidateInvocation {
            model: String::new(),
            reference: Some("r".into()),
            fingerprints: None,
        };
        assert_eq!(pick_source(&inv), SourceVerdict::EmptyModel);
    }
}
