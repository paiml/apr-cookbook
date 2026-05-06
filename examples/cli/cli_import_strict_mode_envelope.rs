//! # apr import — `--strict` Mode Envelope
//!
//! `apr import <SOURCE> --strict` rejects unverified architectures and
//! converts validation warnings into hard errors. This recipe models the
//! strict-vs-default classifier and asserts the contract: known
//! architectures pass either way; unknown architectures pass only without
//! `--strict`; tokenizer/config warnings are upgraded to errors only when
//! `--strict` is set.
//!
//! Demonstrates the **IMPORT.4** recipe for PMAT-099 (apr import coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-IMPORT-001 + Hugging Face architecture allowlist
//!
//! Run with: cargo run --example cli_import_strict_mode_envelope
//!
//! Added by PMAT-099 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const VERIFIED_ARCHITECTURES: &[&str] = &[
    "whisper",
    "llama",
    "bert",
    "qwen2",
    "qwen3",
    "gpt2",
    "starcoder",
    "gpt-neox",
    "opt",
    "phi",
    "gemma",
    "falcon",
    "mamba",
    "t5",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningSource {
    UnverifiedArchitecture,
    MissingTokenizer,
    MissingConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ImportVerdict {
    Pass,
    PassWithWarnings(Vec<WarningSource>),
    Fail(Vec<WarningSource>),
}

pub fn classify_import(
    architecture: &str,
    has_tokenizer: bool,
    has_config: bool,
    strict: bool,
) -> ImportVerdict {
    let mut warnings = Vec::new();
    if !VERIFIED_ARCHITECTURES.contains(&architecture) {
        warnings.push(WarningSource::UnverifiedArchitecture);
    }
    if !has_tokenizer {
        warnings.push(WarningSource::MissingTokenizer);
    }
    if !has_config {
        warnings.push(WarningSource::MissingConfig);
    }

    if warnings.is_empty() {
        ImportVerdict::Pass
    } else if strict {
        ImportVerdict::Fail(warnings)
    } else {
        ImportVerdict::PassWithWarnings(warnings)
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_import_strict_mode_envelope")?;

    let cases = [
        ("verified, all present, strict", "qwen3", true, true, true),
        (
            "unverified, all present, default",
            "exotic",
            true,
            true,
            false,
        ),
        (
            "unverified, all present, strict",
            "exotic",
            true,
            true,
            true,
        ),
        ("verified, no config, strict", "llama", true, false, true),
        ("verified, no config, default", "llama", true, false, false),
    ];
    for (label, arch, tok, cfg, strict) in cases {
        let v = classify_import(arch, tok, cfg, strict);
        println!("{label:>40}  →  {v:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strict_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn verified_arch_with_full_metadata_passes() {
        assert_eq!(
            classify_import("qwen3", true, true, false),
            ImportVerdict::Pass
        );
        assert_eq!(
            classify_import("qwen3", true, true, true),
            ImportVerdict::Pass
        );
    }

    #[test]
    fn unverified_arch_warns_in_default_mode() {
        let v = classify_import("exotic", true, true, false);
        assert!(
            matches!(v, ImportVerdict::PassWithWarnings(ws) if ws.contains(&WarningSource::UnverifiedArchitecture))
        );
    }

    #[test]
    fn unverified_arch_fails_in_strict_mode() {
        let v = classify_import("exotic", true, true, true);
        assert!(
            matches!(v, ImportVerdict::Fail(ws) if ws.contains(&WarningSource::UnverifiedArchitecture))
        );
    }

    #[test]
    fn missing_tokenizer_only_warns_in_default() {
        let v = classify_import("llama", false, true, false);
        assert!(
            matches!(v, ImportVerdict::PassWithWarnings(ws) if ws.contains(&WarningSource::MissingTokenizer))
        );
    }

    #[test]
    fn missing_tokenizer_fails_in_strict() {
        let v = classify_import("llama", false, true, true);
        assert!(
            matches!(v, ImportVerdict::Fail(ws) if ws.contains(&WarningSource::MissingTokenizer))
        );
    }

    #[test]
    fn multiple_warnings_aggregate() {
        // Unverified arch + no tokenizer + no config → 3 warnings.
        let v = classify_import("exotic", false, false, false);
        if let ImportVerdict::PassWithWarnings(ws) = v {
            assert_eq!(ws.len(), 3);
        } else {
            panic!("expected PassWithWarnings");
        }
    }
}
