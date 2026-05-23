//! # Architecture Resolution Pipeline — Compose Alias Resolver + Detector
//!
//! End-to-end family resolution: given an HF repo identifier and a
//! `config.json` body, return a single `DetectedFamily`. The pipeline
//! tries the alias resolver first (covers derived models like
//! `codellama/*` → llama) and falls through to the discriminator-based
//! detector when the repo is not aliased.
//!
//! Demonstrates the **ARCH-RESOLUTION-PIPELINE** recipe per
//! `docs/specifications/architecture-demos.md` v1.1.2 — a forward-bridge
//! to the upstream `aprender::format::FamilyRegistry` API
//! ([aprender#1562](https://github.com/paiml/aprender/pull/1562), open).
//! When that PR ships in an aprender release, the pipeline body becomes
//! a thin wrapper over `FamilyRegistry::resolve_alias` +
//! `FamilyRegistry::detect_from_config_str`. Until then, this recipe
//! composes the two cookbook-side reverse implementations:
//! `inference_arch_alias_resolver::resolve` and
//! `inference_arch_detector::detect_from_str`.
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/inference-arch-resolution-pipeline-v1.yaml
//! Citation: aprender#1562 (FamilyRegistry compose target); HuggingFace `config.json` schema
//!
//! Run with: cargo run --example inference_arch_resolution_pipeline
//!
//! Added by PMAT-320 (architecture-demos v1.2: forward-bridge resolution pipeline).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[path = "inference_arch_alias_resolver.rs"]
#[allow(dead_code, unreachable_pub)]
mod alias_resolver;

#[path = "inference_arch_detector.rs"]
#[allow(dead_code, unreachable_pub)]
mod detector;

#[derive(Debug, PartialEq)]
pub enum ResolutionVerdict {
    AliasHit {
        hf_repo: String,
        family: String,
        matched_pattern: String,
    },
    DetectorHit {
        family: String,
        match_reason: String,
    },
    Unknown {
        hf_repo: String,
        config_excerpt: String,
    },
    InvalidInput,
}

/// Resolve a (hf_repo, config_body) pair to a single family.
///
/// Priority: alias resolver wins over detector. Aliases encode known
/// derived-model relationships (codellama/*→llama, distilbert/distilgpt2→gpt2),
/// so when both signals are available the alias is authoritative.
pub fn resolve(hf_repo: &str, config_body: &str) -> ResolutionVerdict {
    if hf_repo.is_empty() && config_body.is_empty() {
        return ResolutionVerdict::InvalidInput;
    }

    // Pass 1: alias resolver (only when hf_repo is non-empty)
    if !hf_repo.is_empty() {
        match alias_resolver::resolve(hf_repo) {
            alias_resolver::AliasVerdict::Ok {
                hf_repo: r,
                resolved_family,
                matched_pattern,
            } => {
                return ResolutionVerdict::AliasHit {
                    hf_repo: r,
                    family: resolved_family,
                    matched_pattern,
                };
            }
            alias_resolver::AliasVerdict::NoMatch { .. } => {
                // Fall through to detector
            }
            alias_resolver::AliasVerdict::InvalidInput => {
                // Empty hf_repo would have been caught above
            }
        }
    }

    // Pass 2: discriminator-based detector (works on raw config body)
    if config_body.is_empty() {
        return ResolutionVerdict::Unknown {
            hf_repo: hf_repo.to_string(),
            config_excerpt: String::new(),
        };
    }
    match detector::detect_from_str(config_body) {
        detector::DetectorVerdict::Family {
            family,
            match_reason,
        } => ResolutionVerdict::DetectorHit {
            family: family.as_str().to_string(),
            match_reason,
        },
        detector::DetectorVerdict::UnknownFamily { config_excerpt } => ResolutionVerdict::Unknown {
            hf_repo: hf_repo.to_string(),
            config_excerpt,
        },
        detector::DetectorVerdict::InvalidFixture => ResolutionVerdict::InvalidInput,
    }
}

fn fixture_body(family: &str) -> Option<String> {
    let path = format!("tests/fixtures/architectures/{family}/config.json");
    std::fs::read_to_string(path).ok()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("inference_arch_resolution_pipeline")?;

    println!("== Aliased derivatives (alias resolver hits) ==");
    let aliased = [
        ("codellama/CodeLlama-7b-hf", "llama"),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama"),
        ("EleutherAI/pythia-70m", "gptneox"),
        ("mistralai/Codestral-22B-v0.1", "mistral"),
        ("distilbert/distilgpt2", "gpt2"),
    ];
    for (repo, parent) in aliased {
        let body = fixture_body(parent).unwrap_or_default();
        match resolve(repo, &body) {
            ResolutionVerdict::AliasHit {
                family,
                matched_pattern,
                ..
            } => println!("  {repo:<48} ↦ {family} (alias: {matched_pattern})"),
            other => println!("  {repo:<48} ↦ unexpected {other:?}"),
        }
    }

    println!("\n== Direct detector hits (no alias) ==");
    let direct = ["llama", "mistral", "qwen3", "phi", "bert", "deepseek"];
    for fam in direct {
        let body = fixture_body(fam).unwrap_or_default();
        let canonical_repo = format!("paiml/{fam}-test");
        match resolve(&canonical_repo, &body) {
            ResolutionVerdict::DetectorHit {
                family,
                match_reason,
            } => println!("  {canonical_repo:<48} ↦ {family} (detector: {match_reason})"),
            other => println!("  {canonical_repo:<48} ↦ unexpected {other:?}"),
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(family: &str) -> String {
        std::fs::read_to_string(format!("tests/fixtures/architectures/{family}/config.json"))
            .unwrap_or_default()
    }

    #[test]
    fn pipeline_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_inputs_are_invalid() {
        assert_eq!(resolve("", ""), ResolutionVerdict::InvalidInput);
    }

    #[test]
    fn alias_takes_priority_over_detector() {
        // codellama/* repo with llama config — alias should win.
        let body = fixture("llama");
        match resolve("codellama/CodeLlama-7b-hf", &body) {
            ResolutionVerdict::AliasHit {
                family,
                matched_pattern,
                ..
            } => {
                assert_eq!(family, "llama");
                assert_eq!(matched_pattern, "codellama/*");
            }
            other => panic!("expected AliasHit, got {other:?}"),
        }
    }

    #[test]
    fn detector_handles_unaliased_repo() {
        let body = fixture("mistral");
        match resolve("mistralai/Mistral-7B-Instruct-v0.3", &body) {
            ResolutionVerdict::DetectorHit { family, .. } => assert_eq!(family, "mistral"),
            other => panic!("expected DetectorHit, got {other:?}"),
        }
    }

    #[test]
    fn detector_works_with_empty_repo() {
        let body = fixture("phi");
        match resolve("", &body) {
            ResolutionVerdict::DetectorHit { family, .. } => assert_eq!(family, "phi"),
            other => panic!("expected DetectorHit, got {other:?}"),
        }
    }

    #[test]
    fn unknown_repo_and_unknown_config_returns_unknown() {
        let body = r#"{"model_type": "fictional_2050"}"#;
        match resolve("acme/totally-unknown-arch", body) {
            ResolutionVerdict::Unknown { hf_repo, .. } => {
                assert_eq!(hf_repo, "acme/totally-unknown-arch");
            }
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    #[test]
    fn aliased_repo_with_empty_config_still_resolves() {
        // Alias resolver doesn't need the config body to fire.
        match resolve("codellama/CodeLlama-7b-hf", "") {
            ResolutionVerdict::AliasHit { family, .. } => assert_eq!(family, "llama"),
            other => panic!("expected AliasHit, got {other:?}"),
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let body = fixture("qwen3");
        let a = resolve("Qwen/Qwen3-7B", &body);
        let b = resolve("Qwen/Qwen3-7B", &body);
        assert_eq!(a, b);
    }

    #[test]
    fn all_alias_eligible_resolve_to_parent() {
        // Mirrors the 16-row alias table from inference_arch_alias_resolver.
        // This is the falsification claim: every alias-eligible blocked
        // family in the manifest resolves to a known parent via the pipeline.
        let cases: &[(&str, &str)] = &[
            ("codellama/CodeLlama-7b-hf", "llama"),
            ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "llama"),
            ("lmsys/vicuna-13b-v1.5", "llama"),
            ("01-ai/Yi-6B", "llama"),
            ("HuggingFaceTB/SmolLM-135M", "llama"),
            ("HuggingFaceTB/SmolLM2-1.7B", "llama"),
            ("mistralai/Codestral-22B-v0.1", "mistral"),
            ("HuggingFaceH4/zephyr-7b-beta", "mistral"),
            ("cognitivecomputations/dolphin-2.9-llama3-8b", "llama"),
            ("NousResearch/Hermes-3-Llama-3.1-8B", "llama"),
            ("openchat/openchat-3.5-1210", "llama"),
            ("WizardLM/WizardCoder-15B-V1.0", "llama"),
            ("distilbert/distilgpt2", "gpt2"),
            ("EleutherAI/pythia-70m", "gptneox"),
            ("facebook/galactica-1.3b", "opt"),
            ("google/codegemma-7b", "gemma"),
        ];
        for (repo, expected_parent) in cases {
            match resolve(repo, "") {
                ResolutionVerdict::AliasHit { family, .. } => assert_eq!(
                    &family, expected_parent,
                    "{repo} expected {expected_parent}, got {family}"
                ),
                other => panic!("{repo} expected AliasHit, got {other:?}"),
            }
        }
    }

    #[test]
    fn detector_works_on_all_18_certified_families() {
        // Every certified family in manifest.yaml has a fixture; the pipeline
        // must produce a DetectorHit for each (no aliases involved).
        for fam in [
            "llama",
            "mistral",
            "qwen2",
            "qwen3",
            "qwen3_5",
            "phi",
            "gemma",
            "gpt2",
            "gptneox",
            "deepseek",
            "falcon_h1",
            "rwkv7",
            "openelm",
            "opt",
            "mamba",
            "bert",
        ] {
            let body = fixture(fam);
            let repo = format!("paiml/{fam}-fixture");
            match resolve(&repo, &body) {
                ResolutionVerdict::DetectorHit { family, .. } => assert_eq!(
                    &family, fam,
                    "{fam} fixture expected detector hit on {fam}, got {family}"
                ),
                other => panic!("{fam} fixture expected DetectorHit, got {other:?}"),
            }
        }
    }

    #[test]
    fn alias_repo_with_mismatched_config_still_picks_alias() {
        // codellama/* repo but a phi config — alias still wins because the
        // pipeline trusts the repo identifier as the more specific signal.
        let body = fixture("phi");
        match resolve("codellama/CodeLlama-7b-hf", &body) {
            ResolutionVerdict::AliasHit { family, .. } => assert_eq!(family, "llama"),
            other => panic!("expected AliasHit (alias wins), got {other:?}"),
        }
    }
}
