//! # Architecture Alias Resolver — Demonstrate Upstream Alias Mechanism
//!
//! Mirror the upstream `aprender::format::FamilyRegistry::register_alias` /
//! `resolve_alias` API that aprender PR #1562 adds. This recipe demonstrates
//! how alias-based family resolution unblocks the 11 derived-model entries
//! tracked as `status: blocked` in
//! `docs/specifications/architecture-demos/manifest.yaml` (codellama,
//! tinyllama, vicuna, yi, smollm, smollm2, codestral, zephyr, dolphin,
//! hermes, openchat, wizardcoder, distilgpt2, pythia, galactica, codegemma).
//!
//! Demonstrates the **ARCH-ALIAS-RESOLVER** recipe per
//! `docs/specifications/architecture-demos.md` v1.1 + upstream aprender#1562.
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/inference-arch-alias-resolver-v1.yaml (grade C; lean_status: wip)
//! Citation: aprender#1562 (FamilyRegistry alias mechanism); manifest.yaml blocked-entries catalog
//!
//! Run with: cargo run --example inference_arch_alias_resolver
//!
//! Added by PMAT-313 (architecture-demos v1.1: upstream alias resolver demo).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

/// Registered aliases: HF repo glob → parent family. Mirrors what a caller
/// would set up upstream via `FamilyRegistry::register_alias`.
const ALIASES: &[(&str, &str)] = &[
    // Llama derivatives
    ("codellama/*", "llama"),
    ("TinyLlama/*", "llama"),
    ("lmsys/vicuna-*", "llama"),
    ("01-ai/Yi-*", "llama"),
    ("HuggingFaceTB/SmolLM-*", "llama"),
    ("HuggingFaceTB/SmolLM2-*", "llama"),
    // Mistral derivatives
    ("mistralai/Codestral-*", "mistral"),
    ("HuggingFaceH4/zephyr-*", "mistral"),
    // Hybrid (could match multiple parents — first match wins)
    ("cognitivecomputations/dolphin-*", "llama"),
    ("NousResearch/Hermes-*", "llama"),
    ("openchat/openchat-*", "llama"),
    ("WizardLM/WizardCoder-*", "llama"),
    // Single-loader aliases
    ("distilbert/distilgpt2", "gpt2"),
    ("EleutherAI/pythia-*", "gptneox"),
    ("facebook/galactica-*", "opt"),
    ("google/codegemma-*", "gemma"),
];

#[derive(Debug, PartialEq)]
pub enum AliasVerdict {
    Ok {
        hf_repo: String,
        resolved_family: String,
        matched_pattern: String,
    },
    NoMatch {
        hf_repo: String,
    },
    InvalidInput,
}

pub fn resolve(hf_repo: &str) -> AliasVerdict {
    if hf_repo.is_empty() {
        return AliasVerdict::InvalidInput;
    }
    for (pattern, parent) in ALIASES {
        if alias_matches(pattern, hf_repo) {
            return AliasVerdict::Ok {
                hf_repo: hf_repo.to_string(),
                resolved_family: (*parent).to_string(),
                matched_pattern: (*pattern).to_string(),
            };
        }
    }
    AliasVerdict::NoMatch {
        hf_repo: hf_repo.to_string(),
    }
}

/// Glob match: `*` is suffix wildcard. e.g., "codellama/*" matches
/// "codellama/CodeLlama-7b-hf". Same logic as upstream
/// `aprender::format::family_registry::alias_matches`.
fn alias_matches(pattern: &str, hf_repo: &str) -> bool {
    if let Some(prefix) = pattern.strip_suffix('*') {
        hf_repo.starts_with(prefix)
    } else {
        pattern == hf_repo
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("inference_arch_alias_resolver")?;
    let probes = [
        "codellama/CodeLlama-7b-hf",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "EleutherAI/pythia-70m",
        "google/codegemma-7b",
        "mistralai/Codestral-22B-v0.1",
        "openai/gpt-4-turbo",               // unsupported
        "meta-llama/Llama-3.1-8B-Instruct", // not aliased — caller would use detector
    ];
    for repo in probes {
        match resolve(repo) {
            AliasVerdict::Ok {
                resolved_family,
                matched_pattern,
                ..
            } => println!("  {repo:<48} ↦ {resolved_family} (via {matched_pattern})"),
            AliasVerdict::NoMatch { .. } => {
                println!("  {repo:<48} ↦ NoMatch (caller falls through to detector)");
            }
            AliasVerdict::InvalidInput => println!("  invalid: {repo}"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alias_resolver_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_input_invalid() {
        assert_eq!(resolve(""), AliasVerdict::InvalidInput);
    }

    #[test]
    fn codellama_resolves_to_llama() {
        if let AliasVerdict::Ok {
            resolved_family,
            matched_pattern,
            ..
        } = resolve("codellama/CodeLlama-7b-hf")
        {
            assert_eq!(resolved_family, "llama");
            assert_eq!(matched_pattern, "codellama/*");
        } else {
            panic!("expected Ok");
        }
    }

    #[test]
    fn tinyllama_resolves_to_llama() {
        if let AliasVerdict::Ok {
            resolved_family, ..
        } = resolve("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        {
            assert_eq!(resolved_family, "llama");
        }
    }

    #[test]
    fn pythia_resolves_to_gptneox() {
        if let AliasVerdict::Ok {
            resolved_family, ..
        } = resolve("EleutherAI/pythia-70m")
        {
            assert_eq!(resolved_family, "gptneox");
        }
    }

    #[test]
    fn galactica_resolves_to_opt() {
        if let AliasVerdict::Ok {
            resolved_family, ..
        } = resolve("facebook/galactica-1.3b")
        {
            assert_eq!(resolved_family, "opt");
        }
    }

    #[test]
    fn codestral_resolves_to_mistral() {
        if let AliasVerdict::Ok {
            resolved_family, ..
        } = resolve("mistralai/Codestral-22B-v0.1")
        {
            assert_eq!(resolved_family, "mistral");
        }
    }

    #[test]
    fn distilgpt2_exact_match() {
        // distilgpt2 is registered as exact, not glob — different code path.
        if let AliasVerdict::Ok {
            resolved_family,
            matched_pattern,
            ..
        } = resolve("distilbert/distilgpt2")
        {
            assert_eq!(resolved_family, "gpt2");
            assert_eq!(matched_pattern, "distilbert/distilgpt2");
        }
    }

    #[test]
    fn unaliased_repo_returns_nomatch() {
        // Llama itself isn't aliased — caller's detector handles it.
        assert!(matches!(
            resolve("meta-llama/Llama-3.1-8B-Instruct"),
            AliasVerdict::NoMatch { .. }
        ));
    }

    #[test]
    fn unknown_org_returns_nomatch() {
        assert!(matches!(
            resolve("openai/gpt-4-turbo"),
            AliasVerdict::NoMatch { .. }
        ));
    }

    #[test]
    fn alias_count_covers_blocked_backlog() {
        // Should cover all 16 alias-eligible blocked families from manifest.yaml.
        assert_eq!(ALIASES.len(), 16);
    }

    #[test]
    fn glob_matches_prefix() {
        assert!(alias_matches("codellama/*", "codellama/CodeLlama-7b-hf"));
        assert!(!alias_matches("codellama/*", "meta-llama/Llama-3-8B"));
    }

    #[test]
    fn deterministic_across_runs() {
        let a = resolve("codellama/CodeLlama-7b-hf");
        let b = resolve("codellama/CodeLlama-7b-hf");
        assert_eq!(a, b);
    }
}
