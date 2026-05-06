//! # apr pull — Short Alias Resolver
//!
//! `apr pull <ALIAS>` accepts short aliases (e.g., `qwen-coder-7b`) and
//! resolves them to canonical hf:// URIs from `configs/aliases.yaml`.
//! This recipe builds the resolver and asserts the contract: known
//! aliases resolve, unknown short names that LOOK like aliases (no slash,
//! no scheme) reject with a suggestion, fully-qualified inputs pass through.
//!
//! Demonstrates the **PULL.4** recipe for PMAT-101 (apr pull coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender CRUX-A-01 + alias-map convention
//!
//! Run with: cargo run --example cli_pull_alias_resolver
//!
//! Added by PMAT-101 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputKind {
    HfUri,   // hf://org/repo
    OrgRepo, // org/repo (slash-separated)
    Alias,   // single token (no slash, no scheme)
}

pub fn classify_input(s: &str) -> InputKind {
    if s.starts_with("hf://") {
        InputKind::HfUri
    } else if s.contains('/') {
        InputKind::OrgRepo
    } else {
        InputKind::Alias
    }
}

#[derive(Debug, PartialEq)]
pub enum ResolveVerdict {
    Resolved(String),    // canonical hf:// URI
    Passthrough(String), // already canonical, no resolution needed
    UnknownAlias(String),
}

pub fn resolve_alias(s: &str, alias_map: &[(&str, &str)]) -> ResolveVerdict {
    match classify_input(s) {
        InputKind::HfUri => ResolveVerdict::Passthrough(s.into()),
        InputKind::OrgRepo => ResolveVerdict::Passthrough(format!("hf://{s}")),
        InputKind::Alias => match alias_map.iter().find(|(a, _)| *a == s) {
            Some((_, canonical)) => ResolveVerdict::Resolved((*canonical).to_string()),
            None => ResolveVerdict::UnknownAlias(s.into()),
        },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_pull_alias_resolver")?;

    let aliases = &[
        ("qwen-coder-7b", "hf://Qwen/Qwen2.5-Coder-7B-Instruct"),
        ("whisper-tiny", "hf://openai/whisper-tiny"),
        ("llama-3-8b", "hf://meta-llama/Llama-3.1-8B-Instruct"),
    ];

    for s in [
        "qwen-coder-7b",
        "whisper-tiny",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "hf://openai/whisper-large",
        "qwen-typo",
    ] {
        println!("{s:>40}  →  {:?}", resolve_alias(s, aliases));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn aliases() -> Vec<(&'static str, &'static str)> {
        vec![
            ("qwen-coder-7b", "hf://Qwen/Qwen2.5-Coder-7B-Instruct"),
            ("whisper-tiny", "hf://openai/whisper-tiny"),
        ]
    }

    #[test]
    fn resolver_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn input_classified_correctly() {
        assert_eq!(classify_input("hf://Qwen/foo"), InputKind::HfUri);
        assert_eq!(classify_input("Qwen/foo"), InputKind::OrgRepo);
        assert_eq!(classify_input("qwen-coder-7b"), InputKind::Alias);
    }

    #[test]
    fn known_alias_resolves() {
        let v = resolve_alias("qwen-coder-7b", &aliases());
        assert_eq!(
            v,
            ResolveVerdict::Resolved("hf://Qwen/Qwen2.5-Coder-7B-Instruct".into())
        );
    }

    #[test]
    fn org_repo_passes_through_with_hf_prefix() {
        // No alias resolution; just upgrade to canonical hf:// form.
        let v = resolve_alias("Qwen/Qwen2.5-Coder-7B-Instruct", &aliases());
        assert_eq!(
            v,
            ResolveVerdict::Passthrough("hf://Qwen/Qwen2.5-Coder-7B-Instruct".into())
        );
    }

    #[test]
    fn already_hf_uri_passes_through_unchanged() {
        let v = resolve_alias("hf://openai/whisper-large", &aliases());
        assert_eq!(
            v,
            ResolveVerdict::Passthrough("hf://openai/whisper-large".into())
        );
    }

    #[test]
    fn unknown_alias_returns_unknown() {
        // Operator typo "qwen-typo" must NOT silently pass through — they
        // wouldn't get the model they wanted.
        let v = resolve_alias("qwen-typo", &aliases());
        assert!(matches!(v, ResolveVerdict::UnknownAlias(_)));
    }

    #[test]
    fn empty_alias_returns_unknown() {
        let v = resolve_alias("", &aliases());
        assert!(matches!(v, ResolveVerdict::UnknownAlias(_)));
    }
}
