//! # apr parity — Default Prompt Envelope
//!
//! `apr parity <FILE>` accepts `--prompt <TEXT>` (default "What is 2+2?").
//! This recipe documents the prompt-selection contract: the default is
//! a deterministic-output question to maximise CPU/GPU agreement, custom
//! prompts pass through verbatim, empty prompt rejected.
//!
//! Demonstrates the **PARITY.9** recipe for PMAT-111 (apr parity coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PMAT-232
//!
//! Run with: cargo run --example cli_parity_default_prompt_envelope
//!
//! Added by PMAT-111 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

pub const DEFAULT_PROMPT: &str = "What is 2+2?";

#[derive(Debug, PartialEq)]
pub enum PromptVerdict {
    UseDefault,
    UseCustom(String),
    EmptyRejected,
}

pub fn resolve_prompt(custom: Option<&str>) -> PromptVerdict {
    match custom {
        None => PromptVerdict::UseDefault,
        Some(s) if s.trim().is_empty() => PromptVerdict::EmptyRejected,
        Some(s) => PromptVerdict::UseCustom(s.into()),
    }
}

pub fn effective_prompt(custom: Option<&str>) -> Option<String> {
    match resolve_prompt(custom) {
        PromptVerdict::UseDefault => Some(DEFAULT_PROMPT.into()),
        PromptVerdict::UseCustom(s) => Some(s),
        PromptVerdict::EmptyRejected => None,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_parity_default_prompt_envelope")?;

    for c in [None, Some("Hello world"), Some("   "), Some("")] {
        println!("--prompt {c:?}  →  {:?}", resolve_prompt(c));
        println!("  effective: {:?}", effective_prompt(c));
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
    fn no_custom_uses_default() {
        assert_eq!(resolve_prompt(None), PromptVerdict::UseDefault);
        assert_eq!(effective_prompt(None).as_deref(), Some(DEFAULT_PROMPT));
    }

    #[test]
    fn custom_passes_verbatim() {
        let v = resolve_prompt(Some("Hello"));
        assert_eq!(v, PromptVerdict::UseCustom("Hello".into()));
    }

    #[test]
    fn empty_string_rejected() {
        assert_eq!(resolve_prompt(Some("")), PromptVerdict::EmptyRejected);
    }

    #[test]
    fn whitespace_only_rejected() {
        assert_eq!(resolve_prompt(Some("   ")), PromptVerdict::EmptyRejected);
    }

    #[test]
    fn empty_returns_none_effective() {
        assert!(effective_prompt(Some("")).is_none());
    }

    #[test]
    fn default_prompt_is_deterministic_question() {
        // The default is a math question with one canonical answer, chosen
        // because models trivially agree on it across CPU/GPU paths. Any
        // change requires updating the parity baseline.
        assert_eq!(DEFAULT_PROMPT, "What is 2+2?");
    }

    #[test]
    fn custom_with_special_chars_preserved() {
        let v = resolve_prompt(Some("Émojis 🎉 + unicode"));
        if let PromptVerdict::UseCustom(s) = v {
            assert!(s.contains('🎉'));
        }
    }
}
