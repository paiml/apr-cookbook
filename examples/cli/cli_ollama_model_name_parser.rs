//! # apr ollama — Model Name Parser
//!
//! Ollama model names follow `<library>/<name>:<tag>` (library
//! defaults to `library`, tag defaults to `latest`). Sub-tags
//! encode quantization (`:q4_K_M`, `:fp16`). This recipe builds the
//! parser + tag classifier.
//!
//! Demonstrates the **OLLAMA.4** recipe for PMAT-120 (apr ollama coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender OLLAMA-001 + Ollama registry conventions
//!
//! Run with: cargo run --example cli_ollama_model_name_parser
//!
//! Added by PMAT-120 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Eq)]
pub struct ModelRef {
    pub library: String,
    pub name: String,
    pub tag: String,
}

#[derive(Debug, PartialEq)]
pub enum ParseError {
    EmptyName,
    InvalidCharacter,
}

pub fn parse(name: &str) -> std::result::Result<ModelRef, ParseError> {
    if name.is_empty() {
        return Err(ParseError::EmptyName);
    }
    let (library, rest) = match name.split_once('/') {
        Some((lib, rest)) => (lib.to_string(), rest),
        None => ("library".to_string(), name),
    };
    let (model_name, tag) = match rest.split_once(':') {
        Some((n, t)) if !n.is_empty() && !t.is_empty() => (n.to_string(), t.to_string()),
        Some((n, _)) if !n.is_empty() => (n.to_string(), "latest".to_string()),
        _ => (rest.to_string(), "latest".to_string()),
    };
    if model_name.is_empty() {
        return Err(ParseError::EmptyName);
    }
    if !is_valid_segment(&library) || !is_valid_segment(&model_name) || !is_valid_tag(&tag) {
        return Err(ParseError::InvalidCharacter);
    }
    Ok(ModelRef {
        library,
        name: model_name,
        tag,
    })
}

fn is_valid_segment(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

fn is_valid_tag(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | ':'))
}

#[derive(Debug, PartialEq, Eq)]
pub enum TagKind {
    Latest,
    QuantizedInt(u8), // q4, q5, q8
    QuantizedKMix(String),
    Fp16,
    Fp32,
    Other,
}

pub fn classify_tag(tag: &str) -> TagKind {
    if tag == "latest" {
        return TagKind::Latest;
    }
    if tag == "fp16" {
        return TagKind::Fp16;
    }
    if tag == "fp32" {
        return TagKind::Fp32;
    }
    let lower = tag.to_ascii_lowercase();
    if lower.contains("_k")
        || lower.ends_with("_m")
        || lower.ends_with("_s")
        || lower.ends_with("_l")
    {
        return TagKind::QuantizedKMix(tag.into());
    }
    if let Some(rest) = lower.strip_prefix('q') {
        if let Ok(bits) = rest.parse::<u8>() {
            if (2..=8).contains(&bits) {
                return TagKind::QuantizedInt(bits);
            }
        }
    }
    TagKind::Other
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_ollama_model_name_parser")?;

    for n in [
        "llama3:8b",
        "library/llama3",
        "meta-llama/llama-3-70b:q4_K_M",
        "phi-3:fp16",
        "",
        "bad name",
    ] {
        let parsed = parse(n);
        if let Ok(ref m) = parsed {
            println!("{n:<35}  →  {m:?}  tag-kind={:?}", classify_tag(&m.tag));
        } else {
            println!("{n:<35}  →  {parsed:?}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn library_defaults_when_no_slash() {
        let m = parse("llama3:8b").unwrap();
        assert_eq!(m.library, "library");
        assert_eq!(m.name, "llama3");
        assert_eq!(m.tag, "8b");
    }

    #[test]
    fn explicit_library_parsed() {
        let m = parse("meta-llama/llama-3-70b").unwrap();
        assert_eq!(m.library, "meta-llama");
        assert_eq!(m.name, "llama-3-70b");
        assert_eq!(m.tag, "latest");
    }

    #[test]
    fn full_form_with_quant_parsed() {
        let m = parse("meta-llama/llama-3:q4_K_M").unwrap();
        assert_eq!(m.tag, "q4_K_M");
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(parse(""), Err(ParseError::EmptyName));
    }

    #[test]
    fn space_in_name_rejected() {
        assert_eq!(parse("bad name"), Err(ParseError::InvalidCharacter));
    }

    #[test]
    fn classify_latest_tag() {
        assert_eq!(classify_tag("latest"), TagKind::Latest);
    }

    #[test]
    fn classify_q4_int() {
        assert_eq!(classify_tag("q4"), TagKind::QuantizedInt(4));
        assert_eq!(classify_tag("q8"), TagKind::QuantizedInt(8));
    }

    #[test]
    fn classify_k_mix_tags() {
        assert!(matches!(classify_tag("q4_K_M"), TagKind::QuantizedKMix(_)));
        assert!(matches!(classify_tag("q5_K_S"), TagKind::QuantizedKMix(_)));
    }

    #[test]
    fn classify_fp16_fp32() {
        assert_eq!(classify_tag("fp16"), TagKind::Fp16);
        assert_eq!(classify_tag("fp32"), TagKind::Fp32);
    }

    #[test]
    fn classify_unknown_returns_other() {
        assert_eq!(classify_tag("xyz123"), TagKind::Other);
    }
}
