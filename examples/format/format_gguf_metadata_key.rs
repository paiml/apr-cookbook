//! # Format GGUF Metadata Key Validator
//!
//! GGUF metadata keys follow `<namespace>.<field>` convention:
//! `general.architecture`, `general.name`, `llama.attention.head_count`,
//! `tokenizer.ggml.tokens`. This recipe validates the namespace prefix
//! + key shape (lowercase + dot-separated + alphanumeric/underscore).
//!
//! Demonstrates the **FMT.23** recipe for PMAT-133 (format coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GGUF metadata key naming convention (llama.cpp).
//!
//! Run with: cargo run --example format_gguf_metadata_key
//!
//! Added by PMAT-133 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Namespace {
    General,
    Tokenizer,
    Llama,
    Mistral,
    Phi,
    Qwen,
    UnknownArch,
}

#[derive(Debug, PartialEq)]
pub enum KeyVerdict {
    Ok {
        namespace: Namespace,
        suffix: String,
    },
    Empty,
    NoNamespaceSeparator,
    EmptyNamespace,
    EmptySuffix,
    InvalidChars,
    AllUppercase,
}

pub fn classify(key: &str) -> KeyVerdict {
    if key.is_empty() {
        return KeyVerdict::Empty;
    }
    if key.chars().all(|c| c.is_ascii_uppercase() || c == '.') {
        return KeyVerdict::AllUppercase;
    }
    let Some((ns, suffix)) = key.split_once('.') else {
        return KeyVerdict::NoNamespaceSeparator;
    };
    if ns.is_empty() {
        return KeyVerdict::EmptyNamespace;
    }
    if suffix.is_empty() {
        return KeyVerdict::EmptySuffix;
    }
    if !is_valid_chars(key) {
        return KeyVerdict::InvalidChars;
    }
    let namespace = match ns {
        "general" => Namespace::General,
        "tokenizer" => Namespace::Tokenizer,
        "llama" => Namespace::Llama,
        "mistral" => Namespace::Mistral,
        "phi" => Namespace::Phi,
        "qwen" => Namespace::Qwen,
        _ => Namespace::UnknownArch,
    };
    KeyVerdict::Ok {
        namespace,
        suffix: suffix.to_string(),
    }
}

fn is_valid_chars(s: &str) -> bool {
    s.chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '.')
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("format_gguf_metadata_key")?;

    for k in [
        "general.architecture",
        "tokenizer.ggml.tokens",
        "llama.attention.head_count",
        "qwen.rope.theta",
        "newarch.field",
        "no_separator",
        "GENERAL.NAME",
        ".empty_ns",
        "general.",
    ] {
        println!("{k:<35}  →  {:?}", classify(k));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn general_namespace_recognised() {
        let v = classify("general.architecture");
        assert!(matches!(
            v,
            KeyVerdict::Ok {
                namespace: Namespace::General,
                ..
            }
        ));
    }

    #[test]
    fn llama_arch_recognised() {
        let v = classify("llama.attention.head_count");
        assert!(matches!(
            v,
            KeyVerdict::Ok {
                namespace: Namespace::Llama,
                ..
            }
        ));
    }

    #[test]
    fn unknown_namespace_categorised() {
        let v = classify("randomarch.field");
        assert!(matches!(
            v,
            KeyVerdict::Ok {
                namespace: Namespace::UnknownArch,
                ..
            }
        ));
    }

    #[test]
    fn empty_key_rejected() {
        assert_eq!(classify(""), KeyVerdict::Empty);
    }

    #[test]
    fn no_dot_rejected() {
        assert_eq!(classify("flatkey"), KeyVerdict::NoNamespaceSeparator);
    }

    #[test]
    fn empty_namespace_rejected() {
        assert_eq!(classify(".suffix"), KeyVerdict::EmptyNamespace);
    }

    #[test]
    fn empty_suffix_rejected() {
        assert_eq!(classify("general."), KeyVerdict::EmptySuffix);
    }

    #[test]
    fn uppercase_rejected() {
        assert_eq!(classify("GENERAL.NAME"), KeyVerdict::AllUppercase);
    }

    #[test]
    fn invalid_chars_rejected() {
        let v = classify("general.has space");
        assert_eq!(v, KeyVerdict::InvalidChars);
    }

    #[test]
    fn dotted_suffix_extracted() {
        // tokenizer.ggml.tokens — "tokenizer" / "ggml.tokens".
        if let KeyVerdict::Ok { suffix, .. } = classify("tokenizer.ggml.tokens") {
            assert_eq!(suffix, "ggml.tokens");
        }
    }

    #[test]
    fn underscore_in_suffix_allowed() {
        let v = classify("llama.attention.head_count");
        assert!(matches!(v, KeyVerdict::Ok { .. }));
    }
}
