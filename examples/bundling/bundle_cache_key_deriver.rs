//! # Bundle Cache Key Deriver
//!
//! Cache key = `{model}-{format}-{quant}-{version}-{shorthash}`. Used
//! for filesystem cache lookup of pre-built bundles. Components must
//! be normalized (lowercase, no special chars) to avoid case-sensitive
//! filesystem mismatches. This recipe builds the deriver.
//!
//! Demonstrates the **BUNDLE.14** recipe for PMAT-133 (bundling coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HuggingFace Hub model cache key conventions.
//!
//! Run with: cargo run --example bundle_cache_key_deriver
//!
//! Added by PMAT-133 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum KeyVerdict {
    Ok { key: String },
    EmptyComponent { which: &'static str },
    InvalidChars { which: &'static str },
    HashWrongLength { got: usize },
}

const HASH_LEN: usize = 8;

pub fn derive(
    model: &str,
    format: &str,
    quant: &str,
    version: &str,
    shorthash: &str,
) -> KeyVerdict {
    for (label, value) in [
        ("model", model),
        ("format", format),
        ("quant", quant),
        ("version", version),
        ("shorthash", shorthash),
    ] {
        if value.is_empty() {
            return KeyVerdict::EmptyComponent { which: label };
        }
        if !is_safe_chars(value) {
            return KeyVerdict::InvalidChars { which: label };
        }
    }
    if shorthash.len() != HASH_LEN {
        return KeyVerdict::HashWrongLength {
            got: shorthash.len(),
        };
    }
    let key = format!(
        "{}-{}-{}-{}-{}",
        normalize(model),
        normalize(format),
        normalize(quant),
        normalize(version),
        normalize(shorthash)
    );
    KeyVerdict::Ok { key }
}

fn is_safe_chars(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-')
}

fn normalize(s: &str) -> String {
    s.to_ascii_lowercase().replace('_', "-")
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("bundle_cache_key_deriver")?;

    println!(
        "{:?}",
        derive("Llama-3", "GGUF", "Q4_K_M", "1.0.0", "deadbeef")
    );
    println!("{:?}", derive("", "gguf", "q4", "1.0.0", "deadbeef"));
    println!("{:?}", derive("llama-3", "gguf", "q4", "1.0.0", "tooshort"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deriver_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_key_derived_lowercase() {
        let v = derive("Llama-3", "GGUF", "Q4_K_M", "1.0.0", "deadbeef");
        if let KeyVerdict::Ok { key } = v {
            assert_eq!(key, "llama-3-gguf-q4-k-m-1.0.0-deadbeef");
        }
    }

    #[test]
    fn empty_component_rejected() {
        let v = derive("", "gguf", "q4", "1.0", "deadbeef");
        assert!(matches!(v, KeyVerdict::EmptyComponent { which: "model" }));
    }

    #[test]
    fn invalid_chars_rejected() {
        let v = derive("model name", "gguf", "q4", "1.0.0", "deadbeef");
        assert!(matches!(v, KeyVerdict::InvalidChars { which: "model" }));
    }

    #[test]
    fn short_hash_rejected() {
        let v = derive("m", "gguf", "q4", "1.0.0", "abc");
        assert!(matches!(v, KeyVerdict::HashWrongLength { got: 3 }));
    }

    #[test]
    fn long_hash_rejected() {
        let v = derive("m", "gguf", "q4", "1.0.0", "deadbeefcafe");
        assert!(matches!(v, KeyVerdict::HashWrongLength { got: 12 }));
    }

    #[test]
    fn underscore_normalized_to_dash() {
        let v = derive("a_b", "gguf", "q4", "1.0.0", "deadbeef");
        if let KeyVerdict::Ok { key } = v {
            assert!(!key.contains('_'));
            assert!(key.contains("a-b"));
        }
    }

    #[test]
    fn dots_in_version_preserved() {
        let v = derive("m", "gguf", "q4", "1.2.3", "deadbeef");
        if let KeyVerdict::Ok { key } = v {
            assert!(key.contains("1.2.3"));
        }
    }

    #[test]
    fn deterministic_across_calls() {
        let a = derive("m", "gguf", "q4", "1.0.0", "deadbeef");
        let b = derive("m", "gguf", "q4", "1.0.0", "deadbeef");
        assert_eq!(a, b);
    }

    #[test]
    fn different_inputs_yield_different_keys() {
        let a = derive("m", "gguf", "q4", "1.0.0", "deadbeef");
        let b = derive("m", "gguf", "q5", "1.0.0", "deadbeef");
        assert_ne!(a, b);
    }
}
