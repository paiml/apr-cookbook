//! # apr ollama embed --dim — Embedding Dimension Validator
//!
//! Each Ollama embedding model has a fixed output dimension (e.g.,
//! `mxbai-embed-large` = 1024, `nomic-embed-text` = 768, `bge-large` =
//! 1024). When the user pins a `--dim`, it must match the model's
//! native dimension. This recipe builds the lookup + validator.
//!
//! Demonstrates the **OLLAMA.5** recipe for PMAT-120 (apr ollama coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender OLLAMA-001 + Ollama embedding model registry
//!
//! Run with: cargo run --example cli_ollama_embed_dim_validator
//!
//! Added by PMAT-120 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DimVerdict {
    Ok { dim: u32 },
    UnknownModel,
    DimMismatch { expected: u32, requested: u32 },
}

pub fn lookup_native_dim(model: &str) -> Option<u32> {
    match model {
        "mxbai-embed-large" => Some(1024),
        "bge-large-en-v1.5" => Some(1024),
        "bge-base-en-v1.5" => Some(768),
        "bge-small-en-v1.5" => Some(384),
        "nomic-embed-text" => Some(768),
        "all-MiniLM-L6-v2" => Some(384),
        "snowflake-arctic-embed" => Some(1024),
        _ => None,
    }
}

pub fn validate(model: &str, requested_dim: Option<u32>) -> DimVerdict {
    let Some(native) = lookup_native_dim(model) else {
        return DimVerdict::UnknownModel;
    };
    match requested_dim {
        None => DimVerdict::Ok { dim: native },
        Some(d) if d == native => DimVerdict::Ok { dim: native },
        Some(d) => DimVerdict::DimMismatch {
            expected: native,
            requested: d,
        },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_ollama_embed_dim_validator")?;

    let cases = [
        ("mxbai-embed-large", None),
        ("mxbai-embed-large", Some(1024)),
        ("mxbai-embed-large", Some(512)),
        ("nomic-embed-text", Some(768)),
        ("unknown-model", None),
    ];
    for (m, d) in cases {
        println!("{m:<24} dim={d:?}  →  {:?}", validate(m, d));
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
    fn known_model_no_request_returns_native() {
        let v = validate("mxbai-embed-large", None);
        assert_eq!(v, DimVerdict::Ok { dim: 1024 });
    }

    #[test]
    fn matching_dim_passes() {
        assert_eq!(
            validate("nomic-embed-text", Some(768)),
            DimVerdict::Ok { dim: 768 }
        );
    }

    #[test]
    fn mismatched_dim_rejected() {
        let v = validate("nomic-embed-text", Some(512));
        assert!(matches!(
            v,
            DimVerdict::DimMismatch {
                expected: 768,
                requested: 512
            }
        ));
    }

    #[test]
    fn unknown_model_rejected() {
        assert_eq!(validate("typo-embed", Some(768)), DimVerdict::UnknownModel);
    }

    #[test]
    fn all_known_dims_in_supported_set() {
        // Sanity: every model returns one of {384, 768, 1024}.
        let models = [
            "mxbai-embed-large",
            "bge-large-en-v1.5",
            "bge-base-en-v1.5",
            "bge-small-en-v1.5",
            "nomic-embed-text",
            "all-MiniLM-L6-v2",
            "snowflake-arctic-embed",
        ];
        for m in models {
            let d = lookup_native_dim(m).unwrap();
            assert!(matches!(d, 384 | 768 | 1024), "{m} gave {d}");
        }
    }

    #[test]
    fn small_models_are_384() {
        assert_eq!(lookup_native_dim("bge-small-en-v1.5"), Some(384));
        assert_eq!(lookup_native_dim("all-MiniLM-L6-v2"), Some(384));
    }

    #[test]
    fn base_models_are_768() {
        assert_eq!(lookup_native_dim("bge-base-en-v1.5"), Some(768));
        assert_eq!(lookup_native_dim("nomic-embed-text"), Some(768));
    }

    #[test]
    fn large_models_are_1024() {
        assert_eq!(lookup_native_dim("mxbai-embed-large"), Some(1024));
        assert_eq!(lookup_native_dim("bge-large-en-v1.5"), Some(1024));
    }
}
