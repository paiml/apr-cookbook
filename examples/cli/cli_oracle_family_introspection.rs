//! # apr oracle — Family Contract Introspection
//!
//! `apr oracle --family <FAM>` describes the contract for a known model
//! family (qwen2, llama, whisper, bert, …) — the canonical tensor naming,
//! expected dtype, vocab range, head/dim conventions, and the SPDX
//! license expectations. This recipe builds the family-spec lookup
//! decision tree as a pure function so a CI pipeline can preview which
//! family classification would be applied to an arbitrary model name.
//!
//! Demonstrates the **ORACLE.3** recipe for PMAT-093 (apr oracle coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender ORACLE-001 + Hugging Face model-card spec
//!
//! Run with: cargo run --example cli_oracle_family_introspection
//!
//! Added by PMAT-093 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FamilySpec {
    pub family: &'static str,
    pub canonical_attn_q: &'static str,
    pub default_dtype: &'static str,
    pub vocab_typical: u32,
    pub license_typical: &'static str,
}

const SPECS: &[FamilySpec] = &[
    FamilySpec {
        family: "qwen2",
        canonical_attn_q: "model.layers.{n}.self_attn.q_proj.weight",
        default_dtype: "bf16",
        vocab_typical: 152_064,
        license_typical: "Apache-2.0",
    },
    FamilySpec {
        family: "llama",
        canonical_attn_q: "model.layers.{n}.self_attn.q_proj.weight",
        default_dtype: "bf16",
        vocab_typical: 128_256,
        license_typical: "Llama-3-Community",
    },
    FamilySpec {
        family: "whisper",
        canonical_attn_q: "model.encoder.layers.{n}.self_attn.q_proj.weight",
        default_dtype: "fp16",
        vocab_typical: 51_865,
        license_typical: "MIT",
    },
    FamilySpec {
        family: "bert",
        canonical_attn_q: "encoder.layer.{n}.attention.self.query.weight",
        default_dtype: "fp32",
        vocab_typical: 30_522,
        license_typical: "Apache-2.0",
    },
];

pub fn lookup_family(family: &str) -> Option<&'static FamilySpec> {
    SPECS.iter().find(|s| s.family == family)
}

/// Heuristically classify a model name into a family. Used when the user
/// supplies just a model id and the oracle needs to pick a contract.
pub fn classify_model_name(name: &str) -> Option<&'static FamilySpec> {
    let lower = name.to_ascii_lowercase();
    if lower.contains("qwen") {
        lookup_family("qwen2")
    } else if lower.contains("llama") {
        lookup_family("llama")
    } else if lower.contains("whisper") {
        lookup_family("whisper")
    } else if lower.contains("bert") {
        lookup_family("bert")
    } else {
        None
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_oracle_family_introspection")?;

    for fam in ["qwen2", "llama", "whisper", "bert", "gpt-3"] {
        match lookup_family(fam) {
            Some(s) => println!(
                "{:>8}  q_proj={:50}  dtype={:>4}  vocab={:>7}  license={}",
                s.family, s.canonical_attn_q, s.default_dtype, s.vocab_typical, s.license_typical
            ),
            None => println!("{fam:>8}  (unknown family — not in oracle catalog)"),
        }
    }

    for name in [
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "openai/whisper-large-v3",
        "mystery/random-model",
    ] {
        match classify_model_name(name) {
            Some(s) => println!("classify {name:>40} → family {}", s.family),
            None => println!("classify {name:>40} → unclassified"),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn introspection_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn known_families_have_specs() {
        for f in ["qwen2", "llama", "whisper", "bert"] {
            assert!(lookup_family(f).is_some(), "missing spec for {f}");
        }
    }

    #[test]
    fn unknown_family_returns_none() {
        assert!(lookup_family("gpt-3").is_none());
        assert!(lookup_family("").is_none());
    }

    #[test]
    fn classify_qwen_variants() {
        let s = classify_model_name("Qwen/Qwen2.5-Coder-7B-Instruct").unwrap();
        assert_eq!(s.family, "qwen2");
        let s2 = classify_model_name("qwen-7b-q4km").unwrap();
        assert_eq!(s2.family, "qwen2");
    }

    #[test]
    fn classify_unknown_returns_none() {
        // Unknown-name policy: oracle refuses to guess; the operator must
        // pick a family explicitly via --family.
        assert!(classify_model_name("acme/proprietary-12b").is_none());
    }

    #[test]
    fn classify_is_case_insensitive() {
        assert!(classify_model_name("LLAMA-3").is_some());
        assert!(classify_model_name("BERT-base").is_some());
    }

    #[test]
    fn whisper_uses_encoder_attn_path() {
        // Whisper has both encoder + decoder. The canonical q_proj path
        // points at the encoder — this asserts we don't accidentally drift
        // to model.layers.* like the LLaMA family.
        let s = lookup_family("whisper").unwrap();
        assert!(s.canonical_attn_q.contains("encoder.layers"));
    }
}
