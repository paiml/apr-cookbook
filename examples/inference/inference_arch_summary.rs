//! # Architecture Summary — Discriminator Catalog Across All Families
//!
//! Walk every bundled `tests/fixtures/architectures/<family>/config.json`,
//! extract the family-specific discriminator field for each, and emit a
//! deterministic summary line per family. Useful as documentation: a
//! single recipe whose output is the family-discriminator catalog.
//!
//! Demonstrates the **ARCH-SUMMARY** recipe per
//! `docs/specifications/architecture-demos.md` v1.1.
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/inference-arch-summary-v1.yaml (grade C; lean_status: wip)
//! Citation: docs/specifications/architecture-demos.md (per-family discriminator catalog)
//!
//! Run with: cargo run --example inference_arch_summary
//!
//! Added by PMAT-310 (architecture-demos v1.1: summary + compare).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub struct FamilyEntry {
    pub family: String,
    pub discriminator_field: String,
    pub discriminator_value: String,
}

#[derive(Debug, PartialEq)]
pub enum SummaryVerdict {
    Ok { entries: Vec<FamilyEntry> },
    InvalidFixture { missing_family: String },
}

/// Per-family discriminator field. Mirrors what each family recipe surfaces
/// in its `SmokeVerdict::Ok` arm.
const FAMILIES: &[(&str, &str)] = &[
    ("llama", "rope_theta"),
    ("mistral", "sliding_window"),
    ("qwen2", "rope_theta"),
    ("qwen3", "head_dim"),
    ("qwen3_5", "tie_word_embeddings"),
    ("phi", "qkv_proj_fused"),
    ("gemma", "query_pre_attn_scalar"),
    ("gpt2", "n_embd"),
    ("gptneox", "use_parallel_residual"),
    ("deepseek", "n_routed_experts"),
    ("falcon_h1", "mamba_d_state"),
    ("rwkv7", "time_mix_extra_dim"),
    ("openelm", "ffn_multipliers"),
    ("opt", "do_layer_norm_before"),
    ("mamba", "state_size"),
    ("bert", "type_vocab_size"),
];

pub fn summarize() -> SummaryVerdict {
    let mut entries: Vec<FamilyEntry> = Vec::with_capacity(FAMILIES.len());
    for (family, field) in FAMILIES {
        let path = format!("tests/fixtures/architectures/{family}/config.json");
        let Ok(body) = std::fs::read_to_string(&path) else {
            return SummaryVerdict::InvalidFixture {
                missing_family: (*family).to_string(),
            };
        };
        let value = extract_value(&body, field).unwrap_or_else(|| "(absent)".to_string());
        entries.push(FamilyEntry {
            family: (*family).to_string(),
            discriminator_field: (*field).to_string(),
            discriminator_value: value,
        });
    }
    SummaryVerdict::Ok { entries }
}

/// Extract a discriminator value as a free-form string. For numbers and
/// booleans, returns the literal token. For arrays, returns "(array)".
fn extract_value(body: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let start = body.find(&needle)?;
    let after_key = &body[start + needle.len()..];
    let colon = after_key.find(':')?;
    let rest = &after_key[colon + 1..].trim_start();
    if rest.starts_with('[') {
        return Some("(array)".to_string());
    }
    let end = rest.find([',', '}', '\n']).unwrap_or(rest.len());
    Some(rest[..end].trim().to_string())
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("inference_arch_summary")?;
    let v = summarize();
    if let SummaryVerdict::Ok { entries } = &v {
        println!(
            "=== Architecture-Demos Discriminator Catalog ({} families) ===",
            entries.len()
        );
        for e in entries {
            println!(
                "  {:>10}  {:>24} = {}",
                e.family, e.discriminator_field, e.discriminator_value
            );
        }
    } else {
        println!("verdict: {v:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn all_16_families_returned() {
        if let SummaryVerdict::Ok { entries } = summarize() {
            assert_eq!(entries.len(), 16);
        } else {
            panic!("expected Ok, got InvalidFixture");
        }
    }

    #[test]
    fn each_entry_has_nonempty_fields() {
        if let SummaryVerdict::Ok { entries } = summarize() {
            for e in entries {
                assert!(!e.family.is_empty());
                assert!(!e.discriminator_field.is_empty());
                assert!(!e.discriminator_value.is_empty());
            }
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let a = summarize();
        let b = summarize();
        assert_eq!(a, b);
    }

    #[test]
    fn families_unique_in_output() {
        if let SummaryVerdict::Ok { entries } = summarize() {
            let mut names: Vec<_> = entries.iter().map(|e| e.family.clone()).collect();
            names.sort();
            let n = names.len();
            names.dedup();
            assert_eq!(names.len(), n, "duplicate family in summary output");
        }
    }

    #[test]
    fn discriminators_unique_across_families() {
        // Llama and Qwen2 share rope_theta — that's intentional. But every
        // OTHER pair should be distinct: each family has its own
        // discriminator field.
        if let SummaryVerdict::Ok { entries } = summarize() {
            let mut fields: Vec<_> = entries
                .iter()
                .map(|e| e.discriminator_field.clone())
                .collect();
            fields.sort();
            // Expected: 16 entries, but llama+qwen2 share rope_theta so 15 unique.
            fields.dedup();
            assert_eq!(fields.len(), 15, "expected 15 distinct discriminators");
        }
    }

    #[test]
    fn extract_value_handles_int() {
        let body = r#"{"head_dim": 16}"#;
        assert_eq!(extract_value(body, "head_dim"), Some("16".to_string()));
    }

    #[test]
    fn extract_value_handles_bool() {
        let body = r#"{"qkv_proj_fused": true}"#;
        assert_eq!(
            extract_value(body, "qkv_proj_fused"),
            Some("true".to_string())
        );
    }

    #[test]
    fn extract_value_handles_array() {
        let body = r#"{"ffn_multipliers": [0.5, 1.0]}"#;
        assert_eq!(
            extract_value(body, "ffn_multipliers"),
            Some("(array)".to_string())
        );
    }

    #[test]
    fn extract_value_handles_missing_key() {
        let body = r#"{"other_key": 42}"#;
        assert_eq!(extract_value(body, "missing"), None);
    }

    #[test]
    fn output_format_is_stable() {
        // Smoke: order matches FAMILIES const, so first entry is always llama.
        if let SummaryVerdict::Ok { entries } = summarize() {
            assert_eq!(entries[0].family, "llama");
            assert_eq!(entries.last().unwrap().family, "bert");
        }
    }
}
