//! # Architecture Compare — Diff Two Family Configs
//!
//! Take two HF `config.json` paths and emit a structured diff: which
//! discriminator fields they share, which they diverge on, and what
//! their family relationship is (same family / sibling families /
//! distant).
//!
//! Useful when triaging a new HF checkpoint: "is this just a CodeLlama
//! re-skin of Llama, or a genuinely different family?"
//!
//! Demonstrates the **ARCH-COMPARE** recipe per
//! `docs/specifications/architecture-demos.md` v1.1.
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/inference-arch-compare-v1.yaml (grade C; lean_status: wip)
//! Citation: docs/specifications/architecture-demos.md (per-family discriminator catalog)
//!
//! Run with: cargo run --example inference_arch_compare
//!
//! Added by PMAT-311 (architecture-demos v1.1: cross-family compare).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FamilyRelation {
    SameFamily,
    SiblingFamilies, // share at least one discriminator field
    DistantFamilies, // share no discriminator fields
}

#[derive(Debug, PartialEq)]
pub enum CompareVerdict {
    Ok {
        shared_fields: Vec<String>,
        only_in_a: Vec<String>,
        only_in_b: Vec<String>,
        relation: FamilyRelation,
    },
    InvalidFixture {
        which: &'static str,
    },
}

/// Discriminator fields tracked across the 16 in-progress families.
const DISCRIMINATOR_FIELDS: &[&str] = &[
    "rope_theta",
    "sliding_window",
    "head_dim",
    "tie_word_embeddings",
    "qkv_proj_fused",
    "query_pre_attn_scalar",
    "n_embd",
    "use_parallel_residual",
    "n_routed_experts",
    "mamba_d_state",
    "time_mix_extra_dim",
    "ffn_multipliers",
    "do_layer_norm_before",
    "state_size",
    "type_vocab_size",
];

pub fn compare(path_a: &str, path_b: &str) -> CompareVerdict {
    let Ok(body_a) = std::fs::read_to_string(path_a) else {
        return CompareVerdict::InvalidFixture { which: "a" };
    };
    let Ok(body_b) = std::fs::read_to_string(path_b) else {
        return CompareVerdict::InvalidFixture { which: "b" };
    };
    compare_bodies(&body_a, &body_b)
}

pub fn compare_bodies(body_a: &str, body_b: &str) -> CompareVerdict {
    let mut shared: Vec<String> = Vec::new();
    let mut only_a: Vec<String> = Vec::new();
    let mut only_b: Vec<String> = Vec::new();
    for field in DISCRIMINATOR_FIELDS {
        let in_a = body_a.contains(field);
        let in_b = body_b.contains(field);
        match (in_a, in_b) {
            (true, true) => shared.push((*field).to_string()),
            (true, false) => only_a.push((*field).to_string()),
            (false, true) => only_b.push((*field).to_string()),
            (false, false) => {}
        }
    }
    let relation = if !shared.is_empty() && only_a.is_empty() && only_b.is_empty() {
        FamilyRelation::SameFamily
    } else if !shared.is_empty() {
        FamilyRelation::SiblingFamilies
    } else {
        FamilyRelation::DistantFamilies
    };
    CompareVerdict::Ok {
        shared_fields: shared,
        only_in_a: only_a,
        only_in_b: only_b,
        relation,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("inference_arch_compare")?;
    let pairs = [
        ("llama", "qwen2"),
        ("llama", "mistral"),
        ("llama", "bert"),
        ("qwen2", "qwen3"),
        ("phi", "deepseek"),
    ];
    for (a, b) in pairs {
        let pa = format!("tests/fixtures/architectures/{a}/config.json");
        let pb = format!("tests/fixtures/architectures/{b}/config.json");
        if let CompareVerdict::Ok {
            shared_fields,
            relation,
            ..
        } = compare(&pa, &pb)
        {
            println!("{a:>10} vs {b:<10}: shared={shared_fields:?} relation={relation:?}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(family: &str) -> String {
        format!("tests/fixtures/architectures/{family}/config.json")
    }

    #[test]
    fn compare_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn missing_fixture_a_returns_invalid() {
        assert_eq!(
            compare("/no/such/path", &fixture("llama")),
            CompareVerdict::InvalidFixture { which: "a" }
        );
    }

    #[test]
    fn missing_fixture_b_returns_invalid() {
        assert_eq!(
            compare(&fixture("llama"), "/no/such/path"),
            CompareVerdict::InvalidFixture { which: "b" }
        );
    }

    #[test]
    fn same_fixture_yields_same_family() {
        if let CompareVerdict::Ok { relation, .. } = compare(&fixture("llama"), &fixture("llama")) {
            assert_eq!(relation, FamilyRelation::SameFamily);
        }
    }

    #[test]
    fn llama_vs_qwen2_share_rope_theta() {
        // Both have rope_theta but Qwen2 also has tie_word_embeddings/sliding_window etc.
        if let CompareVerdict::Ok { shared_fields, .. } =
            compare(&fixture("llama"), &fixture("qwen2"))
        {
            assert!(shared_fields.contains(&"rope_theta".to_string()));
        }
    }

    #[test]
    fn llama_vs_qwen2_relation_is_sibling() {
        if let CompareVerdict::Ok { relation, .. } = compare(&fixture("llama"), &fixture("qwen2")) {
            assert_eq!(relation, FamilyRelation::SiblingFamilies);
        }
    }

    #[test]
    fn llama_vs_mistral_shared_includes_sliding_window_only_for_mistral() {
        if let CompareVerdict::Ok {
            only_in_b,
            shared_fields,
            ..
        } = compare(&fixture("llama"), &fixture("mistral"))
        {
            assert!(only_in_b.contains(&"sliding_window".to_string()));
            assert!(shared_fields.contains(&"rope_theta".to_string()));
        }
    }

    #[test]
    fn llama_vs_bert_relation_is_distant() {
        // Bert has type_vocab_size; Llama has rope_theta. No shared discriminator.
        if let CompareVerdict::Ok { relation, .. } = compare(&fixture("llama"), &fixture("bert")) {
            assert_eq!(relation, FamilyRelation::DistantFamilies);
        }
    }

    #[test]
    fn qwen2_vs_qwen3_share_rope_theta_diverge_on_head_dim() {
        if let CompareVerdict::Ok {
            shared_fields,
            only_in_b,
            ..
        } = compare(&fixture("qwen2"), &fixture("qwen3"))
        {
            // Both have rope_theta
            assert!(shared_fields.contains(&"rope_theta".to_string()));
            // Only qwen3 has head_dim
            assert!(only_in_b.contains(&"head_dim".to_string()));
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let a = compare(&fixture("llama"), &fixture("qwen2"));
        let b = compare(&fixture("llama"), &fixture("qwen2"));
        assert_eq!(a, b);
    }

    #[test]
    fn order_swap_yields_swapped_only_fields() {
        // compare(a, b) only_in_a should equal compare(b, a) only_in_b.
        if let (
            CompareVerdict::Ok {
                only_in_a: a_only,
                only_in_b: b_only,
                ..
            },
            CompareVerdict::Ok {
                only_in_a: ba_only,
                only_in_b: ab_only,
                ..
            },
        ) = (
            compare(&fixture("llama"), &fixture("phi")),
            compare(&fixture("phi"), &fixture("llama")),
        ) {
            assert_eq!(a_only, ab_only);
            assert_eq!(b_only, ba_only);
        }
    }

    #[test]
    fn discriminator_field_count_correct() {
        // 15 distinct discriminators tracked across the 16 families.
        assert_eq!(DISCRIMINATOR_FIELDS.len(), 15);
    }
}
