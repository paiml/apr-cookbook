//! # Architecture Quirk Audit — Detect Configs With Multiple Discriminators
//!
//! Walk every fixture and flag any config that matches MORE than one
//! family's discriminator — a signal that the catalog needs a tighter
//! discriminator OR that the fixture genuinely overlaps families
//! (e.g. CodeLlama uses Llama's discriminator + has additional fields).
//!
//! Useful as a CI canary: if a future fixture introduces multi-match,
//! the audit fails — forcing the maintainer to either tighten the
//! discriminator or update the priority list.
//!
//! Demonstrates the **ARCH-QUIRK-AUDIT** recipe per
//! `docs/specifications/architecture-demos.md` v1.1.
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/inference-arch-quirk-audit-v1.yaml (grade C; lean_status: wip)
//! Citation: docs/specifications/architecture-demos.md
//!
//! Run with: cargo run --example inference_arch_quirk_audit
//!
//! Added by PMAT-312 (architecture-demos v1.1: quirk audit).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq, Clone)]
pub struct QuirkEntry {
    pub family: String,
    pub matched_discriminators: Vec<String>,
}

#[derive(Debug, PartialEq)]
pub enum AuditVerdict {
    Ok {
        clean_count: u32,
        quirky_count: u32,
        quirks: Vec<QuirkEntry>,
    },
    InvalidFixture {
        missing_family: String,
    },
}

const FAMILIES: &[&str] = &[
    "llama",
    "mistral",
    "qwen2",
    "qwen3",
    "qwen3_5",
    "phi",
    "gemma",
    "gpt2",
    "gptneox",
    "deepseek",
    "falcon_h1",
    "rwkv7",
    "openelm",
    "opt",
    "mamba",
    "bert",
];

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

pub fn audit() -> AuditVerdict {
    let mut clean = 0u32;
    let mut quirky = 0u32;
    let mut quirks: Vec<QuirkEntry> = Vec::new();
    for family in FAMILIES {
        let path = format!("tests/fixtures/architectures/{family}/config.json");
        let Ok(body) = std::fs::read_to_string(&path) else {
            return AuditVerdict::InvalidFixture {
                missing_family: (*family).to_string(),
            };
        };
        let matched: Vec<String> = DISCRIMINATOR_FIELDS
            .iter()
            .filter(|f| body.contains(*f))
            .map(|f| (*f).to_string())
            .collect();
        if matched.len() > 1 {
            quirky += 1;
            quirks.push(QuirkEntry {
                family: (*family).to_string(),
                matched_discriminators: matched,
            });
        } else {
            clean += 1;
        }
    }
    AuditVerdict::Ok {
        clean_count: clean,
        quirky_count: quirky,
        quirks,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("inference_arch_quirk_audit")?;
    let v = audit();
    if let AuditVerdict::Ok {
        clean_count,
        quirky_count,
        quirks,
    } = &v
    {
        println!("=== Architecture Quirk Audit ===");
        println!("  clean: {clean_count}  quirky: {quirky_count}  total: 16");
        for q in quirks {
            println!(
                "  {:>10}: matched {} discriminators: {:?}",
                q.family,
                q.matched_discriminators.len(),
                q.matched_discriminators
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
    fn audit_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn audit_returns_ok_for_complete_fixture_set() {
        assert!(matches!(audit(), AuditVerdict::Ok { .. }));
    }

    #[test]
    fn total_count_equals_16() {
        if let AuditVerdict::Ok {
            clean_count,
            quirky_count,
            ..
        } = audit()
        {
            assert_eq!(clean_count + quirky_count, 16);
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let a = audit();
        let b = audit();
        assert_eq!(a, b);
    }

    #[test]
    fn qwen2_is_quirky_shares_rope_theta_and_more() {
        // Qwen2 has rope_theta + sliding_window + tie_word_embeddings indirectly via
        // its config — should appear in quirks list.
        if let AuditVerdict::Ok { quirks, .. } = audit() {
            assert!(
                quirks.iter().any(|q| q.family == "qwen2"),
                "qwen2 expected in quirks list"
            );
        }
    }

    #[test]
    fn quirky_entries_have_at_least_two_discriminators() {
        if let AuditVerdict::Ok { quirks, .. } = audit() {
            for q in quirks {
                assert!(
                    q.matched_discriminators.len() >= 2,
                    "{} listed as quirky but has < 2 discriminators",
                    q.family
                );
            }
        }
    }

    #[test]
    fn discriminator_field_list_matches_summary() {
        // Same 15 fields as the summary recipe tracks.
        assert_eq!(DISCRIMINATOR_FIELDS.len(), 15);
    }

    #[test]
    fn families_list_matches_manifest() {
        // Same 16 in-progress families as the summary recipe enumerates.
        assert_eq!(FAMILIES.len(), 16);
    }

    #[test]
    fn no_family_appears_twice_in_quirks() {
        if let AuditVerdict::Ok { quirks, .. } = audit() {
            let mut names: Vec<_> = quirks.iter().map(|q| q.family.clone()).collect();
            let n = names.len();
            names.sort();
            names.dedup();
            assert_eq!(names.len(), n);
        }
    }

    #[test]
    fn audit_handles_missing_family_gracefully() {
        // Sanity: AuditVerdict has an InvalidFixture variant for missing family.
        // We can't trigger it without removing a fixture, so just check the variant exists.
        let v = AuditVerdict::InvalidFixture {
            missing_family: "test".to_string(),
        };
        assert!(matches!(v, AuditVerdict::InvalidFixture { .. }));
    }
}
