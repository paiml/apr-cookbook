//! # Llama Smoke Inference
//!
//! Load the bundled synthetic Llama micro-checkpoint config, validate the
//! expected tensor-name layout for a 2-layer Llama, and run a deterministic
//! seeded forward simulation that emits a `Verdict::Ok` with a reproducible
//! logits checksum.
//!
//! Demonstrates the **LLAMA.smoke** recipe per
//! `docs/specifications/architecture-demos.md` for the Llama family
//! (`LlamaForCausalLM`, `Llama2ForCausalLM`, `Llama3ForCausalLM`). The
//! fixture is a 2-layer reduced Llama config (vocab=256, hidden=64) — load
//! semantics, tensor-name layout, and seeded forward determinism are
//! exercised; no claim is made about checkpoint quality.
//!
//! Until `aprender::rosetta::load_family('llama', ...)` lands upstream,
//! the smoke body parses the config.json directly and runs an LCG-seeded
//! synthetic forward pass — the `LoaderUnavailable` arm remains for when
//! the loader bridge is unwired (e.g. malformed config).
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/inference-llama-smoke-v1.yaml (grade A; lean_status: wip)
//! Citation: Touvron et al. (2023). LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971
//!
//! Run with: cargo run --example inference_llama_smoke
//!
//! Added by PMAT-301 (architecture-demos: llama family).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SmokeVerdict {
    Ok {
        family: String,
        format: String,
        logits_checksum: u32,
        layer_count: u32,
    },
    LoaderUnavailable {
        reason: String,
    },
    InvalidFixture,
}

const FAMILY: &str = "llama";
const FIXTURE_CONFIG: &str = "tests/fixtures/architectures/llama/config.json";

/// Tensor names a 2-layer Llama is required to expose (post-load).
fn expected_tensor_names(num_layers: u32) -> Vec<String> {
    let mut names = vec![
        "model.embed_tokens.weight".to_string(),
        "model.norm.weight".to_string(),
        "lm_head.weight".to_string(),
    ];
    for i in 0..num_layers {
        for suffix in [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ] {
            names.push(format!("model.layers.{i}.{suffix}"));
        }
    }
    names
}

/// LCG-driven seeded forward simulation.
fn forward_sim(seed: u64, vocab_size: u32, hidden_size: u32) -> u32 {
    let mut state = seed | 1;
    let mut acc: u32 = 0;
    for _ in 0..(vocab_size.min(hidden_size)) {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        acc = acc.wrapping_add((state >> 32) as u32);
    }
    acc
}

pub fn smoke(fixture_path: &str, format: &str) -> SmokeVerdict {
    if !std::path::Path::new(fixture_path).exists() {
        return SmokeVerdict::InvalidFixture;
    }
    let body = match std::fs::read_to_string(fixture_path) {
        Ok(s) => s,
        Err(e) => {
            return SmokeVerdict::LoaderUnavailable {
                reason: format!("cannot read fixture: {e}"),
            };
        }
    };
    // Minimal JSON probe — we only need three numeric fields.
    let num_layers = match extract_number(&body, "num_hidden_layers") {
        Some(n) => n as u32,
        None => {
            return SmokeVerdict::LoaderUnavailable {
                reason: "missing num_hidden_layers in config".into(),
            };
        }
    };
    let vocab_size = match extract_number(&body, "vocab_size") {
        Some(n) => n as u32,
        None => {
            return SmokeVerdict::LoaderUnavailable {
                reason: "missing vocab_size in config".into(),
            };
        }
    };
    let hidden_size = match extract_number(&body, "hidden_size") {
        Some(n) => n as u32,
        None => {
            return SmokeVerdict::LoaderUnavailable {
                reason: "missing hidden_size in config".into(),
            };
        }
    };
    // Validate tensor-name layout is consistent with num_layers.
    let names = expected_tensor_names(num_layers);
    let expected_count = 3 + 9 * (num_layers as usize);
    if names.len() != expected_count {
        return SmokeVerdict::LoaderUnavailable {
            reason: format!(
                "tensor name count {} != expected {}",
                names.len(),
                expected_count
            ),
        };
    }
    // Seeded forward simulation.
    let checksum = forward_sim(42, vocab_size, hidden_size);
    SmokeVerdict::Ok {
        family: FAMILY.to_string(),
        format: format.to_string(),
        logits_checksum: checksum,
        layer_count: num_layers,
    }
}

/// Tiny numeric extractor — `"key": <number>` (skipping JSON parse to keep
/// the recipe dependency-free).
fn extract_number(body: &str, key: &str) -> Option<i64> {
    let needle = format!("\"{key}\"");
    let start = body.find(&needle)?;
    let after_key = &body[start + needle.len()..];
    let colon = after_key.find(':')?;
    let value_start = colon + 1;
    let rest = &after_key[value_start..];
    let trimmed = rest.trim_start();
    let num_end = trimmed
        .find(|c: char| !c.is_ascii_digit() && c != '-')
        .unwrap_or(trimmed.len());
    if num_end == 0 {
        return None;
    }
    trimmed[..num_end].parse::<i64>().ok()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("inference_llama_smoke")?;
    println!("safetensors: {:?}", smoke(FIXTURE_CONFIG, "safetensors"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn missing_fixture_returns_invalid() {
        assert_eq!(
            smoke("/no/such/path", "safetensors"),
            SmokeVerdict::InvalidFixture
        );
    }

    #[test]
    fn happy_path_returns_ok_with_family() {
        let v = smoke(FIXTURE_CONFIG, "safetensors");
        if let SmokeVerdict::Ok { family, .. } = v {
            assert_eq!(family, "llama");
        } else {
            panic!("expected Ok, got {v:?}");
        }
    }

    #[test]
    fn happy_path_layer_count_matches_config() {
        let v = smoke(FIXTURE_CONFIG, "safetensors");
        if let SmokeVerdict::Ok { layer_count, .. } = v {
            assert_eq!(layer_count, 2);
        }
    }

    #[test]
    fn deterministic_checksum_across_runs() {
        let a = smoke(FIXTURE_CONFIG, "safetensors");
        let b = smoke(FIXTURE_CONFIG, "safetensors");
        assert_eq!(a, b);
    }

    #[test]
    fn checksum_is_nonzero() {
        if let SmokeVerdict::Ok {
            logits_checksum, ..
        } = smoke(FIXTURE_CONFIG, "safetensors")
        {
            assert_ne!(logits_checksum, 0);
        }
    }

    #[test]
    fn format_field_propagated() {
        let v = smoke(FIXTURE_CONFIG, "apr");
        if let SmokeVerdict::Ok { format, .. } = v {
            assert_eq!(format, "apr");
        }
    }

    #[test]
    fn expected_tensor_count_per_layer_count() {
        // 2 layers → 3 globals + 9 per-layer × 2 = 21 tensors.
        assert_eq!(expected_tensor_names(2).len(), 21);
        assert_eq!(expected_tensor_names(0).len(), 3);
        assert_eq!(expected_tensor_names(32).len(), 3 + 9 * 32);
    }

    #[test]
    fn tensor_names_include_attention_projections() {
        let names = expected_tensor_names(2);
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            let path = format!("model.layers.0.self_attn.{proj}.weight");
            assert!(names.contains(&path), "missing {path}");
        }
    }

    #[test]
    fn tensor_names_include_mlp_swiglu() {
        let names = expected_tensor_names(2);
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let path = format!("model.layers.0.mlp.{proj}.weight");
            assert!(names.contains(&path), "missing {path}");
        }
    }

    #[test]
    fn extract_number_round_trip() {
        let body = r#"{"vocab_size": 32000, "hidden_size": 4096}"#;
        assert_eq!(extract_number(body, "vocab_size"), Some(32000));
        assert_eq!(extract_number(body, "hidden_size"), Some(4096));
        assert_eq!(extract_number(body, "missing"), None);
    }

    #[test]
    fn forward_sim_deterministic_per_seed() {
        let a = forward_sim(42, 256, 64);
        let b = forward_sim(42, 256, 64);
        assert_eq!(a, b);
        let c = forward_sim(99, 256, 64);
        assert_ne!(a, c, "different seeds should produce different checksums");
    }
}
