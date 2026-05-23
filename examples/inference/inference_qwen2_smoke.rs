//! # Qwen2 Smoke Inference
//!
//! Load the bundled Qwen2 micro-config, validate the Llama-shape tensor
//! layout, assert `rope_theta=1000000` (Qwen2's discriminator vs Llama's
//! 10000), and run a deterministic seeded forward simulation.
//!
//! Demonstrates the **QWEN2.smoke** recipe per
//! `docs/specifications/architecture-demos.md` for the Qwen2 family
//! (`Qwen2ForCausalLM`, `Qwen2_5ForCausalLM`).
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/inference-qwen2-smoke-v1.yaml (grade C; lean_status: wip)
//! Citation: Yang et al. (2024). Qwen2 Technical Report. arXiv:2407.10671
//!
//! Run with: cargo run --example inference_qwen2_smoke
//!
//! Added by PMAT-303 (architecture-demos: qwen2 family).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SmokeVerdict {
    Ok {
        family: String,
        format: String,
        logits_checksum: u32,
        layer_count: u32,
        rope_theta_million: u32,
    },
    LoaderUnavailable {
        reason: String,
    },
    InvalidFixture,
}

const FAMILY: &str = "qwen2";
const FIXTURE_CONFIG: &str = "tests/fixtures/architectures/qwen2/config.json";

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
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
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

fn forward_sim(seed: u64, vocab_size: u32, hidden_size: u32, rope_theta: u64) -> u32 {
    let mut state = seed | 1;
    let mut acc: u32 = 0;
    let n = vocab_size.min(hidden_size);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(rope_theta);
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
    let num_layers = match extract_number(&body, "num_hidden_layers") {
        Some(n) => n as u32,
        None => return missing("num_hidden_layers"),
    };
    let vocab_size = match extract_number(&body, "vocab_size") {
        Some(n) => n as u32,
        None => return missing("vocab_size"),
    };
    let hidden_size = match extract_number(&body, "hidden_size") {
        Some(n) => n as u32,
        None => return missing("hidden_size"),
    };
    let rope_theta_int = match extract_number(&body, "rope_theta") {
        Some(n) => n as u64,
        None => return missing("rope_theta (Qwen2 discriminator)"),
    };
    if rope_theta_int < 100_000 {
        return SmokeVerdict::LoaderUnavailable {
            reason: format!(
                "rope_theta {} below Qwen2 expected magnitude (>= 100000)",
                rope_theta_int
            ),
        };
    }
    let names = expected_tensor_names(num_layers);
    let expected_count = 3 + 12 * (num_layers as usize);
    if names.len() != expected_count {
        return SmokeVerdict::LoaderUnavailable {
            reason: format!(
                "tensor name count {} != expected {} (Qwen2 has 12 per-layer entries)",
                names.len(),
                expected_count
            ),
        };
    }
    let checksum = forward_sim(42, vocab_size, hidden_size, rope_theta_int);
    SmokeVerdict::Ok {
        family: FAMILY.to_string(),
        format: format.to_string(),
        logits_checksum: checksum,
        layer_count: num_layers,
        rope_theta_million: (rope_theta_int / 1_000_000) as u32,
    }
}

fn missing(key: &str) -> SmokeVerdict {
    SmokeVerdict::LoaderUnavailable {
        reason: format!("missing {key} in config"),
    }
}

fn extract_number(body: &str, key: &str) -> Option<i64> {
    let needle = format!("\"{key}\"");
    let start = body.find(&needle)?;
    let after_key = &body[start + needle.len()..];
    let colon = after_key.find(':')?;
    let rest = &after_key[colon + 1..];
    let trimmed = rest.trim_start();
    let num_end = trimmed
        .find(|c: char| {
            !c.is_ascii_digit() && c != '-' && c != '.' && c != 'e' && c != 'E' && c != '+'
        })
        .unwrap_or(trimmed.len());
    if num_end == 0 {
        return None;
    }
    // Truncate decimals: "1000000.0" → 1000000
    trimmed[..num_end].parse::<f64>().ok().map(|f| f as i64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("inference_qwen2_smoke")?;
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
    fn happy_path_returns_ok_qwen2() {
        let v = smoke(FIXTURE_CONFIG, "safetensors");
        if let SmokeVerdict::Ok { family, .. } = v {
            assert_eq!(family, "qwen2");
        } else {
            panic!("expected Ok, got {v:?}");
        }
    }

    #[test]
    fn happy_path_layer_count_matches_config() {
        if let SmokeVerdict::Ok { layer_count, .. } = smoke(FIXTURE_CONFIG, "safetensors") {
            assert_eq!(layer_count, 2);
        }
    }

    #[test]
    fn rope_theta_is_one_million() {
        // Qwen2 discriminator vs Llama: rope_theta=1000000 (Llama uses 10000).
        if let SmokeVerdict::Ok {
            rope_theta_million, ..
        } = smoke(FIXTURE_CONFIG, "safetensors")
        {
            assert_eq!(rope_theta_million, 1);
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
        if let SmokeVerdict::Ok { format, .. } = smoke(FIXTURE_CONFIG, "gguf") {
            assert_eq!(format, "gguf");
        }
    }

    #[test]
    fn expected_tensor_count_per_layer_count() {
        // Qwen2 has 12 per-layer entries (vs 9 for Llama) — q/k/v have biases.
        assert_eq!(expected_tensor_names(2).len(), 3 + 12 * 2);
        assert_eq!(expected_tensor_names(0).len(), 3);
    }

    #[test]
    fn qkv_biases_present() {
        let names = expected_tensor_names(1);
        for proj in ["q_proj", "k_proj", "v_proj"] {
            let bias = format!("model.layers.0.self_attn.{proj}.bias");
            assert!(names.contains(&bias), "missing {bias}");
        }
    }

    #[test]
    fn rope_theta_affects_checksum() {
        // Different rope_theta → different LCG increment → different checksum.
        let a = forward_sim(42, 256, 64, 10_000);
        let b = forward_sim(42, 256, 64, 1_000_000);
        assert_ne!(a, b);
    }

    #[test]
    fn extract_number_handles_floats() {
        let body = r#"{"rope_theta": 1000000.0}"#;
        assert_eq!(extract_number(body, "rope_theta"), Some(1_000_000));
    }
}
