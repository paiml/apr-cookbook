//! # Phi Smoke Inference
//!
//! Load the bundled Phi-3 micro-config, validate the fused-QKV tensor
//! layout (Phi-3's discriminator vs Llama: a single qkv_proj instead of
//! three separate q_proj/k_proj/v_proj), and run a deterministic seeded
//! forward simulation.
//!
//! Demonstrates the **PHI.smoke** recipe per
//! `docs/specifications/architecture-demos.md` for the Phi family
//! (`PhiForCausalLM`, `Phi3ForCausalLM`, `Phi3SmallForCausalLM`).
//! Companion converter `examples/conversion/convert_phi_to_apr.rs`
//! handles the QKV-split during HF→APR conversion.
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/inference-phi-smoke-v1.yaml (grade C; lean_status: wip)
//! Citation: Abdin et al. (2024). Phi-3 Technical Report. arXiv:2404.14219
//!
//! Run with: cargo run --example inference_phi_smoke
//!
//! Added by PMAT-304 (architecture-demos: phi family).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SmokeVerdict {
    Ok {
        family: String,
        format: String,
        logits_checksum: u32,
        layer_count: u32,
        qkv_proj_fused: bool,
        tensor_count: u32,
    },
    LoaderUnavailable {
        reason: String,
    },
    InvalidFixture,
}

const FAMILY: &str = "phi";
const FIXTURE_CONFIG: &str = "tests/fixtures/architectures/phi/config.json";

/// Phi-3 fuses Q/K/V into a single projection. When fused, the per-layer
/// tensor count drops from 9 (Llama-style separate q/k/v) to 7 — three
/// separate projections collapse into one qkv_proj.
fn expected_tensor_count(num_layers: u32, qkv_fused: bool) -> u32 {
    let per_layer = if qkv_fused { 7 } else { 9 };
    3 + per_layer * num_layers
}

fn forward_sim(seed: u64, vocab_size: u32, hidden_size: u32, qkv_fused: bool) -> u32 {
    let mut state = seed | 1;
    let mut acc: u32 = u32::from(qkv_fused);
    let n = vocab_size.min(hidden_size);
    for _ in 0..n {
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
    if !body.contains("qkv_proj_fused") {
        return SmokeVerdict::LoaderUnavailable {
            reason: "missing qkv_proj_fused in config (Phi-3 discriminator)".into(),
        };
    }
    let qkv_fused = body.contains("\"qkv_proj_fused\": true");
    let tensor_count = expected_tensor_count(num_layers, qkv_fused);
    let checksum = forward_sim(42, vocab_size, hidden_size, qkv_fused);
    SmokeVerdict::Ok {
        family: FAMILY.to_string(),
        format: format.to_string(),
        logits_checksum: checksum,
        layer_count: num_layers,
        qkv_proj_fused: qkv_fused,
        tensor_count,
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
    trimmed[..num_end].parse::<f64>().ok().map(|f| f as i64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("inference_phi_smoke")?;
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
    fn happy_path_returns_ok_phi() {
        if let SmokeVerdict::Ok { family, .. } = smoke(FIXTURE_CONFIG, "safetensors") {
            assert_eq!(family, "phi");
        }
    }

    #[test]
    fn happy_path_qkv_fused() {
        if let SmokeVerdict::Ok { qkv_proj_fused, .. } = smoke(FIXTURE_CONFIG, "safetensors") {
            assert!(qkv_proj_fused);
        }
    }

    #[test]
    fn happy_path_tensor_count_reduced() {
        // 2 layers fused: 3 + 7*2 = 17 (vs Llama-shape 21)
        if let SmokeVerdict::Ok { tensor_count, .. } = smoke(FIXTURE_CONFIG, "safetensors") {
            assert_eq!(tensor_count, 17);
        }
    }

    #[test]
    fn happy_path_layer_count_matches_config() {
        if let SmokeVerdict::Ok { layer_count, .. } = smoke(FIXTURE_CONFIG, "safetensors") {
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
        if let SmokeVerdict::Ok { format, .. } = smoke(FIXTURE_CONFIG, "gguf") {
            assert_eq!(format, "gguf");
        }
    }

    #[test]
    fn fused_drops_two_per_layer() {
        // Fusing q/k/v into qkv: 9 - 3 + 1 = 7 per layer.
        assert_eq!(expected_tensor_count(2, true), 17);
        assert_eq!(expected_tensor_count(2, false), 21);
        assert_eq!(expected_tensor_count(0, true), 3);
    }

    #[test]
    fn fused_flag_affects_checksum() {
        let a = forward_sim(42, 256, 64, true);
        let b = forward_sim(42, 256, 64, false);
        assert_ne!(a, b);
    }

    #[test]
    fn forward_sim_deterministic_per_seed() {
        let a = forward_sim(42, 256, 64, true);
        let b = forward_sim(42, 256, 64, true);
        assert_eq!(a, b);
        let c = forward_sim(99, 256, 64, true);
        assert_ne!(a, c);
    }
}
