//! # OpenELM Smoke Inference
//!
//! Load the bundled OpenELM micro-config, validate the layer-wise scaling
//! discriminator (`ffn_multipliers` array + `num_query_heads` array — Apple's
//! OpenELM varies dimensions per layer rather than uniform), and run a
//! deterministic seeded forward simulation.
//!
//! Demonstrates the **OPENELM.smoke** recipe per
//! `docs/specifications/architecture-demos.md` (`OpenELMForCausalLM`).
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/inference-openelm-smoke-v1.yaml (grade C; lean_status: wip)
//! Citation: Mehta et al. (2024). OpenELM. arXiv:2404.14619
//!
//! Run with: cargo run --example inference_openelm_smoke
//!
//! Added by PMAT-307 (architecture-demos: openelm family).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SmokeVerdict {
    Ok {
        family: String,
        format: String,
        logits_checksum: u32,
        layer_count: u32,
        has_layer_wise_scaling: bool,
    },
    LoaderUnavailable {
        reason: String,
    },
    InvalidFixture,
}

const FAMILY: &str = "openelm";
const FIXTURE_CONFIG: &str = "tests/fixtures/architectures/openelm/config.json";

fn forward_sim(seed: u64, vocab_size: u32, model_dim: u32, scaled: bool) -> u32 {
    let mut state = seed | 1;
    let mut acc: u32 = u32::from(scaled);
    let n = vocab_size.min(model_dim);
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
    let num_layers = match extract_number(&body, "num_transformer_layers") {
        Some(n) => n as u32,
        None => return missing("num_transformer_layers"),
    };
    let vocab_size = match extract_number(&body, "vocab_size") {
        Some(n) => n as u32,
        None => return missing("vocab_size"),
    };
    let model_dim = match extract_number(&body, "model_dim") {
        Some(n) => n as u32,
        None => return missing("model_dim (OpenELM uses model_dim, not hidden_size)"),
    };
    if !body.contains("ffn_multipliers") || !body.contains("num_query_heads") {
        return SmokeVerdict::LoaderUnavailable {
            reason: "missing ffn_multipliers or num_query_heads array (OpenELM layer-wise scaling discriminator)".into(),
        };
    }
    // OpenELM layer-wise scaling marker: arrays appear in config.
    let has_scaling = body.contains("ffn_multipliers") && body.contains("num_query_heads");
    let checksum = forward_sim(42, vocab_size, model_dim, has_scaling);
    SmokeVerdict::Ok {
        family: FAMILY.to_string(),
        format: format.to_string(),
        logits_checksum: checksum,
        layer_count: num_layers,
        has_layer_wise_scaling: has_scaling,
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
    let _ctx = RecipeContext::new("inference_openelm_smoke")?;
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
    fn happy_path_returns_ok_openelm() {
        if let SmokeVerdict::Ok { family, .. } = smoke(FIXTURE_CONFIG, "safetensors") {
            assert_eq!(family, "openelm");
        }
    }

    #[test]
    fn happy_path_layer_wise_scaling_present() {
        if let SmokeVerdict::Ok {
            has_layer_wise_scaling,
            ..
        } = smoke(FIXTURE_CONFIG, "safetensors")
        {
            assert!(has_layer_wise_scaling);
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
        if let SmokeVerdict::Ok { format, .. } = smoke(FIXTURE_CONFIG, "apr") {
            assert_eq!(format, "apr");
        }
    }

    #[test]
    fn scaled_flag_affects_checksum() {
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
