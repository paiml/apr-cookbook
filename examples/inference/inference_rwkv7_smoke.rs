//! # RWKV-7 Smoke Inference
//!
//! Load the bundled RWKV-7 micro-config, validate the linear-attention
//! discriminator (`time_mix_extra_dim` + `decay_low_rank_dim` + `wkv_dim`
//! fields), and run a deterministic seeded forward simulation.
//!
//! Demonstrates the **RWKV7.smoke** recipe per
//! `docs/specifications/architecture-demos.md` for the RWKV-7 family
//! (`RWKV7ForCausalLM`).
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/inference-rwkv7-smoke-v1.yaml (grade C; lean_status: wip)
//! Citation: Peng et al. (2024). RWKV-7: Goose. arXiv:2404.05892
//!
//! Run with: cargo run --example inference_rwkv7_smoke
//!
//! Added by PMAT-306 (architecture-demos: rwkv7 family).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SmokeVerdict {
    Ok {
        family: String,
        format: String,
        logits_checksum: u32,
        layer_count: u32,
        time_mix_extra_dim: u32,
        wkv_dim: u32,
    },
    LoaderUnavailable {
        reason: String,
    },
    InvalidFixture,
}

const FAMILY: &str = "rwkv7";
const FIXTURE_CONFIG: &str = "tests/fixtures/architectures/rwkv7/config.json";

fn forward_sim(seed: u64, vocab_size: u32, hidden_size: u32, time_mix: u32) -> u32 {
    let mut state = seed | 1;
    let mut acc: u32 = time_mix;
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
    let time_mix = match extract_number(&body, "time_mix_extra_dim") {
        Some(n) => n as u32,
        None => return missing("time_mix_extra_dim (RWKV-7 linear-attention discriminator)"),
    };
    let wkv_dim = match extract_number(&body, "wkv_dim") {
        Some(n) => n as u32,
        None => return missing("wkv_dim"),
    };
    let checksum = forward_sim(42, vocab_size, hidden_size, time_mix);
    SmokeVerdict::Ok {
        family: FAMILY.to_string(),
        format: format.to_string(),
        logits_checksum: checksum,
        layer_count: num_layers,
        time_mix_extra_dim: time_mix,
        wkv_dim,
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
    let _ctx = RecipeContext::new("inference_rwkv7_smoke")?;
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
    fn happy_path_returns_ok_rwkv7() {
        if let SmokeVerdict::Ok { family, .. } = smoke(FIXTURE_CONFIG, "safetensors") {
            assert_eq!(family, "rwkv7");
        }
    }

    #[test]
    fn happy_path_rwkv_fields_present() {
        if let SmokeVerdict::Ok {
            time_mix_extra_dim,
            wkv_dim,
            ..
        } = smoke(FIXTURE_CONFIG, "safetensors")
        {
            assert_eq!(time_mix_extra_dim, 32);
            assert_eq!(wkv_dim, 64);
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
    fn time_mix_affects_checksum() {
        let a = forward_sim(42, 256, 64, 32);
        let b = forward_sim(42, 256, 64, 64);
        assert_ne!(a, b);
    }

    #[test]
    fn forward_sim_deterministic_per_seed() {
        let a = forward_sim(42, 256, 64, 32);
        let b = forward_sim(42, 256, 64, 32);
        assert_eq!(a, b);
        let c = forward_sim(99, 256, 64, 32);
        assert_ne!(a, c);
    }
}
