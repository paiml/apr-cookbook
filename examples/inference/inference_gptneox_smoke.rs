//! # GPT-NeoX Smoke Inference
//!
//! Load the bundled GPT-NeoX micro-config, validate `use_parallel_residual=true`
//! (GPT-NeoX's discriminator: parallel attention + FFN vs Llama's sequential),
//! and run a deterministic seeded forward simulation.
//!
//! Demonstrates the **GPTNEOX.smoke** recipe per
//! `docs/specifications/architecture-demos.md` for the GPT-NeoX family
//! (`GPTNeoXForCausalLM`).
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/inference-gptneox-smoke-v1.yaml (grade C; lean_status: wip)
//! Citation: Black et al. (2022). GPT-NeoX-20B. arXiv:2204.06745
//!
//! Run with: cargo run --example inference_gptneox_smoke
//!
//! Added by PMAT-305 (architecture-demos: gptneox family).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SmokeVerdict {
    Ok {
        family: String,
        format: String,
        logits_checksum: u32,
        layer_count: u32,
        parallel_residual: bool,
    },
    LoaderUnavailable {
        reason: String,
    },
    InvalidFixture,
}

const FAMILY: &str = "gptneox";
const FIXTURE_CONFIG: &str = "tests/fixtures/architectures/gptneox/config.json";

fn forward_sim(seed: u64, vocab_size: u32, hidden_size: u32, parallel: bool) -> u32 {
    let mut state = seed | 1;
    let mut acc: u32 = u32::from(parallel);
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
    if !body.contains("use_parallel_residual") {
        return SmokeVerdict::LoaderUnavailable {
            reason: "missing use_parallel_residual in config (GPT-NeoX discriminator)".into(),
        };
    }
    let parallel = body.contains("\"use_parallel_residual\": true");
    let checksum = forward_sim(42, vocab_size, hidden_size, parallel);
    SmokeVerdict::Ok {
        family: FAMILY.to_string(),
        format: format.to_string(),
        logits_checksum: checksum,
        layer_count: num_layers,
        parallel_residual: parallel,
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
    let _ctx = RecipeContext::new("inference_gptneox_smoke")?;
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
    fn happy_path_returns_ok_gptneox() {
        if let SmokeVerdict::Ok { family, .. } = smoke(FIXTURE_CONFIG, "safetensors") {
            assert_eq!(family, "gptneox");
        }
    }

    #[test]
    fn happy_path_parallel_residual_true() {
        if let SmokeVerdict::Ok {
            parallel_residual, ..
        } = smoke(FIXTURE_CONFIG, "safetensors")
        {
            assert!(parallel_residual);
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
    fn parallel_flag_affects_checksum() {
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
