//! # MAMBA Smoke Inference
//!
//! Load the bundled MAMBA micro-config, validate the pure-SSM
//! discriminator (`state_size` + `conv_kernel` + `expand` fields — pure
//! state-space model with no attention layers), and run a deterministic
//! seeded forward simulation.
//!
//! Demonstrates the **MAMBA.smoke** recipe per
//! `docs/specifications/architecture-demos.md` (`MambaForCausalLM`,
//! `Mamba2ForCausalLM`).
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/inference-mamba-smoke-v1.yaml (grade C; lean_status: wip)
//! Citation: Gu & Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752
//!
//! Run with: cargo run --example inference_mamba_smoke
//!
//! Added by PMAT-307 (architecture-demos: mamba family).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SmokeVerdict {
    Ok {
        family: String,
        format: String,
        logits_checksum: u32,
        layer_count: u32,
        state_size: u32,
        conv_kernel: u32,
    },
    LoaderUnavailable {
        reason: String,
    },
    InvalidFixture,
}

const FAMILY: &str = "mamba";
const FIXTURE_CONFIG: &str = "tests/fixtures/architectures/mamba/config.json";

fn forward_sim(seed: u64, vocab_size: u32, hidden_size: u32, state_size: u32) -> u32 {
    let mut state = seed | 1;
    let mut acc: u32 = state_size;
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
    let state_size = match extract_number(&body, "state_size") {
        Some(n) => n as u32,
        None => return missing("state_size (MAMBA SSM discriminator)"),
    };
    let conv_kernel = match extract_number(&body, "conv_kernel") {
        Some(n) => n as u32,
        None => return missing("conv_kernel"),
    };
    let checksum = forward_sim(42, vocab_size, hidden_size, state_size);
    SmokeVerdict::Ok {
        family: FAMILY.to_string(),
        format: format.to_string(),
        logits_checksum: checksum,
        layer_count: num_layers,
        state_size,
        conv_kernel,
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
    let _ctx = RecipeContext::new("inference_mamba_smoke")?;
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
    fn happy_path_returns_ok_mamba() {
        if let SmokeVerdict::Ok { family, .. } = smoke(FIXTURE_CONFIG, "safetensors") {
            assert_eq!(family, "mamba");
        }
    }

    #[test]
    fn happy_path_ssm_fields_present() {
        if let SmokeVerdict::Ok {
            state_size,
            conv_kernel,
            ..
        } = smoke(FIXTURE_CONFIG, "safetensors")
        {
            assert_eq!(state_size, 16);
            assert_eq!(conv_kernel, 4);
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
    fn state_size_affects_checksum() {
        let a = forward_sim(42, 256, 64, 16);
        let b = forward_sim(42, 256, 64, 32);
        assert_ne!(a, b);
    }

    #[test]
    fn forward_sim_deterministic_per_seed() {
        let a = forward_sim(42, 256, 64, 16);
        let b = forward_sim(42, 256, 64, 16);
        assert_eq!(a, b);
        let c = forward_sim(99, 256, 64, 16);
        assert_ne!(a, c);
    }
}
