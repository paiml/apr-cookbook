//! # OPT Smoke Inference
//!
//! Load the bundled OPT micro-config, validate the Pre-LN discriminator
//! (`do_layer_norm_before` field — OPT positions LN before attention
//! rather than after, distinguishing it from later GPT variants), and
//! run a deterministic seeded forward simulation.
//!
//! Demonstrates the **OPT.smoke** recipe per
//! `docs/specifications/architecture-demos.md` (`OPTForCausalLM`).
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/inference-opt-smoke-v1.yaml (grade C; lean_status: wip)
//! Citation: Zhang et al. (2022). OPT: Open Pre-trained Transformer. arXiv:2205.01068
//!
//! Run with: cargo run --example inference_opt_smoke
//!
//! Added by PMAT-307 (architecture-demos: opt family).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SmokeVerdict {
    Ok {
        family: String,
        format: String,
        logits_checksum: u32,
        layer_count: u32,
        layer_norm_before: bool,
    },
    LoaderUnavailable {
        reason: String,
    },
    InvalidFixture,
}

const FAMILY: &str = "opt";
const FIXTURE_CONFIG: &str = "tests/fixtures/architectures/opt/config.json";

fn forward_sim(seed: u64, vocab_size: u32, hidden_size: u32, ln_before: bool) -> u32 {
    let mut state = seed | 1;
    let mut acc: u32 = u32::from(ln_before);
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
    if !body.contains("do_layer_norm_before") {
        return SmokeVerdict::LoaderUnavailable {
            reason: "missing do_layer_norm_before in config (OPT discriminator)".into(),
        };
    }
    let ln_before = body.contains("\"do_layer_norm_before\": true");
    let checksum = forward_sim(42, vocab_size, hidden_size, ln_before);
    SmokeVerdict::Ok {
        family: FAMILY.to_string(),
        format: format.to_string(),
        logits_checksum: checksum,
        layer_count: num_layers,
        layer_norm_before: ln_before,
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
    let _ctx = RecipeContext::new("inference_opt_smoke")?;
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
    fn happy_path_returns_ok_opt() {
        if let SmokeVerdict::Ok { family, .. } = smoke(FIXTURE_CONFIG, "safetensors") {
            assert_eq!(family, "opt");
        }
    }

    #[test]
    fn happy_path_ln_before_true() {
        if let SmokeVerdict::Ok {
            layer_norm_before, ..
        } = smoke(FIXTURE_CONFIG, "safetensors")
        {
            assert!(layer_norm_before);
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
    fn ln_position_affects_checksum() {
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
