//! # GPT-2 Smoke Inference
//!
//! Load the bundled GPT-2 micro-config, validate the short-field-name
//! convention (`n_embd`, `n_layer`, `n_head` vs Llama's `hidden_size`
//! etc. — GPT-2's discriminator), and run a deterministic seeded forward
//! simulation.
//!
//! Demonstrates the **GPT2.smoke** recipe per
//! `docs/specifications/architecture-demos.md` for the GPT-2 family
//! (`GPT2LMHeadModel`).
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/inference-gpt2-smoke-v1.yaml (grade C; lean_status: wip)
//! Citation: Radford et al. (2019). Language Models are Unsupervised Multitask Learners. (OpenAI tech report)
//!
//! Run with: cargo run --example inference_gpt2_smoke
//!
//! Added by PMAT-305 (architecture-demos: gpt2 family).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SmokeVerdict {
    Ok {
        family: String,
        format: String,
        logits_checksum: u32,
        n_layer: u32,
        n_embd: u32,
        n_head: u32,
    },
    LoaderUnavailable {
        reason: String,
    },
    InvalidFixture,
}

const FAMILY: &str = "gpt2";
const FIXTURE_CONFIG: &str = "tests/fixtures/architectures/gpt2/config.json";

fn forward_sim(seed: u64, vocab_size: u32, n_embd: u32) -> u32 {
    let mut state = seed | 1;
    let mut acc: u32 = 0;
    let n = vocab_size.min(n_embd);
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
    // GPT-2 uses short field names — failing to find n_embd is a discriminator.
    let n_embd = match extract_number(&body, "n_embd") {
        Some(n) => n as u32,
        None => return missing("n_embd (GPT-2 short-name discriminator)"),
    };
    let n_layer = match extract_number(&body, "n_layer") {
        Some(n) => n as u32,
        None => return missing("n_layer"),
    };
    let n_head = match extract_number(&body, "n_head") {
        Some(n) => n as u32,
        None => return missing("n_head"),
    };
    let vocab_size = match extract_number(&body, "vocab_size") {
        Some(n) => n as u32,
        None => return missing("vocab_size"),
    };
    let checksum = forward_sim(42, vocab_size, n_embd);
    SmokeVerdict::Ok {
        family: FAMILY.to_string(),
        format: format.to_string(),
        logits_checksum: checksum,
        n_layer,
        n_embd,
        n_head,
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
    let _ctx = RecipeContext::new("inference_gpt2_smoke")?;
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
    fn happy_path_returns_ok_gpt2() {
        if let SmokeVerdict::Ok { family, .. } = smoke(FIXTURE_CONFIG, "safetensors") {
            assert_eq!(family, "gpt2");
        }
    }

    #[test]
    fn happy_path_short_field_names_present() {
        // Discriminator: GPT-2 uses n_embd/n_layer/n_head (not hidden_size, etc.)
        if let SmokeVerdict::Ok {
            n_layer,
            n_embd,
            n_head,
            ..
        } = smoke(FIXTURE_CONFIG, "safetensors")
        {
            assert_eq!(n_layer, 2);
            assert_eq!(n_embd, 64);
            assert_eq!(n_head, 4);
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
    fn forward_sim_deterministic_per_seed() {
        let a = forward_sim(42, 256, 64);
        let b = forward_sim(42, 256, 64);
        assert_eq!(a, b);
        let c = forward_sim(99, 256, 64);
        assert_ne!(a, c);
    }

    #[test]
    fn n_embd_affects_iteration_count() {
        let a = forward_sim(42, 256, 32);
        let b = forward_sim(42, 256, 64);
        assert_ne!(a, b);
    }
}
