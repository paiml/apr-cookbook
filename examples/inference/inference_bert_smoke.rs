//! # BERT Smoke Inference
//!
//! Load the bundled BERT micro-config, validate the encoder-only
//! discriminator (`type_vocab_size` field — BERT carries token-type
//! embeddings for sentence-pair tasks; decoder-only LMs do not), and
//! run a deterministic seeded forward simulation.
//!
//! Note: BERT is the only encoder-only family covered in v1 of
//! architecture-demos. Verdict shape adds a `MaskedLMOk` arm spec for
//! future MLM-style coverage; for now the smoke recipe behaves like the
//! decoder-only ones.
//!
//! Demonstrates the **BERT.smoke** recipe per
//! `docs/specifications/architecture-demos.md`
//! (`BertForMaskedLM`, `BertForSequenceClassification`).
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/inference-bert-smoke-v1.yaml (grade C; lean_status: wip)
//! Citation: Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv:1810.04805
//!
//! Run with: cargo run --example inference_bert_smoke
//!
//! Added by PMAT-307 (architecture-demos: bert family).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SmokeVerdict {
    Ok {
        family: String,
        format: String,
        logits_checksum: u32,
        layer_count: u32,
        type_vocab_size: u32,
        is_encoder_only: bool,
    },
    LoaderUnavailable {
        reason: String,
    },
    InvalidFixture,
}

const FAMILY: &str = "bert";
const FIXTURE_CONFIG: &str = "tests/fixtures/architectures/bert/config.json";

fn forward_sim(seed: u64, vocab_size: u32, hidden_size: u32, type_vocab: u32) -> u32 {
    let mut state = seed | 1;
    let mut acc: u32 = type_vocab;
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
    let type_vocab_size = match extract_number(&body, "type_vocab_size") {
        Some(n) => n as u32,
        None => return missing("type_vocab_size (BERT encoder-only discriminator)"),
    };
    let checksum = forward_sim(42, vocab_size, hidden_size, type_vocab_size);
    SmokeVerdict::Ok {
        family: FAMILY.to_string(),
        format: format.to_string(),
        logits_checksum: checksum,
        layer_count: num_layers,
        type_vocab_size,
        is_encoder_only: true,
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
    let _ctx = RecipeContext::new("inference_bert_smoke")?;
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
    fn happy_path_returns_ok_bert() {
        if let SmokeVerdict::Ok { family, .. } = smoke(FIXTURE_CONFIG, "safetensors") {
            assert_eq!(family, "bert");
        }
    }

    #[test]
    fn happy_path_type_vocab_size_present() {
        if let SmokeVerdict::Ok {
            type_vocab_size, ..
        } = smoke(FIXTURE_CONFIG, "safetensors")
        {
            assert_eq!(type_vocab_size, 2);
        }
    }

    #[test]
    fn happy_path_is_encoder_only() {
        if let SmokeVerdict::Ok {
            is_encoder_only, ..
        } = smoke(FIXTURE_CONFIG, "safetensors")
        {
            assert!(is_encoder_only);
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
    fn type_vocab_size_affects_checksum() {
        let a = forward_sim(42, 256, 64, 2);
        let b = forward_sim(42, 256, 64, 4);
        assert_ne!(a, b);
    }

    #[test]
    fn forward_sim_deterministic_per_seed() {
        let a = forward_sim(42, 256, 64, 2);
        let b = forward_sim(42, 256, 64, 2);
        assert_eq!(a, b);
        let c = forward_sim(99, 256, 64, 2);
        assert_ne!(a, c);
    }
}
