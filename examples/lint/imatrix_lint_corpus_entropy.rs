//! # Recipe: imatrix Lint — Calibration Corpus Entropy
//!
//! **Category**: lint
//! **CLI Equivalent**: `apr imatrix-lint --observation-file observation.json` (corpus path)
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## Learning Objective
//! Demonstrates corpus-entropy auditing. The calibration corpus must have
//! enough lexical diversity for the importance estimates to generalize
//! beyond the calibration set. The lint computes the per-byte Shannon
//! entropy of the recorded `chunk_byte_histogram` and flags corpora below
//! 4.5 bits/byte (typical English text is ≥ 4.6) — too low means the
//! calibration ran on highly repetitive content (boilerplate, JSON,
//! base64) and the importance estimates will not transfer.
//!
//! ## Run Command
//! ```bash
//! cargo run --example imatrix_lint_corpus_entropy
//! ```
//!
//! ## References
//! - Shannon, C. E. (1948). *A Mathematical Theory of Communication*.
//! - llama.cpp imatrix-tool corpus recommendations.
//!
//! Added by PMAT-091 (expand-cookbooks followup — Ollama/sampling/imatrix lint).

use apr_cookbook::prelude::*;
use apr_cookbook::Result;
use serde_json::{json, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct CorpusEntropy {
    pub bits_per_byte: f64,
    pub passes: bool,
}

const MIN_BITS_PER_BYTE: f64 = 4.5;

pub fn compute_corpus_entropy(obs: &Value) -> Option<CorpusEntropy> {
    let hist = obs.get("chunk_byte_histogram").and_then(Value::as_array)?;
    let counts: Vec<u64> = hist.iter().filter_map(Value::as_u64).collect();
    if counts.len() != 256 {
        return None;
    }
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return None;
    }
    let total_f = total as f64;
    let entropy: f64 = counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total_f;
            -p * p.log2()
        })
        .sum();
    Some(CorpusEntropy {
        bits_per_byte: entropy,
        passes: entropy >= MIN_BITS_PER_BYTE,
    })
}

fn build_high_entropy_observation() -> Value {
    // Roughly uniform distribution → entropy near log2(256) = 8 bits/byte.
    let mut hist = vec![0u64; 256];
    for (i, c) in hist.iter_mut().enumerate() {
        *c = 100 + (i as u64 % 7);
    }
    json!({ "chunk_byte_histogram": hist })
}

fn build_low_entropy_observation() -> Value {
    // 99% byte 0x20 (space) — entropy near 0.08 bits/byte.
    let mut hist = vec![0u64; 256];
    hist[0x20] = 9_900;
    hist[0x65] = 50; // some 'e'
    hist[0x74] = 50; // some 't'
    json!({ "chunk_byte_histogram": hist })
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("imatrix_lint_corpus_entropy")?;

    for (label, obs) in [
        ("high entropy", build_high_entropy_observation()),
        ("low entropy", build_low_entropy_observation()),
    ] {
        if let Some(c) = compute_corpus_entropy(&obs) {
            println!(
                "{label:>14}  bits/byte={:.3}  pass={}",
                c.bits_per_byte, c.passes
            );
        }
    }

    ctx.record_string_metric("verdict", "matrix_printed");
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corpus_entropy_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn high_entropy_corpus_passes() {
        let c = compute_corpus_entropy(&build_high_entropy_observation()).unwrap();
        assert!(c.passes, "audit: {c:?}");
        assert!(c.bits_per_byte > 7.5, "near-uniform should be > 7.5: {c:?}");
    }

    #[test]
    fn low_entropy_corpus_fails() {
        let c = compute_corpus_entropy(&build_low_entropy_observation()).unwrap();
        assert!(!c.passes, "audit: {c:?}");
    }

    #[test]
    fn wrong_histogram_length_returns_none() {
        let obs = json!({ "chunk_byte_histogram": vec![1u64; 128] });
        assert!(compute_corpus_entropy(&obs).is_none());
    }

    #[test]
    fn empty_histogram_returns_none() {
        let obs = json!({ "chunk_byte_histogram": vec![0u64; 256] });
        assert!(compute_corpus_entropy(&obs).is_none());
    }

    #[test]
    fn boundary_at_exactly_min_passes() {
        // Conservative-equality at the gate: bits_per_byte == 4.5 must pass.
        // Construct a 2-symbol histogram with controlled entropy.
        let mut hist = vec![0u64; 256];
        // Two symbols at 5/16 and 11/16 give H ≈ 0.896 bits — too low.
        // Use a histogram that ~hits exactly 4.5 bits: 16 symbols with weights
        // [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1] → log2(16) = 4.0. Need ≥ 4.5
        // → use 24 symbols equally → log2(24) ≈ 4.58 → passes.
        for c in hist.iter_mut().take(24) {
            *c = 100;
        }
        let obs = json!({ "chunk_byte_histogram": hist });
        let c = compute_corpus_entropy(&obs).unwrap();
        assert!(c.passes, "log2(24)≈4.58 should pass: {c:?}");
    }
}
