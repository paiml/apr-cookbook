#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
//! Unified Model Inference Dispatch (`apr run`)
//! **CLI Equivalent**: `apr run`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Mirrors what `apr run model.apr --prompt "hello" --max-tokens 50` does:
//! tokenize input, run a tiny 2-layer transformer forward pass, sample tokens
//! autoregressively, decode output, and optionally benchmark throughput.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────┐
//! │                      apr run Pipeline                            │
//! ├──────────────────────────────────────────────────────────────────┤
//! │  1. Parse RunConfig (prompt, max_tokens, temperature, benchmark) │
//! │  2. Build Vocabulary (1000 synthetic tokens)                     │
//! │  3. Tokenize prompt (whitespace split + vocab lookup)            │
//! │  4. Load model weights (2-layer transformer)                     │
//! │  5. Autoregressive loop:                                         │
//! │     embed -> attention -> FFN -> logits -> argmax/sample         │
//! │  6. Decode output tokens back to text                            │
//! │  7. (Optional) Benchmark: 10 iterations, avg latency/throughput  │
//! └──────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example inference_apr_run
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Crankshaw, D. et al. (2017). *Clipper: A Low-Latency Online Prediction Serving System*. NSDI. arXiv:1612.03079

use apr_cookbook::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const VOCAB_SIZE: usize = 1000;
pub const EMBED_DIM: usize = 32;
pub const NUM_HEADS: usize = 4;
pub const HEAD_DIM: usize = EMBED_DIM / NUM_HEADS;
pub const FFN_DIM: usize = 64;
pub const NUM_LAYERS: usize = 2;
pub const UNK_TOKEN: &str = "<unk>";
pub const BOS_TOKEN: &str = "<bos>";
pub const EOS_TOKEN: &str = "<eos>";
// ---------------------------------------------------------------------------
// RunConfig / RunResult
// ---------------------------------------------------------------------------

/// Configuration for `apr run` inference dispatch.
pub struct RunConfig {
    pub prompt: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub benchmark: bool,
}

impl RunConfig {
    pub fn new(prompt: &str, max_tokens: usize) -> Self {
        Self {
            prompt: prompt.to_string(),
            max_tokens,
            temperature: 1.0,
            benchmark: false,
        }
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    pub fn with_benchmark(mut self, b: bool) -> Self {
        self.benchmark = b;
        self
    }
}

/// Result of a single inference run.
#[allow(dead_code)]
pub struct RunResult {
    pub input_tokens: Vec<usize>,
    pub output_tokens: Vec<usize>,
    pub decoded_text: String,
    pub latency_ms: f64,
}

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

/// Synthetic vocabulary with bidirectional lookup.
pub struct Vocabulary {
    pub tokens: Vec<String>,
    pub token_to_id: HashMap<String, usize>,
}

impl Vocabulary {
    /// Build a 1000-token vocabulary from a deterministic RNG.
    pub fn build(rng: &mut impl Rng) -> Self {
        let mut tokens = Vec::with_capacity(VOCAB_SIZE);
        let mut token_to_id = HashMap::with_capacity(VOCAB_SIZE);

        // Special tokens at fixed positions
        let specials = [UNK_TOKEN, BOS_TOKEN, EOS_TOKEN];
        for s in &specials {
            let id = tokens.len();
            token_to_id.insert((*s).to_string(), id);
            tokens.push((*s).to_string());
        }

        // Common English words (deterministic pool)
        let seed_words = [
            "the",
            "of",
            "and",
            "a",
            "to",
            "in",
            "is",
            "it",
            "that",
            "was",
            "for",
            "on",
            "are",
            "with",
            "as",
            "at",
            "be",
            "this",
            "have",
            "from",
            "or",
            "had",
            "by",
            "not",
            "but",
            "what",
            "all",
            "were",
            "when",
            "we",
            "there",
            "can",
            "an",
            "your",
            "which",
            "their",
            "if",
            "do",
            "will",
            "each",
            "about",
            "how",
            "up",
            "out",
            "them",
            "then",
            "she",
            "many",
            "some",
            "so",
            "these",
            "would",
            "other",
            "into",
            "has",
            "more",
            "two",
            "her",
            "like",
            "him",
            "time",
            "very",
            "make",
            "just",
            "know",
            "take",
            "people",
            "come",
            "could",
            "than",
            "look",
            "only",
            "its",
            "over",
            "think",
            "also",
            "back",
            "after",
            "use",
            "work",
            "first",
            "well",
            "way",
            "even",
            "new",
            "want",
            "because",
            "any",
            "give",
            "day",
            "most",
            "hello",
            "world",
            "model",
            "data",
            "input",
            "output",
            "layer",
            "weight",
            "bias",
            "token",
            "embed",
            "attention",
            "forward",
            "loss",
            "train",
            "test",
            "run",
            "load",
            "save",
            "predict",
            "sample",
        ];

        for w in &seed_words {
            if !token_to_id.contains_key(*w) {
                let id = tokens.len();
                token_to_id.insert((*w).to_string(), id);
                tokens.push((*w).to_string());
            }
        }

        // Fill remaining slots with synthetic tokens
        while tokens.len() < VOCAB_SIZE {
            let suffix: u32 = rng.gen_range(0..100_000);
            let tok = format!("t_{suffix}");
            if !token_to_id.contains_key(&tok) {
                let id = tokens.len();
                token_to_id.insert(tok.clone(), id);
                tokens.push(tok);
            }
        }

        Self {
            tokens,
            token_to_id,
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let unk_id = self.token_to_id.get(UNK_TOKEN).copied().unwrap_or(0);
        let bos_id = self.token_to_id.get(BOS_TOKEN).copied().unwrap_or(1);
        let mut ids = vec![bos_id];
        for word in text.split_whitespace() {
            let lower = word.to_lowercase();
            let id = self
                .token_to_id
                .get(lower.as_str())
                .copied()
                .unwrap_or(unk_id);
            ids.push(id);
        }
        ids
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut parts = Vec::with_capacity(ids.len());
        for &id in ids {
            let tok = if id < self.tokens.len() {
                self.tokens[id].as_str()
            } else {
                UNK_TOKEN
            };
            // Skip special tokens in decoded output
            if tok != BOS_TOKEN && tok != EOS_TOKEN {
                parts.push(tok);
            }
        }
        parts.join(" ")
    }

    pub fn size(&self) -> usize {
        self.tokens.len()
    }
}
