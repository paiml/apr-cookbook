//! Unified Model Inference Dispatch (`apr run`)
//! **CLI Equivalent**: `apr run`
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

const VOCAB_SIZE: usize = 1000;
const EMBED_DIM: usize = 32;
const NUM_HEADS: usize = 4;
const HEAD_DIM: usize = EMBED_DIM / NUM_HEADS;
const FFN_DIM: usize = 64;
const NUM_LAYERS: usize = 2;
const UNK_TOKEN: &str = "<unk>";
const BOS_TOKEN: &str = "<bos>";
const EOS_TOKEN: &str = "<eos>";

// ---------------------------------------------------------------------------
// RunConfig / RunResult
// ---------------------------------------------------------------------------

/// Configuration for `apr run` inference dispatch.
struct RunConfig {
    prompt: String,
    max_tokens: usize,
    temperature: f32,
    benchmark: bool,
}

impl RunConfig {
    fn new(prompt: &str, max_tokens: usize) -> Self {
        Self {
            prompt: prompt.to_string(),
            max_tokens,
            temperature: 1.0,
            benchmark: false,
        }
    }

    fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    fn with_benchmark(mut self, b: bool) -> Self {
        self.benchmark = b;
        self
    }
}

/// Result of a single inference run.
#[allow(dead_code)]
struct RunResult {
    input_tokens: Vec<usize>,
    output_tokens: Vec<usize>,
    decoded_text: String,
    latency_ms: f64,
}

// ---------------------------------------------------------------------------
// Vocabulary
// ---------------------------------------------------------------------------

/// Synthetic vocabulary with bidirectional lookup.
struct Vocabulary {
    tokens: Vec<String>,
    token_to_id: HashMap<String, usize>,
}

impl Vocabulary {
    /// Build a 1000-token vocabulary from a deterministic RNG.
    fn build(rng: &mut impl Rng) -> Self {
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

    fn encode(&self, text: &str) -> Vec<usize> {
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

    fn decode(&self, ids: &[usize]) -> String {
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

    fn size(&self) -> usize {
        self.tokens.len()
    }
}

// ---------------------------------------------------------------------------
// Transformer Weights
// ---------------------------------------------------------------------------

/// Weights for a single transformer layer.
struct TransformerLayer {
    wq: Vec<f32>, // EMBED_DIM x EMBED_DIM
    wk: Vec<f32>, // EMBED_DIM x EMBED_DIM
    wv: Vec<f32>, // EMBED_DIM x EMBED_DIM
    wo: Vec<f32>, // EMBED_DIM x EMBED_DIM
    w1: Vec<f32>, // FFN_DIM x EMBED_DIM
    b1: Vec<f32>, // FFN_DIM
    w2: Vec<f32>, // EMBED_DIM x FFN_DIM
    b2: Vec<f32>, // EMBED_DIM
}

/// Full 2-layer transformer model.
struct TinyTransformer {
    embedding: Vec<f32>, // VOCAB_SIZE x EMBED_DIM
    layers: Vec<TransformerLayer>,
    output_proj: Vec<f32>, // VOCAB_SIZE x EMBED_DIM
}

impl TinyTransformer {
    fn new(rng: &mut impl Rng) -> Self {
        let scale = 0.02_f32;

        let embedding = Self::rand_weights(rng, VOCAB_SIZE * EMBED_DIM, scale);

        let mut layers = Vec::with_capacity(NUM_LAYERS);
        for _ in 0..NUM_LAYERS {
            layers.push(TransformerLayer {
                wq: Self::rand_weights(rng, EMBED_DIM * EMBED_DIM, scale),
                wk: Self::rand_weights(rng, EMBED_DIM * EMBED_DIM, scale),
                wv: Self::rand_weights(rng, EMBED_DIM * EMBED_DIM, scale),
                wo: Self::rand_weights(rng, EMBED_DIM * EMBED_DIM, scale),
                w1: Self::rand_weights(rng, FFN_DIM * EMBED_DIM, scale),
                b1: vec![0.0; FFN_DIM],
                w2: Self::rand_weights(rng, EMBED_DIM * FFN_DIM, scale),
                b2: vec![0.0; EMBED_DIM],
            });
        }

        let output_proj = Self::rand_weights(rng, VOCAB_SIZE * EMBED_DIM, scale);

        Self {
            embedding,
            layers,
            output_proj,
        }
    }

    fn rand_weights(rng: &mut impl Rng, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| rng.gen_range(-scale..scale)).collect()
    }

    fn param_count(&self) -> usize {
        let embed_params = self.embedding.len();
        let layer_params: usize = self
            .layers
            .iter()
            .map(|l| {
                l.wq.len()
                    + l.wk.len()
                    + l.wv.len()
                    + l.wo.len()
                    + l.w1.len()
                    + l.b1.len()
                    + l.w2.len()
                    + l.b2.len()
            })
            .sum();
        let output_params = self.output_proj.len();
        embed_params + layer_params + output_params
    }

    /// Look up embedding for a single token.
    fn embed(&self, token_id: usize) -> Vec<f32> {
        let id = token_id.min(VOCAB_SIZE - 1);
        let start = id * EMBED_DIM;
        self.embedding[start..start + EMBED_DIM].to_vec()
    }

    /// Forward pass for the last token position given full context.
    /// Returns logits over the vocabulary.
    fn forward(&self, token_ids: &[usize]) -> Vec<f32> {
        // Embed the last token (simplified: causal LM uses last position)
        let mut hidden = self.embed(*token_ids.last().unwrap_or(&0));

        // Add simple positional signal
        let pos = token_ids.len().saturating_sub(1);
        for (i, h) in hidden.iter_mut().enumerate() {
            let angle = pos as f32 / 10000_f32.powf(2.0 * (i / 2) as f32 / EMBED_DIM as f32);
            if i % 2 == 0 {
                *h += angle.sin() * 0.1;
            } else {
                *h += angle.cos() * 0.1;
            }
        }

        // Pass through each transformer layer
        for layer in &self.layers {
            hidden = self.transformer_layer(&hidden, layer, token_ids);
        }

        // Project to vocabulary logits
        matmul(&hidden, &self.output_proj, VOCAB_SIZE, EMBED_DIM)
    }

    fn transformer_layer(
        &self,
        hidden: &[f32],
        layer: &TransformerLayer,
        context: &[usize],
    ) -> Vec<f32> {
        // --- Multi-head self-attention (simplified single-position) ---
        let q = matmul(hidden, &layer.wq, EMBED_DIM, EMBED_DIM);
        let k = matmul(hidden, &layer.wk, EMBED_DIM, EMBED_DIM);
        let v = matmul(hidden, &layer.wv, EMBED_DIM, EMBED_DIM);

        let mut attn_out = vec![0.0_f32; EMBED_DIM];
        for head in 0..NUM_HEADS {
            let offset = head * HEAD_DIM;
            let q_head = &q[offset..offset + HEAD_DIM];
            let k_head = &k[offset..offset + HEAD_DIM];
            let v_head = &v[offset..offset + HEAD_DIM];

            // Scaled dot-product attention score
            let score: f32 = q_head
                .iter()
                .zip(k_head.iter())
                .map(|(&qi, &ki)| qi * ki)
                .sum::<f32>()
                / (HEAD_DIM as f32).sqrt();

            // Context-length modulated attention (simplification)
            let weight = sigmoid(score + 0.1 * context.len() as f32);

            for (i, &vi) in v_head.iter().enumerate() {
                attn_out[offset + i] += weight * vi;
            }
        }

        // Output projection
        let projected = matmul(&attn_out, &layer.wo, EMBED_DIM, EMBED_DIM);

        // Residual connection
        let mut residual: Vec<f32> = hidden
            .iter()
            .zip(projected.iter())
            .map(|(&h, &p)| h + p)
            .collect();

        // --- Feed-forward network ---
        let ffn_hidden = matmul_bias(&residual, &layer.w1, &layer.b1, FFN_DIM, EMBED_DIM);
        let ffn_activated: Vec<f32> = ffn_hidden.iter().map(|&x| gelu(x)).collect();
        let ffn_out = matmul_bias(&ffn_activated, &layer.w2, &layer.b2, EMBED_DIM, FFN_DIM);

        // Residual connection
        for (r, &f) in residual.iter_mut().zip(ffn_out.iter()) {
            *r += f;
        }

        residual
    }
}

// ---------------------------------------------------------------------------
// Math utilities
// ---------------------------------------------------------------------------

/// Matrix-vector multiply: W (out_dim x in_dim) @ x (in_dim) -> (out_dim)
fn matmul(input: &[f32], weights: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; out_dim];
    for (o, out) in output.iter_mut().enumerate() {
        for (i, &inp) in input.iter().enumerate().take(in_dim) {
            *out += weights[o * in_dim + i] * inp;
        }
    }
    output
}

/// Matrix-vector multiply with bias.
fn matmul_bias(
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let mut output = bias.to_vec();
    for (o, out) in output.iter_mut().enumerate().take(out_dim) {
        for (i, &inp) in input.iter().enumerate().take(in_dim) {
            *out += weights[o * in_dim + i] * inp;
        }
    }
    output
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Argmax sampling (greedy) with temperature scaling.
fn sample_token(logits: &[f32], temperature: f32, rng: &mut impl Rng) -> usize {
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature.max(0.01)).collect();
    let probs = softmax(&scaled);

    // Sample from the distribution
    let r: f32 = rng.gen();
    let mut cumulative = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i;
        }
    }
    probs.len().saturating_sub(1)
}

// ---------------------------------------------------------------------------
// Inference engine
// ---------------------------------------------------------------------------

/// Run autoregressive inference and return the result.
fn run_inference(
    model: &TinyTransformer,
    vocab: &Vocabulary,
    config: &RunConfig,
    rng: &mut impl Rng,
) -> RunResult {
    let input_tokens = vocab.encode(&config.prompt);
    let mut context = input_tokens.clone();
    let mut output_tokens = Vec::with_capacity(config.max_tokens);

    let start = Instant::now();

    let eos_id = vocab.token_to_id.get(EOS_TOKEN).copied().unwrap_or(2);
    for _ in 0..config.max_tokens {
        let logits = model.forward(&context);
        let next_id = sample_token(&logits, config.temperature, rng);

        if next_id == eos_id {
            break;
        }

        output_tokens.push(next_id);
        context.push(next_id);
    }

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    let decoded_text = vocab.decode(&output_tokens);

    RunResult {
        input_tokens,
        output_tokens,
        decoded_text,
        latency_ms,
    }
}

/// Run benchmark: N iterations, report average latency and throughput.
fn run_benchmark(
    model: &TinyTransformer,
    vocab: &Vocabulary,
    config: &RunConfig,
    rng: &mut impl Rng,
    iterations: usize,
) -> (f64, f64, usize) {
    let mut total_ms = 0.0;
    let mut total_tokens = 0_usize;

    for _ in 0..iterations {
        let result = run_inference(model, vocab, config, rng);
        total_ms += result.latency_ms;
        total_tokens += result.output_tokens.len();
    }

    let avg_latency_ms = total_ms / iterations as f64;
    let throughput = if total_ms > 0.0 {
        total_tokens as f64 / (total_ms / 1000.0)
    } else {
        0.0
    };
    let avg_tokens = total_tokens / iterations;

    (avg_latency_ms, throughput, avg_tokens)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("inference_apr_run")?;
    println!("=== APR Run: Unified Model Inference Dispatch ===\n");

    // =========================================================================
    // Section 1: Build Vocabulary
    // =========================================================================
    println!("1. Vocabulary Construction");
    println!("   ─────────────────────────────────────────");

    let vocab = Vocabulary::build(ctx.rng());

    println!("   Vocab size:     {}", vocab.size());
    println!("   Special tokens: {UNK_TOKEN}, {BOS_TOKEN}, {EOS_TOKEN}");
    println!(
        "   Sample tokens:  [{}, {}, {}, {}, {}]",
        vocab.tokens[3], vocab.tokens[4], vocab.tokens[5], vocab.tokens[6], vocab.tokens[7],
    );
    println!();

    // =========================================================================
    // Section 2: Tokenize Input
    // =========================================================================
    println!("2. Input Tokenization");
    println!("   ─────────────────────────────────────────");

    let prompt = "hello world model data input";
    let token_ids = vocab.encode(prompt);

    println!("   Prompt:  \"{}\"", prompt);
    println!("   Tokens:  {:?}", token_ids);
    print!("   Decoded: [");
    for (i, &id) in token_ids.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        let name = if id < vocab.tokens.len() {
            &vocab.tokens[id]
        } else {
            UNK_TOKEN
        };
        print!("{}={}", id, name);
    }
    println!("]");
    println!();

    // =========================================================================
    // Section 3: Load Model Weights
    // =========================================================================
    println!("3. Model Weights (2-layer Transformer)");
    println!("   ─────────────────────────────────────────");

    let model = TinyTransformer::new(ctx.rng());

    println!("   Layers:       {NUM_LAYERS}");
    println!("   Embed dim:    {EMBED_DIM}");
    println!("   Heads:        {NUM_HEADS} (head_dim={HEAD_DIM})");
    println!("   FFN dim:      {FFN_DIM}");
    println!("   Vocab size:   {VOCAB_SIZE}");
    println!("   Parameters:   {}", model.param_count());
    println!(
        "   Memory (f32): {:.1} KB",
        model.param_count() as f64 * 4.0 / 1024.0
    );
    println!();

    // =========================================================================
    // Section 4: Run Inference (simulates `apr run --prompt "hello" --max-tokens 50`)
    // =========================================================================
    println!(
        "4. Inference: apr run --prompt \"{}\" --max-tokens 50",
        prompt
    );
    println!("   ─────────────────────────────────────────");

    let config = RunConfig::new(prompt, 50).with_temperature(1.0);
    let result = run_inference(&model, &vocab, &config, ctx.rng());

    println!("   Input tokens:  {} ids", result.input_tokens.len());
    println!("   Output tokens: {} ids", result.output_tokens.len());
    println!("   Latency:       {:.2} ms", result.latency_ms);
    println!(
        "   Output IDs:    {:?}{}",
        &result.output_tokens[..result.output_tokens.len().min(15)],
        if result.output_tokens.len() > 15 {
            "..."
        } else {
            ""
        }
    );
    println!(
        "   Decoded:       \"{}\"",
        truncate_str(&result.decoded_text, 80)
    );
    println!();

    // =========================================================================
    // Section 5: Per-Step Logits Detail
    // =========================================================================
    println!("5. Per-Step Logits (first 5 steps)");
    println!("   ─────────────────────────────────────────");

    let detail_config = RunConfig::new(prompt, 5).with_temperature(1.0);
    let detail_ids = vocab.encode(&detail_config.prompt);
    let mut detail_ctx = detail_ids.clone();

    for step in 0..5 {
        let logits = model.forward(&detail_ctx);
        let probs = softmax(&logits);
        let top5 = top_k_indices(&probs, 5);
        let sampled = sample_token(&logits, detail_config.temperature, ctx.rng());
        detail_ctx.push(sampled);

        let sampled_name = vocab.tokens.get(sampled).map_or(UNK_TOKEN, String::as_str);
        println!(
            "   Step {}: sampled {} (\"{}\")",
            step, sampled, sampled_name
        );
        print!("     Top-5: ");
        for (rank, &idx) in top5.iter().enumerate() {
            let name = vocab.tokens.get(idx).map_or(UNK_TOKEN, String::as_str);
            if rank > 0 {
                print!(", ");
            }
            print!("{}={:.4}", name, probs[idx]);
        }
        println!();
    }
    println!();

    // =========================================================================
    // Section 6: Temperature Sweep
    // =========================================================================
    println!("6. Temperature Sweep");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>6} {:>8} {:>10} {:>30}",
        "Temp", "Tokens", "Unique%", "Preview"
    );
    println!("   {}", "-".repeat(58));

    for temp in [0.3, 0.7, 1.0, 1.5, 2.0] {
        let sweep_config = RunConfig::new(prompt, 30).with_temperature(temp);
        let sweep_result = run_inference(&model, &vocab, &sweep_config, ctx.rng());

        let unique: std::collections::HashSet<_> = sweep_result.output_tokens.iter().collect();
        let unique_pct = if sweep_result.output_tokens.is_empty() {
            0.0
        } else {
            unique.len() as f64 / sweep_result.output_tokens.len() as f64 * 100.0
        };

        println!(
            "   {:>6.1} {:>8} {:>9.1}% {:>30}",
            temp,
            sweep_result.output_tokens.len(),
            unique_pct,
            truncate_str(&sweep_result.decoded_text, 28),
        );
    }
    println!();

    // =========================================================================
    // Section 7: Benchmark Mode (10 iterations)
    // =========================================================================
    println!("7. Benchmark: 10 iterations");
    println!("   ─────────────────────────────────────────");

    let bench_config = RunConfig::new(prompt, 50)
        .with_temperature(1.0)
        .with_benchmark(true);

    let (avg_latency, throughput, avg_tokens) =
        run_benchmark(&model, &vocab, &bench_config, ctx.rng(), 10);

    println!("   Iterations:    10");
    println!("   Avg tokens:    {}", avg_tokens);
    println!("   Avg latency:   {:.2} ms", avg_latency);
    println!("   Throughput:    {:.0} tokens/sec", throughput);

    ctx.record_float_metric("avg_latency_ms", avg_latency);
    ctx.record_float_metric("throughput_tok_s", throughput);
    ctx.record_metric("param_count", model.param_count() as i64);
    ctx.record_metric("vocab_size", vocab.size() as i64);
    println!();

    println!("=== Example Complete ===");
    ctx.report()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

fn top_k_indices(probs: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.iter().take(k).map(|&(i, _)| i).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_rng() -> rand::rngs::StdRng {
        use rand::SeedableRng;
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_vocabulary_size() {
        let mut rng = test_rng();
        let vocab = Vocabulary::build(&mut rng);
        assert_eq!(vocab.size(), VOCAB_SIZE);
        assert_eq!(vocab.tokens.len(), VOCAB_SIZE);
        assert_eq!(vocab.token_to_id.len(), VOCAB_SIZE);
    }

    #[test]
    fn test_vocabulary_special_tokens() {
        let mut rng = test_rng();
        let vocab = Vocabulary::build(&mut rng);
        assert_eq!(vocab.token_to_id.get(UNK_TOKEN).copied(), Some(0));
        assert_eq!(vocab.token_to_id.get(BOS_TOKEN).copied(), Some(1));
        assert_eq!(vocab.token_to_id.get(EOS_TOKEN).copied(), Some(2));
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let mut rng = test_rng();
        let vocab = Vocabulary::build(&mut rng);
        let text = "the model is a test";
        let ids = vocab.encode(text);

        // Should start with BOS
        assert_eq!(ids[0], vocab.token_to_id[BOS_TOKEN]);

        // Decode should reconstruct words (minus BOS)
        let decoded = vocab.decode(&ids[1..]);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_unknown_token_handling() {
        let mut rng = test_rng();
        let vocab = Vocabulary::build(&mut rng);
        let ids = vocab.encode("xyzzy_nonexistent_word");
        // BOS + one UNK
        assert_eq!(ids.len(), 2);
        assert_eq!(ids[1], 0); // UNK id
    }

    #[test]
    fn test_model_param_count() {
        let mut rng = test_rng();
        let model = TinyTransformer::new(&mut rng);

        let expected_embed = VOCAB_SIZE * EMBED_DIM;
        let expected_layer = 4 * EMBED_DIM * EMBED_DIM  // wq, wk, wv, wo
            + FFN_DIM * EMBED_DIM + FFN_DIM             // w1, b1
            + EMBED_DIM * FFN_DIM + EMBED_DIM; // w2, b2
        let expected_output = VOCAB_SIZE * EMBED_DIM;
        let expected = expected_embed + NUM_LAYERS * expected_layer + expected_output;

        assert_eq!(model.param_count(), expected);
    }

    #[test]
    fn test_forward_output_shape() {
        let mut rng = test_rng();
        let model = TinyTransformer::new(&mut rng);
        let logits = model.forward(&[1, 5, 10]);
        assert_eq!(logits.len(), VOCAB_SIZE);
    }

    #[test]
    fn test_softmax_properties() {
        let probs = softmax(&[1.0, 2.0, 3.0, 0.5]);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax sum: {sum}");
        assert!(probs.iter().all(|&p| p >= 0.0));
        // Highest logit should have highest probability
        assert!(probs[2] > probs[0]);
    }

    #[test]
    fn test_inference_produces_tokens() {
        let mut rng = test_rng();
        let vocab = Vocabulary::build(&mut rng);
        let model = TinyTransformer::new(&mut rng);
        let config = RunConfig::new("hello world", 20).with_temperature(1.0);
        let result = run_inference(&model, &vocab, &config, &mut rng);

        assert!(!result.output_tokens.is_empty(), "Should generate tokens");
        assert!(result.output_tokens.len() <= 20);
        assert!(result.latency_ms >= 0.0);
        assert!(!result.decoded_text.is_empty());
    }

    #[test]
    fn test_deterministic_with_same_seed() {
        let mut rng1 = test_rng();
        let vocab1 = Vocabulary::build(&mut rng1);
        let model1 = TinyTransformer::new(&mut rng1);
        let config = RunConfig::new("hello", 10).with_temperature(1.0);
        let result1 = run_inference(&model1, &vocab1, &config, &mut rng1);

        let mut rng2 = test_rng();
        let vocab2 = Vocabulary::build(&mut rng2);
        let model2 = TinyTransformer::new(&mut rng2);
        let result2 = run_inference(&model2, &vocab2, &config, &mut rng2);

        assert_eq!(result1.output_tokens, result2.output_tokens);
        assert_eq!(result1.decoded_text, result2.decoded_text);
    }

    #[test]
    fn test_benchmark_returns_valid_metrics() {
        let mut rng = test_rng();
        let vocab = Vocabulary::build(&mut rng);
        let model = TinyTransformer::new(&mut rng);
        let config = RunConfig::new("test input", 10).with_temperature(1.0);

        let (avg_latency, throughput, avg_tokens) =
            run_benchmark(&model, &vocab, &config, &mut rng, 3);

        assert!(avg_latency > 0.0, "Latency should be positive");
        assert!(throughput > 0.0, "Throughput should be positive");
        assert!(avg_tokens > 0, "Should produce tokens");
    }
}
