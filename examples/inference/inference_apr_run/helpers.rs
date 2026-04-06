#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports, clippy::wildcard_imports)]
use super::types::*;

use apr_cookbook::prelude::*;
use rand::Rng;
use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Transformer Weights
// ---------------------------------------------------------------------------

/// Weights for a single transformer layer.
pub struct TransformerLayer {
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

/// Full 2-layer transformer model.
pub struct TinyTransformer {
    pub embedding: Vec<f32>,
    pub layers: Vec<TransformerLayer>,
    pub output_proj: Vec<f32>,
}

impl TinyTransformer {
    pub fn new(rng: &mut impl Rng) -> Self {
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

    pub fn rand_weights(rng: &mut impl Rng, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| rng.gen_range(-scale..scale)).collect()
    }

    pub fn param_count(&self) -> usize {
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
    pub fn embed(&self, token_id: usize) -> Vec<f32> {
        let id = token_id.min(VOCAB_SIZE - 1);
        let start = id * EMBED_DIM;
        self.embedding[start..start + EMBED_DIM].to_vec()
    }

    /// Forward pass for the last token position given full context.
    /// Returns logits over the vocabulary.
    pub fn forward(&self, token_ids: &[usize]) -> Vec<f32> {
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

    pub fn transformer_layer(
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
pub fn matmul(input: &[f32], weights: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0_f32; out_dim];
    for (o, out) in output.iter_mut().enumerate() {
        for (i, &inp) in input.iter().enumerate().take(in_dim) {
            *out += weights[o * in_dim + i] * inp;
        }
    }
    output
}

/// Matrix-vector multiply with bias.
pub fn matmul_bias(
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

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Argmax sampling (greedy) with temperature scaling.
pub fn sample_token(logits: &[f32], temperature: f32, rng: &mut impl Rng) -> usize {
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
pub fn run_inference(
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
pub fn run_benchmark(
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
