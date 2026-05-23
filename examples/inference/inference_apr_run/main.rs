#![allow(unused_imports)]
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

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
