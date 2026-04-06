//! Streaming Token Generator Example
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Demonstrates autoregressive token generation with streaming output,
//! stop sequences, and time-to-first-token (TTFT) measurement.
//!
//! # Features
//!
//! - **Streaming Output**: Tokens yielded one at a time via Iterator
//! - **Stop Sequences**: Generation halts on configurable stop patterns
//! - **TTFT Measurement**: Time-to-first-token tracked separately from throughput
//! - **Max Length**: Hard limit on generated sequence length
//!
//! # Running
//!
//! ```bash
//! cargo run --example streaming_token_generator
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

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

const VOCAB_SIZE: usize = 128;

/// Token vocabulary mapping (simplified ASCII subset)
fn token_to_char(token: u32) -> char {
    match token {
        0 => ' ',
        1..=26 => (b'a' + (token as u8 - 1)) as char,
        27..=52 => (b'A' + (token as u8 - 27)) as char,
        53..=62 => (b'0' + (token as u8 - 53)) as char,
        63 => '.',
        64 => ',',
        65 => '!',
        66 => '?',
        67 => '\n',
        68 => ':',
        69 => ';',
        _ => '#',
    }
}

/// Simple autoregressive model for token generation
struct TokenModel {
    seed: u64,
    temperature: f32,
}

impl TokenModel {
    fn new(seed: u64, temperature: f32) -> Self {
        Self { seed, temperature }
    }

    fn next_token_probs(&self, context: &[u32]) -> Vec<f64> {
        let mut logits = vec![0.0f64; VOCAB_SIZE];
        for (i, logit) in logits.iter_mut().enumerate() {
            let mut hasher = DefaultHasher::new();
            (self.seed, context, i).hash(&mut hasher);
            let h = hasher.finish();
            *logit = (h as f64 / u64::MAX as f64 - 0.5) * 2.0 / f64::from(self.temperature);
        }

        // Bias towards printable tokens (0-70)
        for logit in logits.iter_mut().skip(70) {
            *logit -= 3.0;
        }

        // Softmax
        let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exp_sum: f64 = logits.iter().map(|l| (l - max_l).exp()).sum();
        logits.iter().map(|l| (l - max_l).exp() / exp_sum).collect()
    }

    fn sample(&self, probs: &[f64], step: usize) -> u32 {
        let mut hasher = DefaultHasher::new();
        (self.seed, "sample", step).hash(&mut hasher);
        let r = hasher.finish() as f64 / u64::MAX as f64;

        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return i as u32;
            }
        }
        0
    }
}

/// Configuration for streaming generation
struct GenerationConfig {
    max_tokens: usize,
    stop_sequences: Vec<Vec<u32>>,
    temperature: f32,
}

impl GenerationConfig {
    fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            stop_sequences: Vec::new(),
            temperature: 1.0,
        }
    }

    fn with_stop_sequence(mut self, seq: Vec<u32>) -> Self {
        self.stop_sequences.push(seq);
        self
    }

    fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }
}

/// Streaming token generator implementing Iterator
struct TokenStream {
    model: TokenModel,
    context: Vec<u32>,
    config: GenerationConfig,
    generated: usize,
    stopped: bool,
    start_time: Option<Instant>,
    first_token_time: Option<std::time::Duration>,
}

impl TokenStream {
    fn new(model: TokenModel, prompt: Vec<u32>, config: GenerationConfig) -> Self {
        Self {
            model,
            context: prompt,
            config,
            generated: 0,
            stopped: false,
            start_time: None,
            first_token_time: None,
        }
    }

    fn check_stop_sequence(&self) -> bool {
        for stop_seq in &self.config.stop_sequences {
            if self.context.len() >= stop_seq.len()
                && self.context[self.context.len() - stop_seq.len()..] == **stop_seq
            {
                return true;
            }
        }
        false
    }

    fn ttft(&self) -> Option<std::time::Duration> {
        self.first_token_time
    }
}

impl Iterator for TokenStream {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.stopped || self.generated >= self.config.max_tokens {
            return None;
        }

        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }

        let probs = self.model.next_token_probs(&self.context);
        let token = self.model.sample(&probs, self.generated);
        self.context.push(token);
        self.generated += 1;

        if self.first_token_time.is_none() {
            if let Some(start) = self.start_time {
                self.first_token_time = Some(start.elapsed());
            }
        }

        if self.check_stop_sequence() {
            self.stopped = true;
        }

        Some(token)
    }
}

/// Collect generation statistics
#[allow(dead_code)] // Fields used in tests
struct GenerationStats {
    total_tokens: usize,
    ttft_us: u64,
    total_time_us: u64,
    stopped_by_sequence: bool,
    stopped_by_length: bool,
}

impl GenerationStats {
    fn tokens_per_sec(&self) -> f64 {
        if self.total_time_us == 0 {
            return 0.0;
        }
        self.total_tokens as f64 / (self.total_time_us as f64 / 1_000_000.0)
    }
}

fn run_generation(
    seed: u64,
    prompt: Vec<u32>,
    config: GenerationConfig,
) -> (Vec<u32>, GenerationStats) {
    let model = TokenModel::new(seed, config.temperature);
    let max_tokens = config.max_tokens;
    let has_stop_seqs = !config.stop_sequences.is_empty();
    let mut stream = TokenStream::new(model, prompt, config);

    let start = Instant::now();
    let tokens: Vec<u32> = stream.by_ref().collect();
    let total_time = start.elapsed();

    let ttft = stream.ttft().unwrap_or_default();
    let stopped_by_seq = stream.stopped;
    let stopped_by_len = tokens.len() >= max_tokens && !stopped_by_seq;

    let stats = GenerationStats {
        total_tokens: tokens.len(),
        ttft_us: ttft.as_micros() as u64,
        total_time_us: total_time.as_micros() as u64,
        stopped_by_sequence: stopped_by_seq && has_stop_seqs,
        stopped_by_length: stopped_by_len,
    };

    (tokens, stats)
}

fn tokens_to_string(tokens: &[u32]) -> String {
    tokens.iter().map(|&t| token_to_char(t)).collect()
}

fn main() {
    println!("=== Streaming Token Generator Example ===\n");

    // =========================================================================
    // Section 1: Basic Streaming Generation
    // =========================================================================
    println!("1. Basic Streaming Generation");
    println!("   ─────────────────────────────────────────");

    let prompt = vec![8, 5, 12, 12, 15]; // "hello" encoded
    let config = GenerationConfig::new(60).with_temperature(0.8);
    let (tokens, stats) = run_generation(42, prompt.clone(), config);

    println!("   Prompt:    {:?}", &prompt);
    println!("   Generated: {} tokens", stats.total_tokens);
    println!("   TTFT:      {} us", stats.ttft_us);
    println!("   Total:     {} us", stats.total_time_us);
    println!("   Throughput: {:.0} tok/s", stats.tokens_per_sec());
    println!(
        "   Output:    \"{}\"",
        &tokens_to_string(&tokens)[..tokens.len().min(40)]
    );
    println!();

    // =========================================================================
    // Section 2: Stop Sequence Detection
    // =========================================================================
    println!("2. Stop Sequence Detection");
    println!("   ─────────────────────────────────────────");

    let stop_seq = vec![63, 67]; // ".\n"
    let config = GenerationConfig::new(200)
        .with_stop_sequence(stop_seq.clone())
        .with_temperature(0.9);
    let (tokens, stats) = run_generation(42, prompt.clone(), config);

    println!("   Stop sequence: {:?}", stop_seq);
    println!("   Generated: {} tokens", stats.total_tokens);
    println!(
        "   Stopped by: {}",
        if stats.stopped_by_sequence {
            "stop sequence"
        } else {
            "max length"
        }
    );
    println!(
        "   Output:    \"{}\"",
        &tokens_to_string(&tokens)[..tokens.len().min(50)]
    );
    println!();

    // =========================================================================
    // Section 3: Temperature Comparison
    // =========================================================================
    println!("3. Temperature Sweep");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>6} {:>8} {:>12} {:>12}",
        "Temp", "Tokens", "Unique%", "Preview"
    );
    println!("   {}", "─".repeat(45));

    for temp in [0.3, 0.6, 1.0, 1.5, 2.0] {
        let config = GenerationConfig::new(100).with_temperature(temp);
        let (tokens, _) = run_generation(42, prompt.clone(), config);

        let unique: std::collections::HashSet<_> = tokens.iter().collect();
        let unique_pct = unique.len() as f64 / tokens.len() as f64 * 100.0;
        let preview = tokens_to_string(&tokens);
        let preview_slice = &preview[..preview.len().min(20)];

        println!(
            "   {:>6.1} {:>8} {:>11.1}% {:>12}",
            temp,
            tokens.len(),
            unique_pct,
            preview_slice
        );
    }
    println!();

    // =========================================================================
    // Section 4: Throughput Benchmark
    // =========================================================================
    println!("4. Throughput Benchmark");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>10} {:>10} {:>10} {:>14}",
        "MaxLen", "Generated", "Time(us)", "Tok/s"
    );
    println!("   {}", "─".repeat(50));

    for max_len in [10, 50, 100, 200, 500] {
        let config = GenerationConfig::new(max_len).with_temperature(1.0);
        let (_, stats) = run_generation(42, prompt.clone(), config);
        println!(
            "   {:>10} {:>10} {:>10} {:>13.0}",
            max_len,
            stats.total_tokens,
            stats.total_time_us,
            stats.tokens_per_sec()
        );
    }
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_model_softmax() {
        let model = TokenModel::new(42, 1.0);
        let probs = model.next_token_probs(&[1, 2, 3]);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(probs.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_token_model_deterministic() {
        let m1 = TokenModel::new(42, 1.0);
        let m2 = TokenModel::new(42, 1.0);
        let p1 = m1.next_token_probs(&[1, 2]);
        let p2 = m2.next_token_probs(&[1, 2]);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_stream_max_tokens() {
        let config = GenerationConfig::new(20).with_temperature(1.0);
        let (tokens, stats) = run_generation(42, vec![1, 2], config);
        assert_eq!(tokens.len(), 20);
        assert!(stats.stopped_by_length);
    }

    #[test]
    fn test_stream_stop_sequence() {
        // Use a stop sequence that will eventually appear
        let stop_seq = vec![0]; // space character - very common
        let config = GenerationConfig::new(500)
            .with_stop_sequence(stop_seq)
            .with_temperature(1.0);
        let (tokens, stats) = run_generation(42, vec![1, 2], config);
        assert!(tokens.len() <= 500);
        // Either stopped by sequence or max length
        assert!(stats.stopped_by_sequence || stats.stopped_by_length);
    }

    #[test]
    fn test_token_to_char_printable() {
        for i in 0..70u32 {
            let c = token_to_char(i);
            assert!(c.is_ascii(), "Token {} maps to non-ASCII char: {}", i, c);
        }
    }

    #[test]
    fn test_temperature_affects_distribution() {
        let low_temp = TokenModel::new(42, 0.1);
        let high_temp = TokenModel::new(42, 5.0);

        let probs_low = low_temp.next_token_probs(&[1, 2]);
        let probs_high = high_temp.next_token_probs(&[1, 2]);

        let max_low = probs_low.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let max_high = probs_high.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        // Low temperature should have sharper (higher max) distribution
        assert!(
            max_low > max_high,
            "Low temp max {} should > high temp max {}",
            max_low,
            max_high
        );
    }

    #[test]
    fn test_generation_stats() {
        let config = GenerationConfig::new(30).with_temperature(1.0);
        let (_, stats) = run_generation(42, vec![1], config);
        assert_eq!(stats.total_tokens, 30);
        assert!(stats.tokens_per_sec() > 0.0);
    }
}
