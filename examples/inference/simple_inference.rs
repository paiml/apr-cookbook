//! Simple Model Inference Example
//!
//! The "front door" example: load model weights, run a forward pass,
//! interpret the output. Demonstrates the minimal inference loop with
//! no streaming, batching, or caching.
//!
//! # Architecture
//!
//! ```text
//! Input → [Embedding] → [Linear × 2] → [Softmax] → Prediction
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example simple_inference
//! ```

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const VOCAB_SIZE: usize = 64;
const EMBED_DIM: usize = 32;
const HIDDEN_DIM: usize = 16;
const NUM_CLASSES: usize = 5;

/// Class labels for our toy classifier
const CLASS_NAMES: [&str; NUM_CLASSES] = ["positive", "negative", "neutral", "question", "command"];

/// Deterministic weight initialization from seed
fn init_weights(rows: usize, cols: usize, seed: u64, label: &str) -> Vec<f32> {
    (0..rows * cols)
        .map(|i| {
            let mut h = DefaultHasher::new();
            (seed, label, i).hash(&mut h);
            (h.finish() as f32 / u64::MAX as f32 - 0.5) * 0.1
        })
        .collect()
}

/// A simple feedforward classifier
struct SimpleModel {
    embedding: Vec<f32>, // VOCAB_SIZE x EMBED_DIM
    w1: Vec<f32>,        // HIDDEN_DIM x EMBED_DIM
    b1: Vec<f32>,        // HIDDEN_DIM
    w2: Vec<f32>,        // NUM_CLASSES x HIDDEN_DIM
    b2: Vec<f32>,        // NUM_CLASSES
}

impl SimpleModel {
    fn new(seed: u64) -> Self {
        Self {
            embedding: init_weights(VOCAB_SIZE, EMBED_DIM, seed, "embed"),
            w1: init_weights(HIDDEN_DIM, EMBED_DIM, seed, "w1"),
            b1: init_weights(1, HIDDEN_DIM, seed, "b1"),
            w2: init_weights(NUM_CLASSES, HIDDEN_DIM, seed, "w2"),
            b2: init_weights(1, NUM_CLASSES, seed, "b2"),
        }
    }

    /// Count total parameters
    fn param_count(&self) -> usize {
        self.embedding.len() + self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }

    /// Look up embedding for a token
    fn embed(&self, token: usize) -> Vec<f32> {
        let start = token.min(VOCAB_SIZE - 1) * EMBED_DIM;
        self.embedding[start..start + EMBED_DIM].to_vec()
    }

    /// Average pooling over token embeddings
    fn pool(embeddings: &[Vec<f32>]) -> Vec<f32> {
        let n = embeddings.len() as f32;
        let dim = embeddings[0].len();
        let mut pooled = vec![0.0; dim];
        for emb in embeddings {
            for (p, &e) in pooled.iter_mut().zip(emb.iter()) {
                *p += e / n;
            }
        }
        pooled
    }

    /// Linear layer: output = W @ input + bias, with ReLU
    fn linear_relu(input: &[f32], weights: &[f32], bias: &[f32], out_dim: usize) -> Vec<f32> {
        let in_dim = input.len();
        let mut output = bias.to_vec();
        for (o, out) in output.iter_mut().enumerate().take(out_dim) {
            for (i, &inp) in input.iter().enumerate().take(in_dim) {
                *out += weights[o * in_dim + i] * inp;
            }
            *out = out.max(0.0); // ReLU
        }
        output
    }

    /// Linear layer without activation
    fn linear(input: &[f32], weights: &[f32], bias: &[f32], out_dim: usize) -> Vec<f32> {
        let in_dim = input.len();
        let mut output = bias.to_vec();
        for (o, out) in output.iter_mut().enumerate().take(out_dim) {
            for (i, &inp) in input.iter().enumerate().take(in_dim) {
                *out += weights[o * in_dim + i] * inp;
            }
        }
        output
    }

    /// Softmax normalization
    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Full forward pass: tokens → class probabilities
    fn predict(&self, tokens: &[usize]) -> Vec<f32> {
        // Step 1: Embed each token
        let embeddings: Vec<Vec<f32>> = tokens.iter().map(|&t| self.embed(t)).collect();

        // Step 2: Pool embeddings
        let pooled = Self::pool(&embeddings);

        // Step 3: Hidden layer with ReLU
        let hidden = Self::linear_relu(&pooled, &self.w1, &self.b1, HIDDEN_DIM);

        // Step 4: Output layer
        let logits = Self::linear(&hidden, &self.w2, &self.b2, NUM_CLASSES);

        // Step 5: Softmax
        Self::softmax(&logits)
    }
}

/// Simple tokenizer: map ASCII chars to token IDs
fn tokenize(text: &str) -> Vec<usize> {
    text.chars()
        .map(|c| {
            let b = c as usize;
            if b < VOCAB_SIZE {
                b
            } else {
                0
            }
        })
        .collect()
}

/// Format prediction as a readable result
fn format_prediction(probs: &[f32]) -> (usize, &'static str, f32) {
    let (idx, &prob) = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    (idx, CLASS_NAMES[idx], prob)
}

fn main() {
    println!("=== Simple Model Inference Example ===\n");

    // =========================================================================
    // Section 1: Model Loading
    // =========================================================================
    println!("1. Model Loading");
    println!("   ─────────────────────────────────────────");

    let model = SimpleModel::new(42);
    println!("   Architecture: Embedding → Linear → ReLU → Linear → Softmax");
    println!("   Vocab size:   {VOCAB_SIZE}");
    println!("   Embed dim:    {EMBED_DIM}");
    println!("   Hidden dim:   {HIDDEN_DIM}");
    println!("   Classes:      {NUM_CLASSES} ({:?})", CLASS_NAMES);
    println!("   Parameters:   {}", model.param_count());
    println!();

    // =========================================================================
    // Section 2: Single Inference
    // =========================================================================
    println!("2. Single Inference");
    println!("   ─────────────────────────────────────────");

    let text = "hello world";
    let tokens = tokenize(text);
    let probs = model.predict(&tokens);
    let (idx, label, conf) = format_prediction(&probs);

    println!("   Input:      \"{}\"", text);
    println!("   Tokens:     {:?}", &tokens[..tokens.len().min(10)]);
    println!("   Prediction: {} (class {})", label, idx);
    println!("   Confidence: {:.2}%", conf * 100.0);
    println!("   All probs:");
    for (i, (&p, name)) in probs.iter().zip(CLASS_NAMES.iter()).enumerate() {
        let bar = "#".repeat((p * 40.0) as usize);
        println!("     [{i}] {name:>10}: {p:.4} {bar}");
    }
    println!();

    // =========================================================================
    // Section 3: Batch Inference
    // =========================================================================
    println!("3. Batch Inference");
    println!("   ─────────────────────────────────────────");

    let inputs = [
        "I love this product",
        "terrible experience",
        "the weather is okay",
        "what time is it?",
        "stop right now",
    ];

    println!(
        "   {:>25} {:>12} {:>10}",
        "Input", "Prediction", "Confidence"
    );
    println!("   {}", "─".repeat(50));

    for input in &inputs {
        let tokens = tokenize(input);
        let probs = model.predict(&tokens);
        let (_, label, conf) = format_prediction(&probs);
        println!("   {:>25} {:>12} {:>9.2}%", input, label, conf * 100.0);
    }
    println!();

    // =========================================================================
    // Section 4: Inference Timing
    // =========================================================================
    println!("4. Inference Timing");
    println!("   ─────────────────────────────────────────");

    let test_input = tokenize("benchmark this input text for timing");
    let n_iters = 1000;

    let start = std::time::Instant::now();
    for _ in 0..n_iters {
        let _ = model.predict(&test_input);
    }
    let elapsed = start.elapsed();

    let avg_us = elapsed.as_micros() as f64 / f64::from(n_iters);
    let throughput = f64::from(n_iters) / elapsed.as_secs_f64();

    println!("   Iterations:  {n_iters}");
    println!("   Total time:  {} ms", elapsed.as_millis());
    println!("   Avg latency: {avg_us:.1} us/inference");
    println!("   Throughput:  {throughput:.0} inferences/sec");
    println!();

    // =========================================================================
    // Section 5: Model Size Analysis
    // =========================================================================
    println!("5. Model Size Analysis");
    println!("   ─────────────────────────────────────────");

    let layers = [
        ("embedding", VOCAB_SIZE * EMBED_DIM),
        ("w1 (hidden)", HIDDEN_DIM * EMBED_DIM),
        ("b1 (hidden)", HIDDEN_DIM),
        ("w2 (output)", NUM_CLASSES * HIDDEN_DIM),
        ("b2 (output)", NUM_CLASSES),
    ];

    let total: usize = layers.iter().map(|(_, s)| s).sum();
    println!("   {:>15} {:>10} {:>8}", "Layer", "Params", "% Total");
    println!("   {}", "─".repeat(36));
    for (name, size) in &layers {
        let pct = *size as f64 / total as f64 * 100.0;
        println!("   {:>15} {:>10} {:>7.1}%", name, size, pct);
    }
    println!("   {:>15} {:>10}", "TOTAL", total);
    println!(
        "   Memory (f32): {} bytes ({:.1} KB)",
        total * 4,
        total as f64 * 4.0 / 1024.0
    );
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let model = SimpleModel::new(42);
        assert_eq!(model.embedding.len(), VOCAB_SIZE * EMBED_DIM);
        assert_eq!(model.w1.len(), HIDDEN_DIM * EMBED_DIM);
        assert_eq!(model.b1.len(), HIDDEN_DIM);
        assert_eq!(model.w2.len(), NUM_CLASSES * HIDDEN_DIM);
        assert_eq!(model.b2.len(), NUM_CLASSES);
    }

    #[test]
    fn test_predict_probabilities_sum_to_one() {
        let model = SimpleModel::new(42);
        let tokens = tokenize("test input");
        let probs = model.predict(&tokens);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Probs sum to {sum}, expected 1.0");
    }

    #[test]
    fn test_predict_all_nonnegative() {
        let model = SimpleModel::new(42);
        let probs = model.predict(&[1, 2, 3]);
        assert!(probs.iter().all(|&p| p >= 0.0));
    }

    #[test]
    fn test_predict_deterministic() {
        let model = SimpleModel::new(42);
        let tokens = tokenize("deterministic test");
        let p1 = model.predict(&tokens);
        let p2 = model.predict(&tokens);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_tokenize_ascii() {
        // ASCII 'a'=97 > VOCAB_SIZE=64, so maps to 0; space=32 fits
        let tokens = tokenize(" !");
        assert_eq!(tokens, vec![32, 33]);
    }

    #[test]
    fn test_softmax_properties() {
        let probs = SimpleModel::softmax(&[1.0, 2.0, 3.0]);
        assert_eq!(probs.len(), 3);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1] && probs[1] > probs[0]);
    }

    #[test]
    fn test_pool_averages() {
        let embeddings = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let pooled = SimpleModel::pool(&embeddings);
        assert!((pooled[0] - 2.0).abs() < 1e-6);
        assert!((pooled[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_param_count() {
        let model = SimpleModel::new(42);
        let expected = VOCAB_SIZE * EMBED_DIM
            + HIDDEN_DIM * EMBED_DIM
            + HIDDEN_DIM
            + NUM_CLASSES * HIDDEN_DIM
            + NUM_CLASSES;
        assert_eq!(model.param_count(), expected);
    }
}
