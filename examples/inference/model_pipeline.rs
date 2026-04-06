//! Model Pipeline Inference Example
//!
//! Demonstrates chaining multiple models in a pipeline: an encoder
//! produces embeddings, which feed into a classifier. Shows pipeline
//! construction, intermediate inspection, and throughput measurement.
//!
//! # Pipeline Architecture
//!
//! ```text
//! Input → [Tokenizer] → [Encoder] → [Pooling] → [Classifier] → Prediction
//!          (text→ids)    (ids→emb)   (emb→vec)   (vec→class)
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example model_pipeline
//! ```
//!
//! ## References
//! - Crankshaw, D. et al. (2017). *Clipper: A Low-Latency Online Prediction Serving System*. NSDI. arXiv:1612.03079

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const VOCAB_SIZE: usize = 64;
const EMBED_DIM: usize = 32;
const HIDDEN_DIM: usize = 16;
const NUM_CLASSES: usize = 4;
const CLASS_NAMES: [&str; NUM_CLASSES] = ["science", "sports", "tech", "arts"];

/// Deterministic weight initialization
fn init_weights(size: usize, seed: u64, label: &str) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let mut h = DefaultHasher::new();
            (seed, label, i).hash(&mut h);
            (h.finish() as f32 / u64::MAX as f32 - 0.5) * 0.1
        })
        .collect()
}

/// Stage 1: Tokenizer
struct Tokenizer {
    vocab_size: usize,
}

impl Tokenizer {
    fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| (c as usize) % self.vocab_size)
            .collect()
    }
}

/// Stage 2: Encoder (token IDs → embeddings)
struct Encoder {
    embedding_table: Vec<f32>, // VOCAB_SIZE x EMBED_DIM
}

impl Encoder {
    fn new(seed: u64) -> Self {
        Self {
            embedding_table: init_weights(VOCAB_SIZE * EMBED_DIM, seed, "embed"),
        }
    }

    fn encode(&self, token_ids: &[usize]) -> Vec<Vec<f32>> {
        token_ids
            .iter()
            .map(|&id| {
                let start = id.min(VOCAB_SIZE - 1) * EMBED_DIM;
                self.embedding_table[start..start + EMBED_DIM].to_vec()
            })
            .collect()
    }
}

/// Stage 3: Pooling (sequence of embeddings → single vector)
enum PoolingStrategy {
    Mean,
    Max,
    First,
}

impl PoolingStrategy {
    fn pool(&self, embeddings: &[Vec<f32>]) -> Vec<f32> {
        if embeddings.is_empty() {
            return vec![0.0; EMBED_DIM];
        }
        let dim = embeddings[0].len();
        match self {
            PoolingStrategy::Mean => {
                let n = embeddings.len() as f32;
                let mut result = vec![0.0; dim];
                for emb in embeddings {
                    for (r, &e) in result.iter_mut().zip(emb.iter()) {
                        *r += e / n;
                    }
                }
                result
            }
            PoolingStrategy::Max => {
                let mut result = vec![f32::NEG_INFINITY; dim];
                for emb in embeddings {
                    for (r, &e) in result.iter_mut().zip(emb.iter()) {
                        *r = r.max(e);
                    }
                }
                result
            }
            PoolingStrategy::First => embeddings[0].clone(),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            PoolingStrategy::Mean => "mean",
            PoolingStrategy::Max => "max",
            PoolingStrategy::First => "first",
        }
    }
}

/// Stage 4: Classifier (embedding → class probabilities)
struct Classifier {
    w1: Vec<f32>, // HIDDEN_DIM x EMBED_DIM
    b1: Vec<f32>, // HIDDEN_DIM
    w2: Vec<f32>, // NUM_CLASSES x HIDDEN_DIM
    b2: Vec<f32>, // NUM_CLASSES
}

impl Classifier {
    fn new(seed: u64) -> Self {
        Self {
            w1: init_weights(HIDDEN_DIM * EMBED_DIM, seed, "cls_w1"),
            b1: init_weights(HIDDEN_DIM, seed, "cls_b1"),
            w2: init_weights(NUM_CLASSES * HIDDEN_DIM, seed, "cls_w2"),
            b2: init_weights(NUM_CLASSES, seed, "cls_b2"),
        }
    }

    fn predict(&self, embedding: &[f32]) -> Vec<f32> {
        // Hidden layer with ReLU
        let mut hidden = self.b1.clone();
        for (o, h) in hidden.iter_mut().enumerate() {
            for (i, &e) in embedding.iter().enumerate() {
                *h += self.w1[o * EMBED_DIM + i] * e;
            }
            *h = h.max(0.0);
        }

        // Output layer
        let mut logits = self.b2.clone();
        for (o, l) in logits.iter_mut().enumerate() {
            for (i, &h) in hidden.iter().enumerate() {
                *l += self.w2[o * HIDDEN_DIM + i] * h;
            }
        }

        // Softmax
        let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }
}

/// Full pipeline combining all stages
struct Pipeline {
    tokenizer: Tokenizer,
    encoder: Encoder,
    pooling: PoolingStrategy,
    classifier: Classifier,
}

impl Pipeline {
    fn new(pooling: PoolingStrategy, seed: u64) -> Self {
        Self {
            tokenizer: Tokenizer::new(VOCAB_SIZE),
            encoder: Encoder::new(seed),
            pooling,
            classifier: Classifier::new(seed),
        }
    }

    fn predict(&self, text: &str) -> PipelineResult {
        let tokens = self.tokenizer.encode(text);
        let embeddings = self.encoder.encode(&tokens);
        let pooled = self.pooling.pool(&embeddings);
        let probs = self.classifier.predict(&pooled);

        let (class_idx, &confidence) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        PipelineResult {
            class_idx,
            class_name: CLASS_NAMES[class_idx],
            confidence,
            probs,
            n_tokens: tokens.len(),
            embed_dim: EMBED_DIM,
        }
    }
}

struct PipelineResult {
    class_idx: usize,
    class_name: &'static str,
    confidence: f32,
    probs: Vec<f32>,
    n_tokens: usize,
    embed_dim: usize,
}

fn main() {
    println!("=== Model Pipeline Inference Example ===\n");

    // =========================================================================
    // Section 1: Pipeline Construction
    // =========================================================================
    println!("1. Pipeline Construction");
    println!("   ─────────────────────────────────────────");

    let pipeline = Pipeline::new(PoolingStrategy::Mean, 42);

    println!("   Stages:");
    println!("     1. Tokenizer:  vocab={VOCAB_SIZE}");
    println!("     2. Encoder:    embed_dim={EMBED_DIM}");
    println!("     3. Pooling:    {}", pipeline.pooling.name());
    println!("     4. Classifier: {EMBED_DIM}→{HIDDEN_DIM}→{NUM_CLASSES}");
    println!("   Classes: {:?}", CLASS_NAMES);
    println!();

    // =========================================================================
    // Section 2: Single Prediction
    // =========================================================================
    println!("2. Single Prediction");
    println!("   ─────────────────────────────────────────");

    let text = "quantum physics breakthrough";
    let result = pipeline.predict(text);

    println!("   Input:      \"{}\"", text);
    println!("   Tokens:     {}", result.n_tokens);
    println!("   Embed dim:  {}", result.embed_dim);
    println!(
        "   Prediction: {} (class {})",
        result.class_name, result.class_idx
    );
    println!("   Confidence: {:.2}%", result.confidence * 100.0);
    println!("   All probs:");
    for (i, (&p, name)) in result.probs.iter().zip(CLASS_NAMES.iter()).enumerate() {
        let bar = "#".repeat((p * 30.0) as usize);
        println!("     [{i}] {name:>8}: {p:.4} {bar}");
    }
    println!();

    // =========================================================================
    // Section 3: Batch Classification
    // =========================================================================
    println!("3. Batch Classification");
    println!("   ─────────────────────────────────────────");

    let texts = [
        "neural network deep learning",
        "soccer world cup final",
        "rust programming language",
        "painting sculpture gallery",
        "genome DNA sequencing",
        "basketball playoffs",
        "cloud computing kubernetes",
        "symphony orchestra concert",
    ];

    println!(
        "   {:>35} {:>10} {:>10}",
        "Input", "Prediction", "Confidence"
    );
    println!("   {}", "─".repeat(58));

    for text in &texts {
        let result = pipeline.predict(text);
        println!(
            "   {:>35} {:>10} {:>9.1}%",
            text,
            result.class_name,
            result.confidence * 100.0
        );
    }
    println!();

    // =========================================================================
    // Section 4: Pooling Strategy Comparison
    // =========================================================================
    println!("4. Pooling Strategy Comparison");
    println!("   ─────────────────────────────────────────");

    let test_text = "advanced machine learning research";
    let strategies = [
        PoolingStrategy::Mean,
        PoolingStrategy::Max,
        PoolingStrategy::First,
    ];

    println!(
        "   {:>8} {:>10} {:>10} {:>24}",
        "Pool", "Class", "Conf%", "Distribution"
    );
    println!("   {}", "─".repeat(55));

    for strategy in strategies {
        let pipe = Pipeline::new(strategy, 42);
        let result = pipe.predict(test_text);
        let dist: String = result
            .probs
            .iter()
            .map(|p| format!("{:.2}", p))
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "   {:>8} {:>10} {:>9.1}% {:>24}",
            pipe.pooling.name(),
            result.class_name,
            result.confidence * 100.0,
            dist
        );
    }
    println!();

    // =========================================================================
    // Section 5: Pipeline Throughput
    // =========================================================================
    println!("5. Pipeline Throughput");
    println!("   ─────────────────────────────────────────");

    let pipeline = Pipeline::new(PoolingStrategy::Mean, 42);
    let n_iters = 1000;

    let start = std::time::Instant::now();
    for _ in 0..n_iters {
        let _ = pipeline.predict("benchmark text for pipeline throughput");
    }
    let elapsed = start.elapsed();

    println!("   Iterations: {n_iters}");
    println!("   Total time: {} ms", elapsed.as_millis());
    println!(
        "   Throughput: {:.0} predictions/sec",
        f64::from(n_iters) / elapsed.as_secs_f64()
    );
    println!(
        "   Avg latency: {:.1} us",
        elapsed.as_micros() as f64 / f64::from(n_iters)
    );
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_encode() {
        let tok = Tokenizer::new(64);
        let tokens = tok.encode("abc");
        assert_eq!(tokens.len(), 3);
        assert!(tokens.iter().all(|&t| t < 64));
    }

    #[test]
    fn test_encoder_output_shape() {
        let enc = Encoder::new(42);
        let embeddings = enc.encode(&[1, 2, 3]);
        assert_eq!(embeddings.len(), 3);
        assert_eq!(embeddings[0].len(), EMBED_DIM);
    }

    #[test]
    fn test_pooling_mean() {
        let embs = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let pooled = PoolingStrategy::Mean.pool(&embs);
        assert!((pooled[0] - 2.0).abs() < 1e-6);
        assert!((pooled[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_pooling_max() {
        let embs = vec![vec![1.0, 4.0], vec![3.0, 2.0]];
        let pooled = PoolingStrategy::Max.pool(&embs);
        assert!((pooled[0] - 3.0).abs() < 1e-6);
        assert!((pooled[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_classifier_probs_sum() {
        let cls = Classifier::new(42);
        let emb = vec![0.5; EMBED_DIM];
        let probs = cls.predict(&emb);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_pipeline_deterministic() {
        let p = Pipeline::new(PoolingStrategy::Mean, 42);
        let r1 = p.predict("test");
        let r2 = p.predict("test");
        assert_eq!(r1.class_idx, r2.class_idx);
        assert_eq!(r1.probs, r2.probs);
    }

    #[test]
    fn test_pipeline_result_valid() {
        let p = Pipeline::new(PoolingStrategy::Mean, 42);
        let r = p.predict("hello world");
        assert!(r.class_idx < NUM_CLASSES);
        assert!(r.confidence > 0.0 && r.confidence <= 1.0);
        assert_eq!(r.probs.len(), NUM_CLASSES);
    }
}
