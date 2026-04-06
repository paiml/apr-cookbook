#![allow(unused_imports)]
//! # Demo N: Streaming Sentiment Analysis
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! High-throughput sentiment analysis on streaming text data.
//! Demonstrates batching, async patterns, and backpressure handling.
//!
//! ```bash
//! cargo run --example streaming_sentiment
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
//! - Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*. NAACL. arXiv:1810.04805

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== Demo N: Streaming Sentiment Analysis ===\n");
    let mut analyzer = StreamingAnalyzer::new().with_batch_size(8);
    let mut generator = TextGenerator::new(42);
    let mut window = TimeWindow::new(20);

    println!("--- Single Analysis ---");
    for text in [
        "This movie is absolutely amazing and wonderful!",
        "Terrible product, worst purchase I ever made.",
        "It's okay, nothing special.",
        "I love this so much, best thing ever!",
        "Not good at all, very disappointing.",
    ] {
        let r = analyzer.analyze(text);
        println!("{} [{:.2}] \"{}\"", r.sentiment.emoji(), r.score, text);
    }

    println!("\n--- Streaming Batch Processing ---");
    let batch_texts = generator.generate_batch(50, 0.0);
    for text in &batch_texts {
        if let Some(results) = analyzer.submit(text) {
            println!("Processed batch of {} items", results.len());
            for r in results {
                window.add(r);
            }
        }
    }
    let remaining = analyzer.flush();
    println!("Flushed remaining {} items", remaining.len());
    for r in remaining {
        window.add(r);
    }

    let agg = window.aggregate();
    println!("\n--- Window Aggregate ---");
    println!(
        "Items: {}  Avg score: {:.3}  Avg confidence: {:.3}",
        agg.count, agg.avg_score, agg.avg_confidence
    );
    if let Some(d) = agg.dominant_sentiment {
        println!("Dominant: {} {:?}", d.emoji(), d);
    }

    let stats = analyzer.stats();
    println!("\n--- Statistics ---");
    println!(
        "Submitted: {}  Processed: {}  Batches: {}  Avg latency: {:.1}us  Throughput: {:.0}/s",
        stats.total_submitted,
        stats.total_processed,
        stats.batches_processed,
        stats.avg_batch_latency_us(),
        stats.throughput()
    );
    println!("\n=== Demo N Complete ===");
}

// ---- Tests ------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentiment_from_score_and_value() {
        assert_eq!(Sentiment::from_score(-0.8), Sentiment::VeryNegative);
        assert_eq!(Sentiment::from_score(-0.4), Sentiment::Negative);
        assert_eq!(Sentiment::from_score(0.0), Sentiment::Neutral);
        assert_eq!(Sentiment::from_score(0.4), Sentiment::Positive);
        assert_eq!(Sentiment::from_score(0.8), Sentiment::VeryPositive);
        assert_eq!(Sentiment::VeryNegative.to_value(), -2);
        assert_eq!(Sentiment::Neutral.to_value(), 0);
        assert_eq!(Sentiment::VeryPositive.to_value(), 2);
    }

    #[test]
    fn test_sentiment_result() {
        let r = SentimentResult::new("test", 0.5, 0.9, 100);
        assert_eq!(r.text, "test");
        assert!(r.is_positive());
        let r2 = SentimentResult::new("neg", -0.5, 0.9, 100);
        assert!(r2.is_negative());
    }

    #[test]
    fn test_tokenizer_and_model() {
        let tok = Tokenizer::new();
        assert!(tok.vocab_size() > 0);
        let tokens = tok.tokenize("This is good");
        assert!(!tokens.is_empty());
        let model = SentimentModel::new();
        let pos_tokens = tok.tokenize("great excellent amazing");
        assert!(model.predict(&pos_tokens).0 > 0.0);
        let neg_tokens = tok.tokenize("terrible awful horrible");
        assert!(model.predict(&neg_tokens).0 < 0.0);
    }

    #[test]
    fn test_negation() {
        let tok = Tokenizer::new();
        let model = SentimentModel::new();
        let tokens = tok.tokenize("not good");
        assert!(model.predict(&tokens).0 < 0.0);
    }

    #[test]
    fn test_streaming_analyzer_batching() {
        let mut analyzer = StreamingAnalyzer::new().with_batch_size(2);
        assert!(analyzer.submit("test 1").is_none());
        assert!(analyzer.submit("test 2").is_some());

        let mut analyzer2 = StreamingAnalyzer::new();
        analyzer2.submit("test");
        assert_eq!(analyzer2.flush().len(), 1);
        assert_eq!(analyzer2.stats().total_submitted, 1);
    }

    #[test]
    fn test_analyzer_single() {
        let analyzer = StreamingAnalyzer::new();
        let result = analyzer.analyze("This is great!");
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_time_window() {
        let mut window = TimeWindow::new(2);
        assert!(window.is_empty());
        window.add(SentimentResult::new("a", 0.5, 0.9, 100));
        window.add(SentimentResult::new("b", 0.3, 0.8, 100));
        assert_eq!(window.len(), 2);
        window.add(SentimentResult::new("c", 0.5, 0.9, 100));
        assert_eq!(window.len(), 2); // overflow evicts oldest
        let agg = window.aggregate();
        assert_eq!(agg.count, 2);
    }

    #[test]
    fn test_window_aggregate_score() {
        let mut window = TimeWindow::new(10);
        window.add(SentimentResult::new("a", 0.5, 0.9, 100));
        window.add(SentimentResult::new("b", 0.3, 0.8, 100));
        let agg = window.aggregate();
        assert!((agg.avg_score - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_text_generator() {
        let mut gen = TextGenerator::new(42);
        let text = gen.generate(0.0);
        assert!(!text.is_empty());
        let batch = gen.generate_batch(10, 0.0);
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn test_empty_prediction() {
        let model = SentimentModel::new();
        assert_eq!(model.predict(&[]), (0.0, 0.0));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_sentiment_score_bounded(score in -1.0f32..1.0) {
            let value = Sentiment::from_score(score).to_value();
            prop_assert!(value >= -2 && value <= 2);
        }

        #[test]
        fn prop_model_predict_bounded(seed in 0u64..1000) {
            let model = SentimentModel::new();
            let tokens: Vec<usize> = (0..10).map(|i| (seed + i) as usize % 500).collect();
            let (score, confidence) = model.predict(&tokens);
            prop_assert!(score >= -1.0 && score <= 1.0);
            prop_assert!(confidence >= 0.0 && confidence <= 1.0);
        }

        #[test]
        fn prop_window_size_bounded(window_size in 2usize..20, n in 1usize..50) {
            let mut window = TimeWindow::new(window_size);
            for i in 0..n { window.add(SentimentResult::new(&format!("t{}", i), 0.0, 1.0, 0)); }
            prop_assert!(window.len() <= window_size);
        }
    }
}
