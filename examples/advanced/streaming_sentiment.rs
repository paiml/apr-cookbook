//! # Demo N: Streaming Sentiment Analysis
//!
//! High-throughput sentiment analysis on streaming text data.
//! Demonstrates batching, async patterns, and backpressure handling.
//!
//! ```bash
//! cargo run --example streaming_sentiment
//! ```
//!
//! ## References
//! - Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers*. NAACL. arXiv:1810.04805

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

pub const MAX_BATCH_SIZE: usize = 32;
pub const VOCAB_SIZE: usize = 10000;
pub const EMBED_DIM: usize = 64;

// ---- Sentiment Types --------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sentiment {
    VeryNegative,
    Negative,
    Neutral,
    Positive,
    VeryPositive,
}

impl Sentiment {
    #[must_use]
    pub fn from_score(score: f32) -> Self {
        if score < -0.6 {
            Self::VeryNegative
        } else if score < -0.2 {
            Self::Negative
        } else if score < 0.2 {
            Self::Neutral
        } else if score < 0.6 {
            Self::Positive
        } else {
            Self::VeryPositive
        }
    }

    #[must_use]
    pub fn to_value(self) -> i8 {
        match self {
            Self::VeryNegative => -2,
            Self::Negative => -1,
            Self::Neutral => 0,
            Self::Positive => 1,
            Self::VeryPositive => 2,
        }
    }

    #[must_use]
    pub fn emoji(self) -> &'static str {
        match self {
            Self::VeryNegative => "!!",
            Self::Negative => ":-(",
            Self::Neutral => ":-|",
            Self::Positive => ":-)",
            Self::VeryPositive => ":-D",
        }
    }
}

#[derive(Debug, Clone)]
pub struct SentimentResult {
    pub text: String,
    pub sentiment: Sentiment,
    pub confidence: f32,
    pub score: f32,
    pub latency_us: u64,
}

impl SentimentResult {
    #[must_use]
    pub fn new(text: &str, score: f32, confidence: f32, latency_us: u64) -> Self {
        Self {
            text: text.to_string(),
            sentiment: Sentiment::from_score(score),
            confidence,
            score,
            latency_us,
        }
    }
    #[must_use]
    pub fn is_positive(&self) -> bool {
        self.score > 0.0
    }
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.score < 0.0
    }
}

// ---- Tokenizer --------------------------------------------------------------

pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    unknown_token: usize,
}

impl Tokenizer {
    #[must_use]
    pub fn new() -> Self {
        let words: &[(&str, usize)] = &[
            ("good", 1),
            ("great", 2),
            ("excellent", 3),
            ("amazing", 4),
            ("wonderful", 5),
            ("love", 6),
            ("happy", 7),
            ("best", 8),
            ("fantastic", 9),
            ("awesome", 10),
            ("beautiful", 11),
            ("perfect", 12),
            ("nice", 13),
            ("brilliant", 14),
            ("enjoy", 15),
            ("bad", 100),
            ("terrible", 101),
            ("awful", 102),
            ("horrible", 103),
            ("worst", 104),
            ("hate", 105),
            ("sad", 106),
            ("poor", 107),
            ("disappointing", 108),
            ("boring", 109),
            ("ugly", 110),
            ("wrong", 111),
            ("fail", 112),
            ("broken", 113),
            ("annoying", 114),
            ("the", 200),
            ("a", 201),
            ("is", 202),
            ("it", 203),
            ("this", 204),
            ("that", 205),
            ("was", 206),
            ("are", 207),
            ("be", 208),
            ("have", 209),
            ("not", 210),
            ("very", 211),
            ("but", 212),
            ("so", 213),
            ("just", 214),
            ("movie", 300),
            ("product", 301),
            ("service", 302),
            ("food", 303),
            ("place", 304),
        ];
        let vocab = words.iter().map(|&(w, i)| (w.to_string(), i)).collect();
        Self {
            vocab,
            unknown_token: 999,
        }
    }

    #[must_use]
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(|w| *self.vocab.get(w).unwrap_or(&self.unknown_token))
            .collect()
    }
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Sentiment Model --------------------------------------------------------

pub struct SentimentModel {
    word_scores: HashMap<usize, f32>,
    negation_tokens: Vec<usize>,
    intensifier_tokens: Vec<usize>,
}

impl SentimentModel {
    #[must_use]
    pub fn new() -> Self {
        let scores: &[(usize, f32)] = &[
            (1, 0.5),
            (2, 0.7),
            (3, 0.8),
            (4, 0.9),
            (5, 0.85),
            (6, 0.8),
            (7, 0.6),
            (8, 0.9),
            (9, 0.85),
            (10, 0.8),
            (11, 0.6),
            (12, 0.9),
            (13, 0.4),
            (14, 0.75),
            (15, 0.5),
            (100, -0.5),
            (101, -0.8),
            (102, -0.85),
            (103, -0.9),
            (104, -0.95),
            (105, -0.8),
            (106, -0.5),
            (107, -0.4),
            (108, -0.6),
            (109, -0.3),
            (110, -0.4),
            (111, -0.3),
            (112, -0.5),
            (113, -0.4),
            (114, -0.35),
        ];
        Self {
            word_scores: scores.iter().copied().collect(),
            negation_tokens: vec![210],
            intensifier_tokens: vec![211],
        }
    }

    #[must_use]
    pub fn predict(&self, tokens: &[usize]) -> (f32, f32) {
        if tokens.is_empty() {
            return (0.0, 0.0);
        }
        let (mut score, mut weight) = (0.0_f32, 0.0_f32);
        let (mut negate, mut intensify) = (false, false);
        for &token in tokens {
            if self.negation_tokens.contains(&token) {
                negate = true;
                continue;
            }
            if self.intensifier_tokens.contains(&token) {
                intensify = true;
                continue;
            }
            if let Some(&ws) = self.word_scores.get(&token) {
                let mut s = ws;
                if negate {
                    s = -s * 0.8;
                    negate = false;
                }
                if intensify {
                    s *= 1.3;
                    intensify = false;
                }
                score += s;
                weight += 1.0;
            }
        }
        let final_score = if weight > 0.0 {
            (score / weight).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        (final_score, (weight / tokens.len() as f32).min(1.0))
    }
}

impl Default for SentimentModel {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Streaming Pipeline -----------------------------------------------------

#[derive(Debug)]
pub struct TextBatch {
    pub texts: Vec<String>,
    pub timestamps: Vec<u64>,
}

impl TextBatch {
    #[must_use]
    pub fn new() -> Self {
        Self {
            texts: Vec::new(),
            timestamps: Vec::new(),
        }
    }
    pub fn add(&mut self, text: String, timestamp: u64) {
        self.texts.push(text);
        self.timestamps.push(timestamp);
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.texts.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }
    pub fn clear(&mut self) {
        self.texts.clear();
        self.timestamps.clear();
    }
}
impl Default for TextBatch {
    fn default() -> Self {
        Self::new()
    }
}

pub struct StreamingAnalyzer {
    tokenizer: Tokenizer,
    model: SentimentModel,
    batch: TextBatch,
    max_batch_size: usize,
    #[allow(dead_code)]
    results: VecDeque<SentimentResult>,
    stats: StreamStats,
}

impl StreamingAnalyzer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tokenizer: Tokenizer::new(),
            model: SentimentModel::new(),
            batch: TextBatch::new(),
            max_batch_size: MAX_BATCH_SIZE,
            results: VecDeque::new(),
            stats: StreamStats::new(),
        }
    }
    #[must_use]
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    pub fn submit(&mut self, text: &str) -> Option<Vec<SentimentResult>> {
        self.batch.add(text.to_string(), self.stats.total_submitted);
        self.stats.total_submitted += 1;
        if self.batch.len() >= self.max_batch_size {
            Some(self.flush())
        } else {
            None
        }
    }

    pub fn flush(&mut self) -> Vec<SentimentResult> {
        if self.batch.is_empty() {
            return Vec::new();
        }
        let start = Instant::now();
        let results: Vec<SentimentResult> = self
            .batch
            .texts
            .iter()
            .map(|text| {
                let tokens = self.tokenizer.tokenize(text);
                let (score, confidence) = self.model.predict(&tokens);
                SentimentResult::new(text, score, confidence, start.elapsed().as_micros() as u64)
            })
            .collect();
        self.stats.total_processed += self.batch.len() as u64;
        self.stats.batches_processed += 1;
        self.stats.total_latency_us += start.elapsed().as_micros() as u64;
        self.batch.clear();
        results
    }
    #[must_use]
    pub fn stats(&self) -> &StreamStats {
        &self.stats
    }
    #[must_use]
    pub fn analyze(&self, text: &str) -> SentimentResult {
        let start = Instant::now();
        let tokens = self.tokenizer.tokenize(text);
        let (score, confidence) = self.model.predict(&tokens);
        SentimentResult::new(text, score, confidence, start.elapsed().as_micros() as u64)
    }
}
impl Default for StreamingAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct StreamStats {
    pub total_submitted: u64,
    pub total_processed: u64,
    pub batches_processed: u64,
    pub total_latency_us: u64,
}

impl StreamStats {
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_submitted: 0,
            total_processed: 0,
            batches_processed: 0,
            total_latency_us: 0,
        }
    }
    #[must_use]
    pub fn avg_batch_latency_us(&self) -> f64 {
        if self.batches_processed == 0 {
            0.0
        } else {
            self.total_latency_us as f64 / self.batches_processed as f64
        }
    }
    #[must_use]
    pub fn throughput(&self) -> f64 {
        if self.total_latency_us == 0 {
            0.0
        } else {
            self.total_processed as f64 / (self.total_latency_us as f64 / 1_000_000.0)
        }
    }
}
impl Default for StreamStats {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Aggregation ------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct TimeWindow {
    pub size: usize,
    results: VecDeque<SentimentResult>,
}

impl TimeWindow {
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            size,
            results: VecDeque::with_capacity(size),
        }
    }
    pub fn add(&mut self, result: SentimentResult) {
        if self.results.len() >= self.size {
            self.results.pop_front();
        }
        self.results.push_back(result);
    }
    #[must_use]
    pub fn aggregate(&self) -> WindowAggregate {
        if self.results.is_empty() {
            return WindowAggregate::default();
        }
        let mut sentiment_counts = HashMap::new();
        let (mut total_score, mut total_confidence) = (0.0_f32, 0.0_f32);
        for r in &self.results {
            *sentiment_counts.entry(r.sentiment).or_insert(0) += 1;
            total_score += r.score;
            total_confidence += r.confidence;
        }
        let n = self.results.len() as f32;
        WindowAggregate {
            count: self.results.len(),
            avg_score: total_score / n,
            avg_confidence: total_confidence / n,
            sentiment_distribution: sentiment_counts,
            dominant_sentiment: self.dominant_sentiment(),
        }
    }
    #[must_use]
    pub fn dominant_sentiment(&self) -> Option<Sentiment> {
        let mut counts: HashMap<Sentiment, usize> = HashMap::new();
        for r in &self.results {
            *counts.entry(r.sentiment).or_insert(0) += 1;
        }
        counts.into_iter().max_by_key(|(_, c)| *c).map(|(s, _)| s)
    }
    #[must_use]
    pub fn len(&self) -> usize {
        self.results.len()
    }
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }
}

#[derive(Debug, Clone, Default)]
pub struct WindowAggregate {
    pub count: usize,
    pub avg_score: f32,
    pub avg_confidence: f32,
    pub sentiment_distribution: HashMap<Sentiment, usize>,
    pub dominant_sentiment: Option<Sentiment>,
}

// ---- Text Generator ---------------------------------------------------------

pub struct TextGenerator {
    rng: SimpleRng,
    positive_phrases: Vec<&'static str>,
    negative_phrases: Vec<&'static str>,
    neutral_phrases: Vec<&'static str>,
}

impl TextGenerator {
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SimpleRng::new(seed),
            positive_phrases: vec![
                "This is great!",
                "I love this product",
                "Excellent service, very happy",
                "Amazing quality, best purchase ever",
                "Wonderful experience, highly recommend",
                "Perfect in every way",
                "Fantastic results, awesome job",
            ],
            negative_phrases: vec![
                "This is terrible",
                "I hate this product",
                "Awful service, very disappointing",
                "Horrible quality, worst purchase",
                "Bad experience, do not recommend",
                "Poor design, totally broken",
                "Annoying and boring, waste of money",
            ],
            neutral_phrases: vec![
                "This is a product",
                "I bought this item",
                "The service was provided",
                "It arrived on time",
                "Standard quality",
                "As described",
                "Works as expected",
            ],
        }
    }
    pub fn generate(&mut self, sentiment_bias: f32) -> String {
        let r = self.rng.next_f32();
        let adjusted = r + sentiment_bias * 0.3;
        if adjusted > 0.6 {
            self.positive_phrases[self.rng.next_u64() as usize % self.positive_phrases.len()]
                .to_string()
        } else if adjusted < 0.4 {
            self.negative_phrases[self.rng.next_u64() as usize % self.negative_phrases.len()]
                .to_string()
        } else {
            self.neutral_phrases[self.rng.next_u64() as usize % self.neutral_phrases.len()]
                .to_string()
        }
    }
    pub fn generate_batch(&mut self, count: usize, sentiment_bias: f32) -> Vec<String> {
        (0..count).map(|_| self.generate(sentiment_bias)).collect()
    }
}

struct SimpleRng {
    state: u64,
}
impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }
}

// ---- Main -------------------------------------------------------------------

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
