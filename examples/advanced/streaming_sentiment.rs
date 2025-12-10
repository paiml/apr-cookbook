//! # Demo N: Streaming Sentiment Analysis
//!
//! High-throughput sentiment analysis on streaming text data.
//! Demonstrates batching, async patterns, and backpressure handling.
//!
//! ## Toyota Way Principles
//!
//! - **Heijunka**: Level load through adaptive batching
//! - **Jidoka**: Automatic quality checks on predictions
//! - **Kaizen**: Continuous throughput optimization

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Maximum batch size
pub const MAX_BATCH_SIZE: usize = 32;

/// Vocabulary size for sentiment model
pub const VOCAB_SIZE: usize = 10000;

/// Embedding dimension
pub const EMBED_DIM: usize = 64;

// ============================================================================
// Sentiment Types
// ============================================================================

/// Sentiment classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sentiment {
    /// Strongly negative
    VeryNegative,
    /// Negative
    Negative,
    /// Neutral
    Neutral,
    /// Positive
    Positive,
    /// Strongly positive
    VeryPositive,
}

impl Sentiment {
    /// From score (-1 to 1)
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

    /// To numeric value
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

    /// Display emoji
    #[must_use]
    pub fn emoji(self) -> &'static str {
        match self {
            Self::VeryNegative => "😠",
            Self::Negative => "😟",
            Self::Neutral => "😐",
            Self::Positive => "🙂",
            Self::VeryPositive => "😄",
        }
    }
}

/// Sentiment prediction result
#[derive(Debug, Clone)]
pub struct SentimentResult {
    /// Input text
    pub text: String,
    /// Predicted sentiment
    pub sentiment: Sentiment,
    /// Confidence score (0-1)
    pub confidence: f32,
    /// Raw score (-1 to 1)
    pub score: f32,
    /// Processing latency (microseconds)
    pub latency_us: u64,
}

impl SentimentResult {
    /// Create new result
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

    /// Is positive sentiment
    #[must_use]
    pub fn is_positive(&self) -> bool {
        self.score > 0.0
    }

    /// Is negative sentiment
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.score < 0.0
    }
}

// ============================================================================
// Tokenizer
// ============================================================================

/// Simple tokenizer with vocabulary
pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    unknown_token: usize,
}

impl Tokenizer {
    /// Create with common vocabulary
    #[must_use]
    pub fn new() -> Self {
        let mut vocab = HashMap::new();

        // Add common words with sentiment associations
        let words = [
            // Positive
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
            // Negative
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
            // Neutral/common
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

        for (word, idx) in words {
            vocab.insert(word.to_string(), idx);
        }

        Self {
            vocab,
            unknown_token: 999,
        }
    }

    /// Tokenize text to indices
    #[must_use]
    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(|w| *self.vocab.get(w).unwrap_or(&self.unknown_token))
            .collect()
    }

    /// Vocabulary size
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

// ============================================================================
// Sentiment Model
// ============================================================================

/// Simple sentiment classifier
pub struct SentimentModel {
    /// Word sentiment scores
    word_scores: HashMap<usize, f32>,
    /// Negation tokens
    negation_tokens: Vec<usize>,
    /// Intensifier tokens
    intensifier_tokens: Vec<usize>,
}

impl SentimentModel {
    /// Create new model
    #[must_use]
    pub fn new() -> Self {
        let mut word_scores = HashMap::new();

        // Positive words
        word_scores.insert(1, 0.5); // good
        word_scores.insert(2, 0.7); // great
        word_scores.insert(3, 0.8); // excellent
        word_scores.insert(4, 0.9); // amazing
        word_scores.insert(5, 0.85); // wonderful
        word_scores.insert(6, 0.8); // love
        word_scores.insert(7, 0.6); // happy
        word_scores.insert(8, 0.9); // best
        word_scores.insert(9, 0.85); // fantastic
        word_scores.insert(10, 0.8); // awesome
        word_scores.insert(11, 0.6); // beautiful
        word_scores.insert(12, 0.9); // perfect
        word_scores.insert(13, 0.4); // nice
        word_scores.insert(14, 0.75); // brilliant
        word_scores.insert(15, 0.5); // enjoy

        // Negative words
        word_scores.insert(100, -0.5); // bad
        word_scores.insert(101, -0.8); // terrible
        word_scores.insert(102, -0.85); // awful
        word_scores.insert(103, -0.9); // horrible
        word_scores.insert(104, -0.95); // worst
        word_scores.insert(105, -0.8); // hate
        word_scores.insert(106, -0.5); // sad
        word_scores.insert(107, -0.4); // poor
        word_scores.insert(108, -0.6); // disappointing
        word_scores.insert(109, -0.3); // boring
        word_scores.insert(110, -0.4); // ugly
        word_scores.insert(111, -0.3); // wrong
        word_scores.insert(112, -0.5); // fail
        word_scores.insert(113, -0.4); // broken
        word_scores.insert(114, -0.35); // annoying

        Self {
            word_scores,
            negation_tokens: vec![210],    // "not"
            intensifier_tokens: vec![211], // "very"
        }
    }

    /// Predict sentiment for tokens
    #[must_use]
    pub fn predict(&self, tokens: &[usize]) -> (f32, f32) {
        if tokens.is_empty() {
            return (0.0, 0.0);
        }

        let mut score = 0.0_f32;
        let mut weight = 0.0_f32;
        let mut negate = false;
        let mut intensify = false;

        for &token in tokens {
            if self.negation_tokens.contains(&token) {
                negate = true;
                continue;
            }
            if self.intensifier_tokens.contains(&token) {
                intensify = true;
                continue;
            }

            if let Some(&word_score) = self.word_scores.get(&token) {
                let mut s = word_score;
                if negate {
                    s = -s * 0.8; // Negation reduces but doesn't fully flip
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

        // Confidence based on how many sentiment words we found
        let confidence = (weight / tokens.len() as f32).min(1.0);

        (final_score, confidence)
    }
}

impl Default for SentimentModel {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Streaming Pipeline
// ============================================================================

/// Batch of texts for processing
#[derive(Debug)]
pub struct TextBatch {
    pub texts: Vec<String>,
    pub timestamps: Vec<u64>,
}

impl TextBatch {
    /// Create empty batch
    #[must_use]
    pub fn new() -> Self {
        Self {
            texts: Vec::new(),
            timestamps: Vec::new(),
        }
    }

    /// Add text to batch
    pub fn add(&mut self, text: String, timestamp: u64) {
        self.texts.push(text);
        self.timestamps.push(timestamp);
    }

    /// Batch size
    #[must_use]
    pub fn len(&self) -> usize {
        self.texts.len()
    }

    /// Is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }

    /// Clear batch
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

/// Streaming sentiment analyzer
pub struct StreamingAnalyzer {
    tokenizer: Tokenizer,
    model: SentimentModel,
    /// Pending batch
    batch: TextBatch,
    /// Maximum batch size
    max_batch_size: usize,
    /// Results buffer (for future streaming support)
    #[allow(dead_code)]
    results: VecDeque<SentimentResult>,
    /// Statistics
    stats: StreamStats,
}

impl StreamingAnalyzer {
    /// Create new analyzer
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

    /// Set max batch size
    #[must_use]
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    /// Submit text for analysis
    pub fn submit(&mut self, text: &str) -> Option<Vec<SentimentResult>> {
        let timestamp = self.stats.total_submitted;
        self.batch.add(text.to_string(), timestamp);
        self.stats.total_submitted += 1;

        if self.batch.len() >= self.max_batch_size {
            Some(self.flush())
        } else {
            None
        }
    }

    /// Process current batch and return results
    pub fn flush(&mut self) -> Vec<SentimentResult> {
        if self.batch.is_empty() {
            return Vec::new();
        }

        let start = Instant::now();
        let mut results = Vec::with_capacity(self.batch.len());

        for text in &self.batch.texts {
            let tokens = self.tokenizer.tokenize(text);
            let (score, confidence) = self.model.predict(&tokens);
            let latency = start.elapsed().as_micros() as u64;

            let result = SentimentResult::new(text, score, confidence, latency);
            results.push(result);
        }

        self.stats.total_processed += self.batch.len() as u64;
        self.stats.batches_processed += 1;
        self.stats.total_latency_us += start.elapsed().as_micros() as u64;

        self.batch.clear();
        results
    }

    /// Get statistics
    #[must_use]
    pub fn stats(&self) -> &StreamStats {
        &self.stats
    }

    /// Analyze single text immediately
    #[must_use]
    pub fn analyze(&self, text: &str) -> SentimentResult {
        let start = Instant::now();
        let tokens = self.tokenizer.tokenize(text);
        let (score, confidence) = self.model.predict(&tokens);
        let latency = start.elapsed().as_micros() as u64;
        SentimentResult::new(text, score, confidence, latency)
    }
}

impl Default for StreamingAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Stream processing statistics
#[derive(Debug, Clone)]
pub struct StreamStats {
    /// Total texts submitted
    pub total_submitted: u64,
    /// Total texts processed
    pub total_processed: u64,
    /// Batches processed
    pub batches_processed: u64,
    /// Total processing time (microseconds)
    pub total_latency_us: u64,
}

impl StreamStats {
    /// Create new stats
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_submitted: 0,
            total_processed: 0,
            batches_processed: 0,
            total_latency_us: 0,
        }
    }

    /// Average latency per batch
    #[must_use]
    pub fn avg_batch_latency_us(&self) -> f64 {
        if self.batches_processed == 0 {
            0.0
        } else {
            self.total_latency_us as f64 / self.batches_processed as f64
        }
    }

    /// Throughput (items per second estimate)
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

// ============================================================================
// Aggregation
// ============================================================================

/// Time window for aggregation
#[derive(Debug, Clone)]
pub struct TimeWindow {
    /// Window size in items
    pub size: usize,
    /// Results in window
    results: VecDeque<SentimentResult>,
}

impl TimeWindow {
    /// Create new window
    #[must_use]
    pub fn new(size: usize) -> Self {
        Self {
            size,
            results: VecDeque::with_capacity(size),
        }
    }

    /// Add result to window
    pub fn add(&mut self, result: SentimentResult) {
        if self.results.len() >= self.size {
            self.results.pop_front();
        }
        self.results.push_back(result);
    }

    /// Get aggregate sentiment
    #[must_use]
    pub fn aggregate(&self) -> WindowAggregate {
        if self.results.is_empty() {
            return WindowAggregate::default();
        }

        let mut sentiment_counts = HashMap::new();
        let mut total_score = 0.0_f32;
        let mut total_confidence = 0.0_f32;

        for result in &self.results {
            *sentiment_counts.entry(result.sentiment).or_insert(0) += 1;
            total_score += result.score;
            total_confidence += result.confidence;
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

    /// Most common sentiment
    #[must_use]
    pub fn dominant_sentiment(&self) -> Option<Sentiment> {
        let mut counts: HashMap<Sentiment, usize> = HashMap::new();
        for result in &self.results {
            *counts.entry(result.sentiment).or_insert(0) += 1;
        }
        counts.into_iter().max_by_key(|(_, c)| *c).map(|(s, _)| s)
    }

    /// Current window size
    #[must_use]
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }
}

/// Aggregated window statistics
#[derive(Debug, Clone, Default)]
pub struct WindowAggregate {
    /// Number of items in window
    pub count: usize,
    /// Average sentiment score
    pub avg_score: f32,
    /// Average confidence
    pub avg_confidence: f32,
    /// Sentiment distribution
    pub sentiment_distribution: HashMap<Sentiment, usize>,
    /// Most common sentiment
    pub dominant_sentiment: Option<Sentiment>,
}

// ============================================================================
// Text Generator (for testing)
// ============================================================================

/// Generate sample texts for testing
pub struct TextGenerator {
    rng: SimpleRng,
    positive_phrases: Vec<&'static str>,
    negative_phrases: Vec<&'static str>,
    neutral_phrases: Vec<&'static str>,
}

impl TextGenerator {
    /// Create new generator
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

    /// Generate random text with sentiment bias
    pub fn generate(&mut self, sentiment_bias: f32) -> String {
        let r = self.rng.next_f32();
        let adjusted = r + sentiment_bias * 0.3;

        if adjusted > 0.6 {
            let idx = self.rng.next_u64() as usize % self.positive_phrases.len();
            self.positive_phrases[idx].to_string()
        } else if adjusted < 0.4 {
            let idx = self.rng.next_u64() as usize % self.negative_phrases.len();
            self.negative_phrases[idx].to_string()
        } else {
            let idx = self.rng.next_u64() as usize % self.neutral_phrases.len();
            self.neutral_phrases[idx].to_string()
        }
    }

    /// Generate batch of texts
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

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("=== Demo N: Streaming Sentiment Analysis ===\n");

    let mut analyzer = StreamingAnalyzer::new().with_batch_size(8);
    let mut generator = TextGenerator::new(42);
    let mut window = TimeWindow::new(20);

    println!("--- Single Analysis ---");
    let examples = [
        "This movie is absolutely amazing and wonderful!",
        "Terrible product, worst purchase I ever made.",
        "It's okay, nothing special.",
        "I love this so much, best thing ever!",
        "Not good at all, very disappointing.",
    ];

    for text in examples {
        let result = analyzer.analyze(text);
        println!(
            "{} [{:.2}] \"{}\"",
            result.sentiment.emoji(),
            result.score,
            text
        );
    }

    println!("\n--- Streaming Batch Processing ---");
    let batch_texts = generator.generate_batch(50, 0.0);

    for text in &batch_texts {
        if let Some(results) = analyzer.submit(text) {
            println!("Processed batch of {} items", results.len());
            for result in results {
                window.add(result);
            }
        }
    }

    // Flush remaining
    let remaining = analyzer.flush();
    println!("Flushed remaining {} items", remaining.len());
    for result in remaining {
        window.add(result);
    }

    println!("\n--- Window Aggregate ---");
    let agg = window.aggregate();
    println!("Items in window: {}", agg.count);
    println!("Average score: {:.3}", agg.avg_score);
    println!("Average confidence: {:.3}", agg.avg_confidence);
    if let Some(dominant) = agg.dominant_sentiment {
        println!("Dominant sentiment: {} {:?}", dominant.emoji(), dominant);
    }

    println!("\n--- Statistics ---");
    let stats = analyzer.stats();
    println!("Total submitted: {}", stats.total_submitted);
    println!("Total processed: {}", stats.total_processed);
    println!("Batches: {}", stats.batches_processed);
    println!("Avg batch latency: {:.1} µs", stats.avg_batch_latency_us());
    println!("Throughput: {:.0} items/sec", stats.throughput());

    println!("\n=== Demo N Complete ===");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentiment_from_score() {
        assert_eq!(Sentiment::from_score(-0.8), Sentiment::VeryNegative);
        assert_eq!(Sentiment::from_score(-0.4), Sentiment::Negative);
        assert_eq!(Sentiment::from_score(0.0), Sentiment::Neutral);
        assert_eq!(Sentiment::from_score(0.4), Sentiment::Positive);
        assert_eq!(Sentiment::from_score(0.8), Sentiment::VeryPositive);
    }

    #[test]
    fn test_sentiment_to_value() {
        assert_eq!(Sentiment::VeryNegative.to_value(), -2);
        assert_eq!(Sentiment::Neutral.to_value(), 0);
        assert_eq!(Sentiment::VeryPositive.to_value(), 2);
    }

    #[test]
    fn test_sentiment_result_new() {
        let result = SentimentResult::new("test", 0.5, 0.9, 100);
        assert_eq!(result.text, "test");
        assert!(result.is_positive());
    }

    #[test]
    fn test_tokenizer_new() {
        let tokenizer = Tokenizer::new();
        assert!(tokenizer.vocab_size() > 0);
    }

    #[test]
    fn test_tokenizer_tokenize() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.tokenize("This is good");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_sentiment_model_new() {
        let model = SentimentModel::new();
        assert!(!model.word_scores.is_empty());
    }

    #[test]
    fn test_sentiment_model_positive() {
        let tokenizer = Tokenizer::new();
        let model = SentimentModel::new();
        let tokens = tokenizer.tokenize("great excellent amazing");
        let (score, _) = model.predict(&tokens);
        assert!(score > 0.0);
    }

    #[test]
    fn test_sentiment_model_negative() {
        let tokenizer = Tokenizer::new();
        let model = SentimentModel::new();
        let tokens = tokenizer.tokenize("terrible awful horrible");
        let (score, _) = model.predict(&tokens);
        assert!(score < 0.0);
    }

    #[test]
    fn test_sentiment_model_negation() {
        let tokenizer = Tokenizer::new();
        let model = SentimentModel::new();
        let tokens = tokenizer.tokenize("not good");
        let (score, _) = model.predict(&tokens);
        assert!(score < 0.0); // "not good" should be negative
    }

    #[test]
    fn test_text_batch_new() {
        let batch = TextBatch::new();
        assert!(batch.is_empty());
    }

    #[test]
    fn test_text_batch_add() {
        let mut batch = TextBatch::new();
        batch.add("test".to_string(), 0);
        assert_eq!(batch.len(), 1);
    }

    #[test]
    fn test_streaming_analyzer_new() {
        let analyzer = StreamingAnalyzer::new();
        assert_eq!(analyzer.stats().total_submitted, 0);
    }

    #[test]
    fn test_streaming_analyzer_analyze() {
        let analyzer = StreamingAnalyzer::new();
        let result = analyzer.analyze("This is great!");
        assert!(result.score > 0.0);
    }

    #[test]
    fn test_streaming_analyzer_submit() {
        let mut analyzer = StreamingAnalyzer::new().with_batch_size(2);
        let r1 = analyzer.submit("test 1");
        assert!(r1.is_none());
        let r2 = analyzer.submit("test 2");
        assert!(r2.is_some());
    }

    #[test]
    fn test_streaming_analyzer_flush() {
        let mut analyzer = StreamingAnalyzer::new();
        analyzer.submit("test");
        let results = analyzer.flush();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_stream_stats() {
        let stats = StreamStats::new();
        assert_eq!(stats.total_submitted, 0);
        assert_eq!(stats.throughput(), 0.0);
    }

    #[test]
    fn test_time_window_new() {
        let window = TimeWindow::new(10);
        assert!(window.is_empty());
    }

    #[test]
    fn test_time_window_add() {
        let mut window = TimeWindow::new(10);
        window.add(SentimentResult::new("test", 0.5, 0.9, 100));
        assert_eq!(window.len(), 1);
    }

    #[test]
    fn test_time_window_overflow() {
        let mut window = TimeWindow::new(2);
        window.add(SentimentResult::new("a", 0.5, 0.9, 100));
        window.add(SentimentResult::new("b", 0.5, 0.9, 100));
        window.add(SentimentResult::new("c", 0.5, 0.9, 100));
        assert_eq!(window.len(), 2);
    }

    #[test]
    fn test_time_window_aggregate() {
        let mut window = TimeWindow::new(10);
        window.add(SentimentResult::new("a", 0.5, 0.9, 100));
        window.add(SentimentResult::new("b", 0.3, 0.8, 100));
        let agg = window.aggregate();
        assert_eq!(agg.count, 2);
        assert!((agg.avg_score - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_text_generator() {
        let mut gen = TextGenerator::new(42);
        let text = gen.generate(0.0);
        assert!(!text.is_empty());
    }

    #[test]
    fn test_text_generator_batch() {
        let mut gen = TextGenerator::new(42);
        let batch = gen.generate_batch(10, 0.0);
        assert_eq!(batch.len(), 10);
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
            let sentiment = Sentiment::from_score(score);
            let value = sentiment.to_value();
            prop_assert!(value >= -2 && value <= 2);
        }

        #[test]
        fn prop_tokenize_non_empty(text in "[a-z ]{5,50}") {
            let tokenizer = Tokenizer::new();
            let tokens = tokenizer.tokenize(&text);
            // May be empty if all words are filtered
            prop_assert!(tokens.len() <= text.split_whitespace().count());
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
        fn prop_batch_size_respected(size in 1usize..20, n in 1usize..50) {
            let mut analyzer = StreamingAnalyzer::new().with_batch_size(size);
            let mut batches = 0;
            for i in 0..n {
                if analyzer.submit(&format!("text {}", i)).is_some() {
                    batches += 1;
                }
            }
            prop_assert!(batches <= n / size + 1);
        }

        #[test]
        fn prop_window_size_bounded(window_size in 2usize..20, n in 1usize..50) {
            let mut window = TimeWindow::new(window_size);
            for i in 0..n {
                window.add(SentimentResult::new(&format!("t{}", i), 0.0, 1.0, 0));
            }
            prop_assert!(window.len() <= window_size);
        }

        #[test]
        fn prop_aggregate_count_correct(n in 1usize..20) {
            let mut window = TimeWindow::new(100);
            for i in 0..n {
                window.add(SentimentResult::new(&format!("t{}", i), 0.0, 1.0, 0));
            }
            let agg = window.aggregate();
            prop_assert_eq!(agg.count, n);
        }
    }
}
