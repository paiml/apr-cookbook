#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
use proptest::prelude::*;
#[allow(unused_imports)]
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
    pub vocab: HashMap<String, usize>,
    pub unknown_token: usize,
}

impl Tokenizer {
    // new() moved to helpers.rs

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

// ---- Sentiment Model --------------------------------------------------------

pub struct SentimentModel {
    pub word_scores: HashMap<usize, f32>,
    pub negation_tokens: Vec<usize>,
    pub intensifier_tokens: Vec<usize>,
}

impl SentimentModel {
    // new() moved to helpers.rs

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
    pub tokenizer: Tokenizer,
    pub model: SentimentModel,
    pub batch: TextBatch,
    pub max_batch_size: usize,
    #[allow(dead_code)]
    pub results: VecDeque<SentimentResult>,
    pub stats: StreamStats,
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

    // submit(), flush(), analyze() moved to helpers.rs

    #[must_use]
    pub fn stats(&self) -> &StreamStats {
        &self.stats
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
    pub results: VecDeque<SentimentResult>,
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
    pub rng: SimpleRng,
    pub positive_phrases: Vec<&'static str>,
    pub negative_phrases: Vec<&'static str>,
    pub neutral_phrases: Vec<&'static str>,
}

// TextGenerator impl (new, generate, generate_batch) moved to helpers.rs

pub struct SimpleRng {
    pub state: u64,
}
impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }
}

// ---- Main -------------------------------------------------------------------
