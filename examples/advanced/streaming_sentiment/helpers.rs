#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports, clippy::wildcard_imports)]
use super::types::*;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

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
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
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
}

impl Default for SentimentModel {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamingAnalyzer {
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
    pub fn analyze(&self, text: &str) -> SentimentResult {
        let start = Instant::now();
        let tokens = self.tokenizer.tokenize(text);
        let (score, confidence) = self.model.predict(&tokens);
        SentimentResult::new(text, score, confidence, start.elapsed().as_micros() as u64)
    }
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
