#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports, clippy::wildcard_imports)]
use super::helpers::*;
use proptest::prelude::*;
#[allow(unused_imports)]
use std::collections::{HashMap, HashSet};

/// Maximum document size in characters
pub const MAX_DOC_SIZE: usize = 100_000;

/// Default summary length (sentences)
pub const DEFAULT_SUMMARY_LENGTH: usize = 3;

// ============================================================================
// Document Processing
// ============================================================================

/// A document to summarize
#[derive(Debug, Clone)]
pub struct Document {
    // Document title
    pub title: String,
    // Full text content
    pub content: String,
    // Extracted sentences
    pub sentences: Vec<String>,
}

impl Document {
    /// Create from text
    #[must_use]
    pub fn new(title: &str, content: &str) -> Self {
        let sentences = split_sentences(content);
        Self {
            title: title.to_string(),
            content: content.to_string(),
            sentences,
        }
    }

    /// Word count
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
    }

    /// Sentence count
    #[must_use]
    pub fn sentence_count(&self) -> usize {
        self.sentences.len()
    }

    /// Character count
    #[must_use]
    pub fn char_count(&self) -> usize {
        self.content.chars().count()
    }
}

// ============================================================================
// Text Statistics
// ============================================================================

/// Word frequency counter
#[derive(Debug, Clone, Default)]
pub struct WordFrequency {
    pub counts: HashMap<String, usize>,
    pub total: usize,
}

impl WordFrequency {
    /// Create new counter
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Count words in text
    pub fn count(&mut self, text: &str) {
        for word in tokenize(text) {
            if !is_stopword(&word) {
                *self.counts.entry(word).or_insert(0) += 1;
                self.total += 1;
            }
        }
    }

    /// Get frequency of word
    #[must_use]
    pub fn frequency(&self, word: &str) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        let count = self.counts.get(&word.to_lowercase()).copied().unwrap_or(0);
        count as f32 / self.total as f32
    }

    /// Get term frequency (TF)
    #[must_use]
    pub fn tf(&self, word: &str) -> f32 {
        let count = self.counts.get(&word.to_lowercase()).copied().unwrap_or(0);
        if count == 0 {
            0.0
        } else {
            1.0 + (count as f32).ln()
        }
    }

    /// Get top N words
    #[must_use]
    pub fn top_words(&self, n: usize) -> Vec<(String, usize)> {
        let mut sorted: Vec<_> = self.counts.iter().map(|(k, v)| (k.clone(), *v)).collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted.into_iter().take(n).collect()
    }

    /// Total unique words
    #[must_use]
    pub fn vocabulary_size(&self) -> usize {
        self.counts.len()
    }
}

// ============================================================================
// Sentence Scoring
// ============================================================================

/// Sentence with score
#[derive(Debug, Clone)]
pub struct ScoredSentence {
    // Original sentence text
    pub text: String,
    // Position in document (0-indexed)
    pub position: usize,
    // Overall score
    pub score: f32,
    // Score breakdown
    pub score_components: ScoreComponents,
}

/// Score breakdown
#[derive(Debug, Clone, Default)]
pub struct ScoreComponents {
    // TF-IDF based score
    pub tfidf_score: f32,
    // Position score (earlier = higher)
    pub position_score: f32,
    // Length score (prefer medium length)
    pub length_score: f32,
    // Title overlap score
    pub title_score: f32,
}

/// Sentence scorer
pub struct SentenceScorer {
    // Document-level word frequencies
    pub doc_freq: WordFrequency,
    // Title words (for overlap scoring)
    pub title_words: HashSet<String>,
    // Total sentences in document
    pub sentence_count: usize,
}

impl SentenceScorer {
    /// Create scorer for document
    #[must_use]
    pub fn new(doc: &Document) -> Self {
        let mut doc_freq = WordFrequency::new();
        for sentence in &doc.sentences {
            doc_freq.count(sentence);
        }

        let title_words: HashSet<String> = tokenize(&doc.title)
            .into_iter()
            .filter(|w| !is_stopword(w))
            .collect();

        Self {
            doc_freq,
            title_words,
            sentence_count: doc.sentences.len(),
        }
    }

    /// Score a sentence
    #[must_use]
    pub fn score(&self, sentence: &str, position: usize) -> ScoredSentence {
        let words = tokenize(sentence);
        let word_count = words.len();

        // TF-IDF score
        let tfidf_score: f32 = words
            .iter()
            .filter(|w| !is_stopword(w))
            .map(|w| self.doc_freq.tf(w))
            .sum::<f32>()
            / (word_count as f32).max(1.0);

        // Position score (first sentences are more important)
        let position_score = if self.sentence_count > 0 {
            1.0 - (position as f32 / self.sentence_count as f32)
        } else {
            0.5
        };

        // Length score (prefer 10-25 words)
        let length_score = if word_count < 5 {
            0.3
        } else if word_count < 10 {
            0.7
        } else if word_count <= 25 {
            1.0
        } else if word_count <= 40 {
            0.8
        } else {
            0.5
        };

        // Title overlap score
        let title_overlap: usize = words
            .iter()
            .filter(|w| self.title_words.contains(*w))
            .count();
        let title_score = if self.title_words.is_empty() {
            0.5
        } else {
            (title_overlap as f32 / self.title_words.len() as f32).min(1.0)
        };

        // Combined score
        let score =
            tfidf_score * 0.4 + position_score * 0.3 + length_score * 0.15 + title_score * 0.15;

        ScoredSentence {
            text: sentence.to_string(),
            position,
            score,
            score_components: ScoreComponents {
                tfidf_score,
                position_score,
                length_score,
                title_score,
            },
        }
    }
}

// ============================================================================
// Summarization
// ============================================================================

/// Summary configuration
#[derive(Debug, Clone)]
pub struct SummaryConfig {
    // Number of sentences in summary
    pub num_sentences: usize,
    // Minimum sentence length (words)
    pub min_sentence_words: usize,
    // Maximum sentence length (words)
    pub max_sentence_words: usize,
    // Preserve original order
    pub preserve_order: bool,
}

impl Default for SummaryConfig {
    fn default() -> Self {
        Self {
            num_sentences: DEFAULT_SUMMARY_LENGTH,
            min_sentence_words: 5,
            max_sentence_words: 50,
            preserve_order: true,
        }
    }
}

impl SummaryConfig {
    /// Create new config
    #[must_use]
    pub fn new(num_sentences: usize) -> Self {
        Self {
            num_sentences,
            ..Default::default()
        }
    }

    /// Set preserve order
    #[must_use]
    pub fn preserve_order(mut self, preserve: bool) -> Self {
        self.preserve_order = preserve;
        self
    }
}

/// Summarization result
#[derive(Debug)]
pub struct Summary {
    // Summary sentences
    pub sentences: Vec<ScoredSentence>,
    // Original document stats
    pub original_word_count: usize,
    // Summary word count
    pub summary_word_count: usize,
    // Compression ratio
    pub compression_ratio: f32,
    // Top keywords
    pub keywords: Vec<String>,
}

impl Summary {
    /// Get summary text
    #[must_use]
    pub fn text(&self) -> String {
        self.sentences
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Get sentences as list
    #[must_use]
    pub fn as_bullet_points(&self) -> String {
        self.sentences
            .iter()
            .map(|s| format!("• {}", s.text))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Document summarizer
pub struct Summarizer {
    pub config: SummaryConfig,
}

impl Summarizer {
    /// Create with default config
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SummaryConfig::default(),
        }
    }

    /// Create with config
    #[must_use]
    pub fn with_config(config: SummaryConfig) -> Self {
        Self { config }
    }

    /// Summarize a document
    #[must_use]
    pub fn summarize(&self, doc: &Document) -> Summary {
        let scorer = SentenceScorer::new(doc);

        // Score all sentences
        let mut scored: Vec<ScoredSentence> = doc
            .sentences
            .iter()
            .enumerate()
            .map(|(i, s)| scorer.score(s, i))
            .filter(|s| {
                let word_count = s.text.split_whitespace().count();
                word_count >= self.config.min_sentence_words
                    && word_count <= self.config.max_sentence_words
            })
            .collect();

        // Sort by score (descending)
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top sentences
        let mut selected: Vec<ScoredSentence> =
            scored.into_iter().take(self.config.num_sentences).collect();

        // Optionally restore original order
        if self.config.preserve_order {
            selected.sort_by_key(|s| s.position);
        }

        let summary_word_count: usize = selected
            .iter()
            .map(|s| s.text.split_whitespace().count())
            .sum();

        let compression_ratio = if doc.word_count() > 0 {
            summary_word_count as f32 / doc.word_count() as f32
        } else {
            1.0
        };

        // Extract keywords
        let keywords = scorer
            .doc_freq
            .top_words(5)
            .into_iter()
            .map(|(w, _)| w)
            .collect();

        Summary {
            sentences: selected,
            original_word_count: doc.word_count(),
            summary_word_count,
            compression_ratio,
            keywords,
        }
    }
}

impl Default for Summarizer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Summary Quality
// ============================================================================

/// Quality metrics for summary
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    // Coverage of important terms
    pub term_coverage: f32,
    // Redundancy score (lower is better)
    pub redundancy: f32,
    // Coherence estimate
    pub coherence: f32,
    // Overall quality score
    pub quality_score: f32,
}

impl QualityMetrics {
    /// Evaluate summary quality
    #[must_use]
    pub fn evaluate(doc: &Document, summary: &Summary) -> Self {
        // Term coverage: how many important doc terms appear in summary
        let mut doc_freq = WordFrequency::new();
        doc_freq.count(&doc.content);
        let top_terms: HashSet<String> =
            doc_freq.top_words(20).into_iter().map(|(w, _)| w).collect();

        let summary_text = summary.text();
        let summary_words: HashSet<String> = tokenize(&summary_text).into_iter().collect();

        let covered = top_terms.intersection(&summary_words).count();
        let term_coverage = if top_terms.is_empty() {
            1.0
        } else {
            covered as f32 / top_terms.len() as f32
        };

        // Redundancy: similarity between summary sentences
        let redundancy = if summary.sentences.len() < 2 {
            0.0
        } else {
            let mut total_sim = 0.0_f32;
            let mut pairs = 0;
            for i in 0..summary.sentences.len() {
                for j in (i + 1)..summary.sentences.len() {
                    let w1: HashSet<String> =
                        tokenize(&summary.sentences[i].text).into_iter().collect();
                    let w2: HashSet<String> =
                        tokenize(&summary.sentences[j].text).into_iter().collect();
                    let intersection = w1.intersection(&w2).count();
                    let union = w1.union(&w2).count();
                    if union > 0 {
                        total_sim += intersection as f32 / union as f32;
                    }
                    pairs += 1;
                }
            }
            if pairs > 0 {
                total_sim / pairs as f32
            } else {
                0.0
            }
        };

        // Coherence: average position distance (smaller = more coherent flow)
        let coherence = if summary.sentences.len() < 2 {
            1.0
        } else {
            let mut total_dist = 0.0_f32;
            for i in 1..summary.sentences.len() {
                let dist = (summary.sentences[i].position as f32
                    - summary.sentences[i - 1].position as f32)
                    .abs();
                total_dist += dist;
            }
            let avg_dist = total_dist / (summary.sentences.len() - 1) as f32;
            1.0 / (1.0 + avg_dist * 0.1)
        };

        // Overall quality
        let quality_score = term_coverage * 0.4 + (1.0 - redundancy) * 0.3 + coherence * 0.3;

        Self {
            term_coverage,
            redundancy,
            coherence,
            quality_score,
        }
    }
}

// ============================================================================
// Main
// ============================================================================
