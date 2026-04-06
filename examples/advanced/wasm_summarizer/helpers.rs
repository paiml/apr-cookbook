#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use super::types::*;

use proptest::prelude::*;
#[allow(unused_imports)]
use std::collections::{HashMap, HashSet};

/// Split text into sentences
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() && trimmed.split_whitespace().count() >= 3 {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    // Handle text without sentence-ending punctuation
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() && trimmed.split_whitespace().count() >= 3 {
        sentences.push(trimmed);
    }

    sentences
}

/// Tokenize text into words
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .filter(|w| w.len() >= 2)
        .map(String::from)
        .collect()
}

/// Common English stopwords
pub fn is_stopword(word: &str) -> bool {
    pub const STOPWORDS: &[&str] = &[
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
        "from", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
        "does", "did", "will", "would", "could", "should", "may", "might", "must", "can", "this",
        "that", "these", "those", "it", "its", "as", "if", "then", "than", "so", "such", "no",
        "not", "only", "own", "same", "too", "very", "just", "also", "now", "here", "there",
        "when", "where", "why", "how", "all", "each", "every", "both", "few", "more", "most",
        "other", "some", "any", "into", "through", "during", "before", "after", "above", "below",
        "up", "down", "out", "off", "over", "under", "again", "further", "once", "he", "she",
        "they", "we", "you", "i", "me", "my", "your", "his", "her", "their", "our", "which", "who",
        "whom", "what", "whose",
    ];
    STOPWORDS.contains(&word.to_lowercase().as_str())
}

// --- SentenceScorer impl ---

impl SentenceScorer {
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

// --- Summarizer impl ---

impl Summarizer {
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

// --- QualityMetrics impl ---

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
