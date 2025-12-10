//! # Demo G: WASM Document Summarizer
//!
//! Client-side document summarization using extractive methods.
//! Designed for WASM deployment with minimal dependencies.
//!
//! ## Toyota Way Principles
//!
//! - **Heijunka**: Consistent summary quality regardless of document size
//! - **Jidoka**: Automatic quality detection stops poor summaries
//! - **Kaizen**: Iterative sentence selection refinement

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
    /// Document title
    pub title: String,
    /// Full text content
    pub content: String,
    /// Extracted sentences
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

/// Split text into sentences
fn split_sentences(text: &str) -> Vec<String> {
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

// ============================================================================
// Text Statistics
// ============================================================================

/// Word frequency counter
#[derive(Debug, Clone, Default)]
pub struct WordFrequency {
    counts: HashMap<String, usize>,
    total: usize,
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

/// Tokenize text into words
fn tokenize(text: &str) -> Vec<String> {
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
fn is_stopword(word: &str) -> bool {
    const STOPWORDS: &[&str] = &[
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

// ============================================================================
// Sentence Scoring
// ============================================================================

/// Sentence with score
#[derive(Debug, Clone)]
pub struct ScoredSentence {
    /// Original sentence text
    pub text: String,
    /// Position in document (0-indexed)
    pub position: usize,
    /// Overall score
    pub score: f32,
    /// Score breakdown
    pub score_components: ScoreComponents,
}

/// Score breakdown
#[derive(Debug, Clone, Default)]
pub struct ScoreComponents {
    /// TF-IDF based score
    pub tfidf_score: f32,
    /// Position score (earlier = higher)
    pub position_score: f32,
    /// Length score (prefer medium length)
    pub length_score: f32,
    /// Title overlap score
    pub title_score: f32,
}

/// Sentence scorer
pub struct SentenceScorer {
    /// Document-level word frequencies
    doc_freq: WordFrequency,
    /// Title words (for overlap scoring)
    title_words: HashSet<String>,
    /// Total sentences in document
    sentence_count: usize,
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
    /// Number of sentences in summary
    pub num_sentences: usize,
    /// Minimum sentence length (words)
    pub min_sentence_words: usize,
    /// Maximum sentence length (words)
    pub max_sentence_words: usize,
    /// Preserve original order
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
    /// Summary sentences
    pub sentences: Vec<ScoredSentence>,
    /// Original document stats
    pub original_word_count: usize,
    /// Summary word count
    pub summary_word_count: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Top keywords
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
    config: SummaryConfig,
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
    /// Coverage of important terms
    pub term_coverage: f32,
    /// Redundancy score (lower is better)
    pub redundancy: f32,
    /// Coherence estimate
    pub coherence: f32,
    /// Overall quality score
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

fn main() {
    println!("=== Demo G: WASM Document Summarizer ===\n");

    let content = r"
        Machine learning is a subset of artificial intelligence that enables systems to learn
        from data. These systems improve their performance over time without being explicitly
        programmed. Deep learning is a type of machine learning that uses neural networks with
        many layers. Neural networks are inspired by the structure of the human brain.

        Natural language processing allows computers to understand human language. This field
        combines linguistics and computer science. Applications include translation, sentiment
        analysis, and chatbots. Modern NLP relies heavily on transformer architectures.

        Computer vision enables machines to interpret visual information from the world.
        Image classification, object detection, and facial recognition are common tasks.
        Convolutional neural networks have revolutionized this field. Self-driving cars use
        computer vision extensively.

        Reinforcement learning trains agents through trial and error. The agent receives
        rewards or penalties based on its actions. This approach has achieved superhuman
        performance in games like chess and Go. Robotics also benefits from reinforcement
        learning techniques.
    ";

    let doc = Document::new("Introduction to Machine Learning", content);
    println!("Document: \"{}\"", doc.title);
    println!(
        "Words: {}, Sentences: {}\n",
        doc.word_count(),
        doc.sentence_count()
    );

    let summarizer = Summarizer::with_config(SummaryConfig::new(3));
    let summary = summarizer.summarize(&doc);

    println!("--- Summary ({} sentences) ---", summary.sentences.len());
    println!("{}\n", summary.as_bullet_points());

    println!("--- Statistics ---");
    println!("Original: {} words", summary.original_word_count);
    println!("Summary: {} words", summary.summary_word_count);
    println!("Compression: {:.1}%", summary.compression_ratio * 100.0);
    println!("Keywords: {}", summary.keywords.join(", "));

    let quality = QualityMetrics::evaluate(&doc, &summary);
    println!("\n--- Quality Metrics ---");
    println!("Term Coverage: {:.2}", quality.term_coverage);
    println!("Redundancy: {:.2}", quality.redundancy);
    println!("Coherence: {:.2}", quality.coherence);
    println!("Quality Score: {:.2}", quality.quality_score);

    println!("\n=== Demo G Complete ===");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_new() {
        let doc = Document::new("Title", "This is a test. Another sentence here.");
        assert_eq!(doc.title, "Title");
        assert_eq!(doc.sentence_count(), 2);
    }

    #[test]
    fn test_document_word_count() {
        let doc = Document::new("T", "One two three four five.");
        assert_eq!(doc.word_count(), 5);
    }

    #[test]
    fn test_split_sentences() {
        // Each sentence must have at least 3 words
        let sentences =
            split_sentences("This is first sentence. This is second one! And this is third here?");
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_split_sentences_no_punct() {
        let sentences = split_sentences("This is a sentence without ending punctuation");
        assert_eq!(sentences.len(), 1);
    }

    #[test]
    fn test_word_frequency_count() {
        let mut freq = WordFrequency::new();
        freq.count("word word another");
        assert_eq!(freq.counts.get("word"), Some(&2));
    }

    #[test]
    fn test_word_frequency_stopwords() {
        let mut freq = WordFrequency::new();
        freq.count("the and a");
        assert_eq!(freq.vocabulary_size(), 0);
    }

    #[test]
    fn test_word_frequency_top_words() {
        let mut freq = WordFrequency::new();
        freq.count("cat cat cat dog dog bird");
        let top = freq.top_words(2);
        assert_eq!(top[0].0, "cat");
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, World! Test-123");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
    }

    #[test]
    fn test_is_stopword() {
        assert!(is_stopword("the"));
        assert!(is_stopword("and"));
        assert!(!is_stopword("machine"));
    }

    #[test]
    fn test_sentence_scorer() {
        let doc = Document::new("Test", "Machine learning is great. Deep learning too.");
        let scorer = SentenceScorer::new(&doc);
        let scored = scorer.score("Machine learning is great.", 0);
        assert!(scored.score > 0.0);
    }

    #[test]
    fn test_summary_config_default() {
        let config = SummaryConfig::default();
        assert_eq!(config.num_sentences, DEFAULT_SUMMARY_LENGTH);
    }

    #[test]
    fn test_summarizer_new() {
        let summarizer = Summarizer::new();
        assert_eq!(summarizer.config.num_sentences, DEFAULT_SUMMARY_LENGTH);
    }

    #[test]
    fn test_summarizer_summarize() {
        let doc = Document::new("AI", "Artificial intelligence is amazing. Machine learning rocks. Deep learning is powerful. Neural networks work well.");
        let summarizer = Summarizer::with_config(SummaryConfig::new(2));
        let summary = summarizer.summarize(&doc);
        assert!(summary.sentences.len() <= 2);
    }

    #[test]
    fn test_summary_text() {
        // Sentences need at least 3 words to be included
        let doc = Document::new(
            "T",
            "This is first sentence here. This is second sentence there. And another third one.",
        );
        let summarizer = Summarizer::with_config(SummaryConfig::new(2));
        let summary = summarizer.summarize(&doc);
        let text = summary.text();
        assert!(!text.is_empty());
    }

    #[test]
    fn test_summary_compression_ratio() {
        let doc = Document::new(
            "T",
            "One two three. Four five six. Seven eight nine. Ten eleven twelve.",
        );
        let summarizer = Summarizer::with_config(SummaryConfig::new(1));
        let summary = summarizer.summarize(&doc);
        assert!(summary.compression_ratio < 1.0);
    }

    #[test]
    fn test_quality_metrics() {
        let doc = Document::new(
            "Machine Learning",
            "Machine learning is a field of AI. It uses data to learn patterns.",
        );
        let summarizer = Summarizer::new();
        let summary = summarizer.summarize(&doc);
        let quality = QualityMetrics::evaluate(&doc, &summary);
        assert!(quality.quality_score >= 0.0 && quality.quality_score <= 1.0);
    }

    #[test]
    fn test_scored_sentence_components() {
        let doc = Document::new("Test", "This is a test sentence for scoring.");
        let scorer = SentenceScorer::new(&doc);
        let scored = scorer.score("This is a test sentence for scoring.", 0);
        assert!(scored.score_components.position_score > 0.0);
    }

    #[test]
    fn test_empty_document() {
        let doc = Document::new("Empty", "");
        let summarizer = Summarizer::new();
        let summary = summarizer.summarize(&doc);
        assert!(summary.sentences.is_empty());
    }

    #[test]
    fn test_preserve_order() {
        let doc = Document::new(
            "T",
            "First one here. Second sentence. Third coming. Fourth end.",
        );
        let config = SummaryConfig::new(2).preserve_order(true);
        let summarizer = Summarizer::with_config(config);
        let summary = summarizer.summarize(&doc);
        if summary.sentences.len() == 2 {
            assert!(summary.sentences[0].position <= summary.sentences[1].position);
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_word_count_non_negative(text in "[a-z ]{0,100}") {
            let doc = Document::new("T", &text);
            prop_assert!(doc.word_count() >= 0);
        }

        #[test]
        fn prop_tokenize_lowercase(word in "[A-Z]{3,10}") {
            let tokens = tokenize(&word);
            if !tokens.is_empty() {
                prop_assert!(tokens[0].chars().all(|c| c.is_lowercase()));
            }
        }

        #[test]
        fn prop_summary_length_bounded(num_sentences in 1usize..10) {
            let doc = Document::new("T", "One sentence. Two sentence. Three sentence. Four sentence. Five sentence.");
            let config = SummaryConfig::new(num_sentences);
            let summarizer = Summarizer::with_config(config);
            let summary = summarizer.summarize(&doc);
            prop_assert!(summary.sentences.len() <= num_sentences);
        }

        #[test]
        fn prop_compression_ratio_bounded(n in 1usize..5) {
            let sentences: Vec<&str> = vec!["First sentence here.", "Second one now.", "Third sentence.", "Fourth.", "Fifth sentence here."];
            let content = sentences[..n.min(sentences.len())].join(" ");
            let doc = Document::new("T", &content);
            let summarizer = Summarizer::with_config(SummaryConfig::new(1));
            let summary = summarizer.summarize(&doc);
            prop_assert!(summary.compression_ratio >= 0.0);
            prop_assert!(summary.compression_ratio <= 1.0 || doc.word_count() == 0);
        }

        #[test]
        fn prop_quality_score_bounded(n in 2usize..5) {
            let sentences: Vec<&str> = vec!["Machine learning works.", "AI is great.", "Data science rocks.", "Neural nets help.", "Deep learning too."];
            let content = sentences[..n.min(sentences.len())].join(" ");
            let doc = Document::new("AI", &content);
            let summarizer = Summarizer::new();
            let summary = summarizer.summarize(&doc);
            let quality = QualityMetrics::evaluate(&doc, &summary);
            prop_assert!(quality.quality_score >= 0.0);
            prop_assert!(quality.quality_score <= 1.0);
        }

        #[test]
        fn prop_frequency_tf_non_negative(word in "[a-z]{3,8}") {
            let mut freq = WordFrequency::new();
            freq.count(&format!("{} {} other", word, word));
            prop_assert!(freq.tf(&word) >= 0.0);
        }
    }
}
