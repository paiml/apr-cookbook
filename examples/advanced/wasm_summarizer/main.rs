#![allow(unused_imports)]
//! # Demo G: WASM Document Summarizer
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Client-side document summarization using extractive methods.
//! Designed for WASM deployment with minimal dependencies.
//!
//! ## Toyota Way Principles
//!
//! - **Heijunka**: Consistent summary quality regardless of document size
//! - **Jidoka**: Automatic quality detection stops poor summaries
//! - **Kaizen**: Iterative sentence selection refinement
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Haas, A. et al. (2017). *Bringing the Web up to Speed with WebAssembly*. PLDI. DOI: 10.1145/3062341.3062363

use std::collections::{HashMap, HashSet};

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
    #[allow(unused_imports, clippy::wildcard_imports)]
    use super::helpers::*;
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
    #[allow(unused_imports, clippy::wildcard_imports)]
    use super::helpers::*;
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_word_count_non_negative(text in "[a-z ]{0,100}") {
            let doc = Document::new("T", &text);
            let _ = doc.word_count();
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
