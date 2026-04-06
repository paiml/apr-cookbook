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
