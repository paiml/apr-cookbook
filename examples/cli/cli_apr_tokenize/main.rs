#![allow(unused_imports)]
//! # Recipe: APR Tokenizer Training CLI
//!
//! **Category**: CLI Tools
//! **CLI Equivalent**: `apr tokenize`
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/cli-parity-v1.yaml
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Demonstrate the `apr tokenize` workflow: BPE tokenizer training pipeline.
//! Trains a byte-pair encoding vocabulary from a corpus, iteratively merging
//! the most frequent adjacent symbol pairs until `vocab_size` is reached.
//!
//! ## Run Command
//! ```bash
//! cargo run --example cli_apr_tokenize
//! cargo run --example cli_apr_tokenize -- --demo
//! cargo run --example cli_apr_tokenize -- --demo --vocab-size 80
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr inspect model.apr          # APR native format
//! apr inspect model.gguf         # GGUF (llama.cpp compatible)
//! apr inspect model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Amershi, S. et al. (2019). *Software Engineering for Machine Learning: A Case Study*. ICSE. DOI: 10.1109/ICSE-SEIP.2019.00042

use apr_cookbook::prelude::*;
use aprender::demo::reliable::AdaptiveOutput;
use clap::Parser;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

fn main() -> Result<()> {
    let config = TokenizeConfig::parse();
    run_tokenize(&config)
}

mod types;
#[allow(unused_imports, clippy::wildcard_imports)]
use types::*;

mod helpers;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;

mod tests;

#[cfg(test)]
fn parse_args(args: &[String]) -> std::result::Result<TokenizeConfig, clap::Error> {
    TokenizeConfig::try_parse_from(args)
}

// ---------------------------------------------------------------------------
// Demo corpus
// ---------------------------------------------------------------------------

fn demo_corpus() -> &'static str {
    concat!(
        "the cat sat on the mat and the dog sat on the log. ",
        "a fat cat sat on a flat mat while the rat ran past. ",
        "the quick brown fox jumps over the lazy dog every day. ",
        "machine learning models transform data into predictions. ",
        "tokenization splits text into smaller meaningful units. ",
        "byte pair encoding merges frequent character pairs iteratively. ",
        "the model was trained on a large corpus of english text. ",
        "neural networks learn representations from raw input data. ",
    )
}

// ---------------------------------------------------------------------------
// Character frequency counting
// ---------------------------------------------------------------------------

/// Count character frequencies across all words in the corpus.
/// Returns a map from character to its total occurrence count.
fn count_char_frequencies(text: &str) -> HashMap<char, u64> {
    let mut freq = HashMap::new();
    for ch in text.chars() {
        *freq.entry(ch).or_insert(0) += 1;
    }
    freq
}

/// Split the corpus into word-level token sequences, where each word is
/// represented as a vector of single-character strings with a trailing
/// end-of-word marker ("</w>"). Returns (word_tokens, word_counts).
fn split_into_words(text: &str) -> (Vec<Vec<String>>, Vec<u64>) {
    let mut word_freq: HashMap<String, u64> = HashMap::new();
    for word in text.split_whitespace() {
        *word_freq.entry(word.to_string()).or_insert(0) += 1;
    }

    // Sort deterministically using a seeded order for reproducibility
    let mut entries: Vec<(String, u64)> = word_freq.into_iter().collect();
    entries.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then_with(|| deterministic_str_cmp(&a.0, &b.0))
    });

    let mut words = Vec::new();
    let mut counts = Vec::new();
    for (word, count) in entries {
        let mut chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        chars.push("</w>".to_string());
        words.push(chars);
        counts.push(count);
    }

    (words, counts)
}

/// Deterministic string comparison using hash-based tiebreaking.
fn deterministic_str_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    a.cmp(b)
}

// ---------------------------------------------------------------------------
// Pair counting and merging
// ---------------------------------------------------------------------------

/// Count adjacent pair frequencies across all words weighted by word count.
fn count_pairs(words: &[Vec<String>], counts: &[u64]) -> HashMap<(String, String), u64> {
    let mut pairs = HashMap::new();
    for (word, &count) in words.iter().zip(counts.iter()) {
        if word.len() < 2 {
            continue;
        }
        for j in 0..word.len() - 1 {
            let pair = (word[j].clone(), word[j + 1].clone());
            *pairs.entry(pair).or_insert(0) += count;
        }
    }
    pairs
}

/// Find the most frequent pair. Ties are broken deterministically by
/// lexicographic order of the pair components.
fn most_frequent_pair(pairs: &HashMap<(String, String), u64>) -> Option<((String, String), u64)> {
    pairs
        .iter()
        .max_by(|a, b| {
            a.1.cmp(b.1)
                .then_with(|| a.0 .0.cmp(&b.0 .0))
                .then_with(|| a.0 .1.cmp(&b.0 .1))
        })
        .map(|(pair, &count)| (pair.clone(), count))
}

/// Apply a merge: replace every adjacent occurrence of (left, right) with
/// the concatenated symbol "left+right" in all words.
fn apply_merge(words: &mut [Vec<String>], left: &str, right: &str) {
    let merged = format!("{}{}", left, right);
    for word in words.iter_mut() {
        let mut i = 0;
        while i + 1 < word.len() {
            if word[i] == left && word[i + 1] == right {
                word[i] = merged.clone();
                word.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// BPE training
// ---------------------------------------------------------------------------

/// Train a BPE tokenizer from raw text, iterating merges until `vocab_size`
/// distinct symbols are reached.
fn train_bpe(text: &str, vocab_size: usize) -> BpeTrainer {
    let output = AdaptiveOutput::new();
    let mut trainer = BpeTrainer::new();
    let (mut words, counts) = split_into_words(text);

    // Seed the initial vocabulary with individual characters + </w>
    let char_freq = count_char_frequencies(text);
    let mut initial_chars: Vec<(char, u64)> = char_freq.into_iter().collect();
    initial_chars.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut next_id: u32 = 0;
    for (ch, _) in &initial_chars {
        trainer.vocab.entry(ch.to_string()).or_insert_with(|| {
            let id = next_id;
            next_id += 1;
            id
        });
    }
    trainer.vocab.entry("</w>".to_string()).or_insert_with(|| {
        let id = next_id;
        next_id += 1;
        id
    });

    // Iteratively merge the most frequent pair
    let merges_needed = vocab_size.saturating_sub(trainer.vocab.len());
    while trainer.vocab.len() < vocab_size {
        output.progress(trainer.merges.len() + 1, merges_needed, "BPE merge");
        let pairs = count_pairs(&words, &counts);
        let best = most_frequent_pair(&pairs);
        match best {
            Some(((left, right), _count)) => {
                let merged = format!("{}{}", left, right);
                apply_merge(&mut words, &left, &right);
                trainer.merges.push((left, right));
                trainer.vocab.entry(merged).or_insert_with(|| {
                    let id = next_id;
                    next_id += 1;
                    id
                });
            }
            None => break, // no more pairs to merge
        }
    }
    output.status(""); // clear progress line

    trainer
}

// ---------------------------------------------------------------------------
// Tokenization (apply learned merges)
// ---------------------------------------------------------------------------

/// Tokenize a string by applying the learned BPE merges in order.
fn tokenize(text: &str, trainer: &BpeTrainer) -> Vec<String> {
    let mut result = Vec::new();
    for word in text.split_whitespace() {
        let mut symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        symbols.push("</w>".to_string());

        // Re-apply each merge in training order
        for (left, right) in &trainer.merges {
            let merged = format!("{}{}", left, right);
            let mut i = 0;
            while i + 1 < symbols.len() {
                if symbols[i] == *left && symbols[i + 1] == *right {
                    symbols[i] = merged.clone();
                    symbols.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        result.extend(symbols);
    }
    result
}

/// Reconstruct text from BPE tokens. Removes end-of-word markers and
/// inserts spaces between words.
fn detokenize(tokens: &[String]) -> String {
    let mut result = String::new();
    for token in tokens {
        if token == "</w>" {
            result.push(' ');
        } else if token.ends_with("</w>") {
            result.push_str(&token[..token.len() - 4]);
            result.push(' ');
        } else {
            result.push_str(token);
        }
    }

    // Trim trailing space
    let trimmed = result.trim_end().to_string();
    trimmed
}

// ---------------------------------------------------------------------------
// Deterministic helpers
// ---------------------------------------------------------------------------

fn deterministic_seed(name: &str) -> u64 {
    let mut h = DefaultHasher::new();
    name.hash(&mut h);
    h.finish()
}

// ---------------------------------------------------------------------------
// Main driver
// ---------------------------------------------------------------------------

fn run_tokenize(config: &TokenizeConfig) -> Result<()> {
    let mut ctx = RecipeContext::new("cli_apr_tokenize")?;

    let corpus = if config.demo {
        demo_corpus().to_string()
    } else if let Some(path) = &config.corpus_path {
        std::fs::read_to_string(path).map_err(|e| {
            CookbookError::invalid_format(format!("Failed to read corpus {}: {}", path, e))
        })?
    } else {
        println!("No corpus provided. Use --demo or specify a corpus path.");
        return Ok(());
    };

    println!("APR Tokenize Pipeline");
    println!("=====================");
    println!();

    // Corpus stats
    let word_count = corpus.split_whitespace().count();
    let char_count = corpus.len();
    let unique_chars = {
        let mut chars: Vec<char> = corpus.chars().collect();
        chars.sort_unstable();
        chars.dedup();
        chars.len()
    };

    println!("Corpus Statistics:");
    println!("  Characters:    {}", char_count);
    println!("  Words:         {}", word_count);
    println!("  Unique chars:  {}", unique_chars);
    println!("  Method:        {}", config.token_method().as_str());
    println!("  Target vocab:  {}", config.vocab_size);
    println!();

    // Train
    let trainer = train_bpe(&corpus, config.vocab_size);

    // Merge history
    let merge_display = trainer.merges.len().min(20);
    println!(
        "Merge History (top {} of {}):",
        merge_display,
        trainer.merges.len()
    );
    println!(
        "  {:<5} {:<15} {:<15} {:<20}",
        "Step", "Left", "Right", "Result"
    );
    println!("  {:-<58}", "");
    for (i, (left, right)) in trainer.merges.iter().take(merge_display).enumerate() {
        let merged = format!("{}{}", left, right);
        println!(
            "  {:<5} {:<15} {:<15} {:<20}",
            i + 1,
            format!("\"{}\"", left),
            format!("\"{}\"", right),
            format!("\"{}\"", merged),
        );
    }
    println!();

    // Final vocabulary
    let mut sorted_vocab: Vec<(&String, &u32)> = trainer.vocab.iter().collect();
    sorted_vocab.sort_by(|a, b| a.1.cmp(b.1));

    println!("Final Vocabulary ({} tokens):", trainer.vocab_size());
    println!("  {:<6} {:<30}", "ID", "Token");
    println!("  {:-<38}", "");
    for (token, id) in sorted_vocab.iter().take(30) {
        println!("  {:<6} \"{}\"", id, token);
    }
    if trainer.vocab_size() > 30 {
        println!("  ... ({} more tokens)", trainer.vocab_size() - 30);
    }
    println!();

    // Example tokenization
    let example = if config.demo {
        "the cat sat on the mat"
    } else {
        corpus.split('\n').next().unwrap_or("hello world")
    };
    let example_trimmed = if example.len() > 60 {
        &example[..60]
    } else {
        example
    };

    let tokens = tokenize(example_trimmed, &trainer);
    let roundtrip = detokenize(&tokens);

    println!("Example Tokenization:");
    println!("  Input:      \"{}\"", example_trimmed);
    println!("  Tokens ({}): {:?}", tokens.len(), tokens);
    println!("  Roundtrip:  \"{}\"", roundtrip);
    println!(
        "  Match:      {}",
        if roundtrip == example_trimmed {
            "YES"
        } else {
            "NO"
        }
    );
    println!();

    // Compression ratio
    let original_chars = example_trimmed.split_whitespace().count()
        + example_trimmed
            .split_whitespace()
            .map(str::len)
            .sum::<usize>();
    let token_count = tokens.len();
    let compression = if token_count > 0 {
        original_chars as f64 / token_count as f64
    } else {
        0.0
    };
    println!(
        "Compression: {:.2}x (orig symbols: {}, BPE tokens: {})",
        compression, original_chars, token_count
    );

    // Deterministic seed for reproducibility verification
    let seed = deterministic_seed("cli_apr_tokenize");
    ctx.record_metric("corpus_chars", char_count as i64);
    ctx.record_metric("corpus_words", word_count as i64);
    ctx.record_metric("vocab_size", trainer.vocab_size() as i64);
    ctx.record_metric("merge_count", trainer.merges.len() as i64);
    ctx.record_metric("seed", seed as i64);

    Ok(())
}
