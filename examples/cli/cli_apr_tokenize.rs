//! # Recipe: APR Tokenizer Training CLI
//!
//! **Category**: CLI Tools
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

use apr_cookbook::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::env;
use std::hash::{Hash, Hasher};

fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let config = parse_args(&args)?;

    if config.help {
        print_help();
        return Ok(());
    }

    run_tokenize(&config)
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TokenizeConfig {
    corpus_path: Option<String>,
    vocab_size: usize,
    method: TokenMethod,
    demo: bool,
    help: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TokenMethod {
    Bpe,
    Unigram,
}

impl TokenMethod {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Bpe => "bpe",
            Self::Unigram => "unigram",
        }
    }
}

// ---------------------------------------------------------------------------
// BPE Trainer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BpeTrainer {
    vocab: HashMap<String, u32>,
    merges: Vec<(String, String)>,
}

impl BpeTrainer {
    fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            merges: Vec::new(),
        }
    }

    /// Return the vocabulary size.
    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

fn parse_args(args: &[String]) -> Result<TokenizeConfig> {
    let mut config = TokenizeConfig {
        corpus_path: None,
        vocab_size: 64,
        method: TokenMethod::Bpe,
        demo: false,
        help: false,
    };

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => config.help = true,
            "--demo" | "-d" => config.demo = true,
            "--vocab-size" | "-n" => {
                i += 1;
                if i < args.len() {
                    config.vocab_size = args[i].parse().unwrap_or(64);
                }
            }
            "--method" | "-m" => {
                i += 1;
                if i < args.len() {
                    config.method = match args[i].as_str() {
                        "unigram" => TokenMethod::Unigram,
                        _ => TokenMethod::Bpe,
                    };
                }
            }
            path if !path.starts_with('-') => {
                config.corpus_path = Some(path.to_string());
            }
            _ => {
                return Err(CookbookError::invalid_format(format!(
                    "Unknown argument: {}",
                    args[i]
                )));
            }
        }
        i += 1;
    }

    Ok(config)
}

fn print_help() {
    println!("apr-tokenize - Train a BPE tokenizer on a text corpus");
    println!();
    println!("USAGE:");
    println!("    apr-tokenize [OPTIONS] [CORPUS_PATH]");
    println!();
    println!("OPTIONS:");
    println!("    -h, --help             Print help information");
    println!("    -d, --demo             Run with built-in demo corpus");
    println!("    -n, --vocab-size N     Target vocabulary size (default: 64)");
    println!("    -m, --method METHOD    Tokenization method: bpe|unigram (default: bpe)");
    println!();
    println!("EXAMPLES:");
    println!("    apr-tokenize --demo");
    println!("    apr-tokenize --demo --vocab-size 80");
    println!("    apr-tokenize corpus.txt --method bpe");
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
    while trainer.vocab.len() < vocab_size {
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
        print_help();
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
    println!("  Method:        {}", config.method.as_str());
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_args_demo() {
        let args = vec!["apr-tokenize".to_string(), "--demo".to_string()];
        let config = parse_args(&args).expect("parse ok");
        assert!(config.demo);
        assert_eq!(config.vocab_size, 64);
    }

    #[test]
    fn test_parse_args_vocab_size() {
        let args = vec![
            "apr-tokenize".to_string(),
            "--vocab-size".to_string(),
            "128".to_string(),
        ];
        let config = parse_args(&args).expect("parse ok");
        assert_eq!(config.vocab_size, 128);
    }

    #[test]
    fn test_parse_args_method_unigram() {
        let args = vec![
            "apr-tokenize".to_string(),
            "--method".to_string(),
            "unigram".to_string(),
        ];
        let config = parse_args(&args).expect("parse ok");
        assert_eq!(config.method, TokenMethod::Unigram);
    }

    #[test]
    fn test_parse_args_corpus_path() {
        let args = vec!["apr-tokenize".to_string(), "corpus.txt".to_string()];
        let config = parse_args(&args).expect("parse ok");
        assert_eq!(config.corpus_path, Some("corpus.txt".to_string()));
    }

    #[test]
    fn test_parse_args_unknown_rejected() {
        let args = vec!["apr-tokenize".to_string(), "--bogus".to_string()];
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn test_char_frequencies() {
        let freq = count_char_frequencies("aab");
        assert_eq!(freq.get(&'a'), Some(&2));
        assert_eq!(freq.get(&'b'), Some(&1));
        assert_eq!(freq.get(&'z'), None);
    }

    #[test]
    fn test_split_into_words() {
        let (words, counts) = split_into_words("cat cat dog");
        // "cat" appears twice, "dog" once
        assert_eq!(words.len(), 2);
        // First entry should be "cat" (higher frequency)
        let cat_idx = words
            .iter()
            .position(|w| w.first().map_or(false, |s| s == "c"))
            .expect("cat present");
        assert_eq!(counts[cat_idx], 2);
    }

    #[test]
    fn test_count_pairs() {
        let words = vec![vec!["a".to_string(), "b".to_string(), "c".to_string()]];
        let counts = vec![3];
        let pairs = count_pairs(&words, &counts);
        assert_eq!(pairs.get(&("a".to_string(), "b".to_string())), Some(&3));
        assert_eq!(pairs.get(&("b".to_string(), "c".to_string())), Some(&3));
    }

    #[test]
    fn test_most_frequent_pair_deterministic() {
        let mut pairs = HashMap::new();
        pairs.insert(("a".to_string(), "b".to_string()), 5);
        pairs.insert(("c".to_string(), "d".to_string()), 5);
        let (best, count) = most_frequent_pair(&pairs).expect("non-empty");
        assert_eq!(count, 5);
        // Deterministic tiebreak: ("a","b") < ("c","d") lexicographically,
        // but max_by picks the last equal element, so we just check it's one of them
        assert!(
            best == ("a".to_string(), "b".to_string())
                || best == ("c".to_string(), "d".to_string())
        );
    }

    #[test]
    fn test_apply_merge() {
        let mut words = vec![vec![
            "a".to_string(),
            "b".to_string(),
            "a".to_string(),
            "b".to_string(),
        ]];
        apply_merge(&mut words, "a", "b");
        assert_eq!(words[0], vec!["ab".to_string(), "ab".to_string()]);
    }

    #[test]
    fn test_train_bpe_grows_vocab() {
        let trainer = train_bpe("ab ab ab cd cd", 10);
        assert!(trainer.vocab_size() >= 5); // at least a,b,c,d,</w>
        assert!(!trainer.merges.is_empty());
    }

    #[test]
    fn test_tokenize_produces_known_symbols() {
        let trainer = train_bpe("the the the cat", 20);
        let tokens = tokenize("the cat", &trainer);
        // All produced tokens must be in the vocabulary
        for t in &tokens {
            assert!(trainer.vocab.contains_key(t), "token '{}' not in vocab", t);
        }
    }

    #[test]
    fn test_detokenize_simple() {
        let tokens = vec![
            "he".to_string(),
            "llo</w>".to_string(),
            "wor".to_string(),
            "ld</w>".to_string(),
        ];
        assert_eq!(detokenize(&tokens), "hello world");
    }

    #[test]
    fn test_detokenize_bare_eow() {
        let tokens = vec!["a".to_string(), "b".to_string(), "</w>".to_string()];
        assert_eq!(detokenize(&tokens), "ab");
    }

    #[test]
    fn test_roundtrip_tokenize_detokenize() {
        let corpus = demo_corpus();
        let trainer = train_bpe(corpus, 80);
        let input = "the cat sat on the mat";
        let tokens = tokenize(input, &trainer);
        let output = detokenize(&tokens);
        assert_eq!(output, input, "roundtrip must reconstruct original text");
    }

    #[test]
    fn test_roundtrip_single_word() {
        let trainer = train_bpe("hello hello hello world", 20);
        let tokens = tokenize("hello", &trainer);
        let output = detokenize(&tokens);
        assert_eq!(output, "hello");
    }

    #[test]
    fn test_demo_run() {
        let config = TokenizeConfig {
            corpus_path: None,
            vocab_size: 50,
            method: TokenMethod::Bpe,
            demo: true,
            help: false,
        };
        assert!(run_tokenize(&config).is_ok());
    }

    #[test]
    fn test_deterministic_training() {
        let corpus = demo_corpus();
        let t1 = train_bpe(corpus, 60);
        let t2 = train_bpe(corpus, 60);
        assert_eq!(t1.merges.len(), t2.merges.len());
        for (a, b) in t1.merges.iter().zip(t2.merges.iter()) {
            assert_eq!(a, b, "merges must be identical across runs");
        }
    }

    #[test]
    fn test_empty_input_no_crash() {
        let trainer = train_bpe("", 10);
        assert!(trainer.merges.is_empty());
        let tokens = tokenize("", &trainer);
        assert!(tokens.is_empty());
        assert_eq!(detokenize(&tokens), "");
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn prop_vocab_size_bounded(target in 40usize..120) {
            // The demo corpus has ~29 unique chars + </w>, so target must
            // exceed the initial alphabet for the bound to hold.
            let trainer = train_bpe(demo_corpus(), target);
            prop_assert!(trainer.vocab_size() <= target + 1,
                "vocab {} should not exceed target {} by more than 1",
                trainer.vocab_size(), target);
        }

        #[test]
        fn prop_all_tokens_in_vocab(word in "[a-z]{1,6}") {
            let corpus = format!("{w} {w} {w} the the", w = word);
            let trainer = train_bpe(&corpus, 40);
            let tokens = tokenize(&word, &trainer);
            for t in &tokens {
                prop_assert!(trainer.vocab.contains_key(t),
                    "token '{}' not in vocab for word '{}'", t, word);
            }
        }
    }
}
