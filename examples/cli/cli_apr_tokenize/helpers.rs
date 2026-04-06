#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use aprender::demo::reliable::AdaptiveOutput;
use clap::Parser;
use proptest::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Parser)]
#[command(
    name = "apr-tokenize",
    about = "Train a BPE tokenizer on a text corpus"
)]
pub struct TokenizeConfig {
    // Path to the text corpus file
    pub corpus_path: Option<String>,

    /// Target vocabulary size
    #[arg(short = 'n', long = "vocab-size", default_value_t = 64)]
    pub vocab_size: usize,

    /// Tokenization method: bpe|unigram
    #[arg(short, long, default_value = "bpe")]
    pub method: String,

    /// Run with built-in demo corpus
    #[arg(long, short = 'd')]
    pub demo: bool,
}

impl TokenizeConfig {
    pub fn token_method(&self) -> TokenMethod {
        match self.method.as_str() {
            "unigram" => TokenMethod::Unigram,
            _ => TokenMethod::Bpe,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenMethod {
    Bpe,
    Unigram,
}

impl TokenMethod {
    pub fn as_str(&self) -> &'static str {
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
pub struct BpeTrainer {
    pub vocab: HashMap<String, u32>,
    pub merges: Vec<(String, String)>,
}

impl BpeTrainer {
    pub fn new() -> Self {
        Self {
            vocab: HashMap::new(),
            merges: Vec::new(),
        }
    }

    /// Return the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ---------------------------------------------------------------------------
// Argument parsing (test helper)
// ---------------------------------------------------------------------------
