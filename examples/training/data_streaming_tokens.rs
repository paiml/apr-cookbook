//! # Recipe: Streaming Tokenization Pipeline
//!
//! **Category**: training
//! **CLI Equivalent**: `apr data tokenize --stream --chunk-size 64 --out tokens.bin`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example data_streaming_tokens` exits 0
//! 2. [x] `cargo test --example data_streaming_tokens` passes
//! 3. [x] Deterministic output (same seed -> same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr data` streaming tokenization in-process (no shell-out)
//! 10. [x] Unit tests cover chunk boundaries, EOF flush, vocab hits
//!
//! ## Learning Objective
//! Demonstrates a streaming tokenizer that reads a corpus chunk-by-chunk without
//! loading it fully into memory, emitting integer token IDs to an output stream.
//! Includes a fixed-size rolling buffer that handles partial-token boundaries
//! across chunk reads.
//!
//! ## Run Command
//! ```bash
//! cargo run --example data_streaming_tokens
//! ```
//!
//! ## References
//! - Dean, J. & Ghemawat, S. (2008). *MapReduce: Simplified Data Processing on Large Clusters*. CACM. DOI: 10.1145/1327452.1327492

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;
use std::collections::HashMap;
use std::io::{Read, Write};

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Vocab {
    map: HashMap<String, u32>,
    unk_id: u32,
}

impl Vocab {
    fn new() -> Self {
        let mut m = HashMap::new();
        for (i, w) in [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "a", "and", "of", "to",
            "in", "is",
        ]
        .iter()
        .enumerate()
        {
            m.insert((*w).to_string(), i as u32 + 1);
        }
        Self { map: m, unk_id: 0 }
    }
    fn encode(&self, w: &str) -> u32 {
        self.map.get(w).copied().unwrap_or(self.unk_id)
    }
}

#[derive(Debug, Clone, Default)]
struct StreamStats {
    chunks_read: usize,
    bytes_read: usize,
    tokens_emitted: usize,
    unk_tokens: usize,
}

// ---------------------------------------------------------------------------
// Streaming tokenizer
// ---------------------------------------------------------------------------

/// Stream-tokenize input in fixed-size chunks. Partial words at the tail of a
/// chunk are carried into the next chunk via a leftover buffer.
fn stream_tokenize(
    input: &mut dyn Read,
    output: &mut dyn Write,
    vocab: &Vocab,
    chunk_size: usize,
) -> std::io::Result<StreamStats> {
    let mut stats = StreamStats::default();
    let mut buf = vec![0_u8; chunk_size];
    let mut carry: Vec<u8> = Vec::new();

    loop {
        let n = input.read(&mut buf)?;
        if n == 0 {
            // Flush any carry as a final word.
            if !carry.is_empty() {
                let word = String::from_utf8_lossy(&carry).to_string();
                if !word.trim().is_empty() {
                    let id = vocab.encode(word.trim());
                    if id == vocab.unk_id {
                        stats.unk_tokens += 1;
                    }
                    output.write_all(&id.to_le_bytes())?;
                    stats.tokens_emitted += 1;
                }
                carry.clear();
            }
            break;
        }
        stats.chunks_read += 1;
        stats.bytes_read += n;

        // Find the last whitespace in the new chunk so we can safely split.
        let slice = &buf[..n];
        let mut combined = Vec::with_capacity(carry.len() + slice.len());
        combined.extend_from_slice(&carry);
        combined.extend_from_slice(slice);

        // Find last whitespace to avoid splitting a token. If there is no
        // whitespace at all, keep everything in the carry for the next chunk.
        let last_ws = combined.iter().rposition(u8::is_ascii_whitespace);
        let split_at = match last_ws {
            Some(p) => p + 1,
            None => 0, // nothing emittable yet — carry it all forward
        };

        let (head, tail) = combined.split_at(split_at);
        let head_str = String::from_utf8_lossy(head);
        for tok in head_str.split_ascii_whitespace() {
            let id = vocab.encode(tok);
            if id == vocab.unk_id {
                stats.unk_tokens += 1;
            }
            output.write_all(&id.to_le_bytes())?;
            stats.tokens_emitted += 1;
        }
        carry.clear();
        carry.extend_from_slice(tail);
    }

    Ok(stats)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("data_streaming_tokens")?;
    println!("=== Recipe: {} ===", ctx.name());

    // Build a deterministic corpus.
    let corpus =
        "the quick brown fox jumps over the lazy dog and the dog is a dog the fox is quick "
            .repeat(4);
    let corpus_path = ctx.path("corpus.txt");
    std::fs::write(&corpus_path, &corpus)?;
    println!(
        "Corpus: {} bytes at {}",
        corpus.len(),
        corpus_path.display()
    );

    let vocab = Vocab::new();
    let chunk_size = 32;

    let tokens_path = ctx.path("tokens.bin");
    let mut input = std::fs::File::open(&corpus_path)?;
    let mut output = std::fs::File::create(&tokens_path)?;
    let stats = stream_tokenize(&mut input, &mut output, &vocab, chunk_size)?;

    println!("\n--- Streaming Tokenization ---");
    println!("Chunk size:      {chunk_size} bytes");
    println!("Chunks read:     {}", stats.chunks_read);
    println!("Bytes read:      {}", stats.bytes_read);
    println!("Tokens emitted:  {}", stats.tokens_emitted);
    println!("UNK tokens:      {}", stats.unk_tokens);

    let token_file_len = std::fs::metadata(&tokens_path)?.len() as usize;
    assert_eq!(
        token_file_len,
        stats.tokens_emitted * 4,
        "each token should be a u32 LE"
    );

    let out = json!({
        "recipe": ctx.name(),
        "corpus_bytes": corpus.len(),
        "chunk_size": chunk_size,
        "chunks_read": stats.chunks_read,
        "tokens_emitted": stats.tokens_emitted,
        "unk_tokens": stats.unk_tokens,
    });
    let out_path = ctx.path("stream-stats.json");
    let out_bytes =
        serde_json::to_vec_pretty(&out).map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out_path, out_bytes)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_simple_sentence() {
        let vocab = Vocab::new();
        let text = "the quick brown fox";
        let mut input = Cursor::new(text.as_bytes().to_vec());
        let mut out: Vec<u8> = Vec::new();
        let s = stream_tokenize(&mut input, &mut out, &vocab, 4).expect("tok");
        assert_eq!(s.tokens_emitted, 4);
        assert_eq!(out.len(), 16);
    }

    #[test]
    fn test_chunk_boundary_preserves_tokens() {
        let vocab = Vocab::new();
        // Chunk size 3 forces splits inside tokens.
        let text = "the quick brown fox jumps";
        let mut input = Cursor::new(text.as_bytes().to_vec());
        let mut out: Vec<u8> = Vec::new();
        let s = stream_tokenize(&mut input, &mut out, &vocab, 3).expect("tok");
        assert_eq!(s.tokens_emitted, 5);
    }

    #[test]
    fn test_unk_tokens_counted() {
        let vocab = Vocab::new();
        let text = "foobar wiggle the";
        let mut input = Cursor::new(text.as_bytes().to_vec());
        let mut out: Vec<u8> = Vec::new();
        let s = stream_tokenize(&mut input, &mut out, &vocab, 32).expect("tok");
        assert_eq!(s.tokens_emitted, 3);
        assert_eq!(s.unk_tokens, 2);
    }

    #[test]
    fn test_empty_input_emits_no_tokens() {
        let vocab = Vocab::new();
        let mut input = Cursor::new(Vec::new());
        let mut out: Vec<u8> = Vec::new();
        let s = stream_tokenize(&mut input, &mut out, &vocab, 8).expect("tok");
        assert_eq!(s.tokens_emitted, 0);
        assert_eq!(s.bytes_read, 0);
    }

    #[test]
    fn test_trailing_token_flushed_at_eof() {
        let vocab = Vocab::new();
        // No trailing whitespace: should still flush "fox" at EOF.
        let text = "fox";
        let mut input = Cursor::new(text.as_bytes().to_vec());
        let mut out: Vec<u8> = Vec::new();
        let s = stream_tokenize(&mut input, &mut out, &vocab, 16).expect("tok");
        assert_eq!(s.tokens_emitted, 1);
    }
}
