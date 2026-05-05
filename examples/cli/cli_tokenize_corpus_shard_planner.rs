//! # apr tokenize encode-corpus — Shard Planner
//!
//! `apr tokenize encode-corpus <CORPUS> --shard-size <BYTES>` plans the
//! .bin shard layout for the pretokenized output. Shards must (a) be
//! ≤ shard_size bytes, (b) end on a token boundary (not mid-token), and
//! (c) be numbered with a deterministic 5-digit suffix
//! (`corpus.00000.bin`, `corpus.00001.bin`, …). This recipe builds the
//! planner and asserts the contract.
//!
//! Demonstrates the **TOKENIZE.5** recipe for PMAT-095 (apr tokenize coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender contracts/pretokenize-bin-v1.yaml
//!
//! Run with: cargo run --example cli_tokenize_corpus_shard_planner
//!
//! Added by PMAT-095 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardPlan {
    pub name: String,
    pub start_token: u64,
    pub end_token: u64, // exclusive
    pub byte_size: u64,
}

const TOKEN_BYTES: u64 = 4; // u32 token IDs

pub fn plan_shards(total_tokens: u64, shard_size_bytes: u64) -> Vec<ShardPlan> {
    if total_tokens == 0 || shard_size_bytes == 0 {
        return Vec::new();
    }
    let tokens_per_shard = (shard_size_bytes / TOKEN_BYTES).max(1);
    let mut out = Vec::new();
    let mut start = 0u64;
    let mut idx = 0u32;
    while start < total_tokens {
        let end = (start + tokens_per_shard).min(total_tokens);
        let count = end - start;
        out.push(ShardPlan {
            name: format!("corpus.{idx:05}.bin"),
            start_token: start,
            end_token: end,
            byte_size: count * TOKEN_BYTES,
        });
        start = end;
        idx += 1;
    }
    out
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_tokenize_corpus_shard_planner")?;

    println!("=== 1M tokens, 1MB shard ===");
    for s in plan_shards(1_000_000, 1_000_000) {
        println!(
            "  {}  tokens [{}..{})  bytes={}",
            s.name, s.start_token, s.end_token, s.byte_size
        );
    }

    println!("\n=== 100 tokens, 1MB shard (single shard) ===");
    for s in plan_shards(100, 1_000_000) {
        println!(
            "  {}  tokens [{}..{})  bytes={}",
            s.name, s.start_token, s.end_token, s.byte_size
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn empty_corpus_produces_no_shards() {
        assert!(plan_shards(0, 1_000_000).is_empty());
    }

    #[test]
    fn zero_shard_size_produces_no_shards() {
        // Refuse to plan with a degenerate shard size — caller bug.
        assert!(plan_shards(1_000_000, 0).is_empty());
    }

    #[test]
    fn small_corpus_yields_single_shard() {
        let s = plan_shards(100, 1_000_000);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].name, "corpus.00000.bin");
        assert_eq!(s[0].start_token, 0);
        assert_eq!(s[0].end_token, 100);
    }

    #[test]
    fn shards_align_on_token_boundaries() {
        // Every byte_size must be divisible by TOKEN_BYTES — never split mid-token.
        let s = plan_shards(1_000_000, 999_999);
        for p in &s {
            assert_eq!(p.byte_size % TOKEN_BYTES, 0, "split mid-token: {p:?}");
        }
    }

    #[test]
    fn shards_cover_corpus_exactly_once() {
        // Sum of (end - start) must equal total_tokens, and shards must be
        // contiguous with no gaps and no overlaps.
        let total = 1_234_567u64;
        let s = plan_shards(total, 16_384);
        let sum: u64 = s.iter().map(|p| p.end_token - p.start_token).sum();
        assert_eq!(sum, total);
        for w in s.windows(2) {
            assert_eq!(w[0].end_token, w[1].start_token);
        }
    }

    #[test]
    fn shard_names_use_5_digit_suffix() {
        // Sortable filenames matter for downstream consumers (shuffle list).
        let s = plan_shards(10_000, 100);
        assert!(s.iter().all(|p| {
            let suffix = &p.name["corpus.".len()..];
            suffix.starts_with(|c: char| c.is_ascii_digit())
                && suffix.split('.').next().unwrap().len() == 5
        }));
    }

    #[test]
    fn first_shard_is_indexed_zero() {
        let s = plan_shards(1000, 100);
        assert_eq!(s[0].name, "corpus.00000.bin");
    }
}
