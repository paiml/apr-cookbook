//! # Creation Vocabulary Size Validator
//!
//! Vocab size constraints: must be ≥ 256 (cover bytes); should be a
//! multiple of 64 for SIMD-friendly logits matmul; many tokenizers
//! reserve 256 byte tokens + special tokens; common picks: 32_000
//! (Llama), 50_257 (GPT-2), 128_256 (Llama-3). This recipe validates +
//! suggests the next-multiple alignment.
//!
//! Demonstrates the **CREATE.8** recipe for PMAT-127 (creation coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Kudo & Richardson (2018). SentencePiece. arXiv:1808.06226.
//!
//! Run with: cargo run --example create_vocab_size_validator
//!
//! Added by PMAT-127 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const MIN_VOCAB: u32 = 256;
const MAX_VOCAB: u32 = 1_000_000;
const SIMD_ALIGN: u32 = 64;

#[derive(Debug, PartialEq)]
pub enum VocabVerdict {
    Ok,
    BelowByteFloor,
    AboveCeiling,
    NotAligned { suggested: u32 },
    InvalidZero,
}

pub fn validate(vocab_size: u32) -> VocabVerdict {
    if vocab_size == 0 {
        return VocabVerdict::InvalidZero;
    }
    if vocab_size < MIN_VOCAB {
        return VocabVerdict::BelowByteFloor;
    }
    if vocab_size > MAX_VOCAB {
        return VocabVerdict::AboveCeiling;
    }
    if vocab_size % SIMD_ALIGN != 0 {
        let next = next_aligned(vocab_size, SIMD_ALIGN);
        return VocabVerdict::NotAligned { suggested: next };
    }
    VocabVerdict::Ok
}

fn next_aligned(n: u32, align: u32) -> u32 {
    if align == 0 {
        return n;
    }
    let r = n % align;
    if r == 0 {
        n
    } else {
        n + (align - r)
    }
}

pub fn padding_overhead_pct(vocab_size: u32, align: u32) -> Option<f64> {
    if vocab_size == 0 || align == 0 {
        return None;
    }
    let aligned = next_aligned(vocab_size, align);
    Some((f64::from(aligned - vocab_size) / f64::from(vocab_size)) * 100.0)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("create_vocab_size_validator")?;

    for n in [0u32, 200, 32_000, 50_257, 128_256, 2_000_000] {
        println!("vocab={n}  →  {:?}", validate(n));
    }
    println!(
        "padding 50257 to 64: {:?}%",
        padding_overhead_pct(50_257, 64)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_llama_vocab_passes() {
        // 32000 is a multiple of 64.
        assert_eq!(validate(32_000), VocabVerdict::Ok);
    }

    #[test]
    fn llama3_vocab_passes() {
        // 128_256 is a multiple of 64 (256 × 501).
        assert_eq!(validate(128_256), VocabVerdict::Ok);
    }

    #[test]
    fn gpt2_vocab_misaligned() {
        // 50_257 is prime → not a multiple of 64.
        let v = validate(50_257);
        assert!(matches!(v, VocabVerdict::NotAligned { suggested: 50_304 }));
    }

    #[test]
    fn under_256_byte_floor_rejected() {
        assert_eq!(validate(200), VocabVerdict::BelowByteFloor);
    }

    #[test]
    fn over_max_rejected() {
        assert_eq!(validate(2_000_000), VocabVerdict::AboveCeiling);
    }

    #[test]
    fn zero_invalid() {
        assert_eq!(validate(0), VocabVerdict::InvalidZero);
    }

    #[test]
    fn at_byte_floor_passes() {
        // 256 is exactly at floor and is multiple of 64.
        assert_eq!(validate(256), VocabVerdict::Ok);
    }

    #[test]
    fn next_aligned_basic() {
        assert_eq!(next_aligned(257, 64), 320);
        assert_eq!(next_aligned(320, 64), 320);
    }

    #[test]
    fn padding_overhead_returns_some() {
        // 50_257 → 50_304: (47/50_257) ≈ 0.094%.
        let pct = padding_overhead_pct(50_257, 64).unwrap();
        assert!(pct > 0.0 && pct < 1.0);
    }

    #[test]
    fn padding_overhead_zero_when_aligned() {
        let pct = padding_overhead_pct(32_000, 64).unwrap();
        assert!(pct.abs() < 1e-9);
    }
}
