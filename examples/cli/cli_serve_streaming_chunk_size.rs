//! # apr serve --stream-chunk-size — SSE Chunk Size Picker
//!
//! `apr serve --stream` uses Server-Sent Events. Chunk size: 1 token =
//! lowest TTFT (time to first token) but most overhead; 32 tokens =
//! batched. Floor: 1 (per-token); ceiling: 256 (latency wash). This
//! recipe builds the picker + flush-cadence estimator.
//!
//! Demonstrates the **SERVE.5** recipe for PMAT-116 (apr serve coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender SERVE-001 + W3C SSE recommendation
//!
//! Run with: cargo run --example cli_serve_streaming_chunk_size
//!
//! Added by PMAT-116 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ChunkVerdict {
    Ok,
    InvalidZero,
    AboveLatencyCap { recommended: u32 },
}

const MAX_CHUNK_TOKENS: u32 = 256;

pub fn classify(chunk: u32) -> ChunkVerdict {
    if chunk == 0 {
        return ChunkVerdict::InvalidZero;
    }
    if chunk > MAX_CHUNK_TOKENS {
        return ChunkVerdict::AboveLatencyCap {
            recommended: MAX_CHUNK_TOKENS,
        };
    }
    ChunkVerdict::Ok
}

pub fn estimated_ttft_ms(chunk: u32, tokens_per_sec: u32) -> Option<f64> {
    if chunk == 0 || tokens_per_sec == 0 {
        return None;
    }
    Some(f64::from(chunk) * 1000.0 / f64::from(tokens_per_sec))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_serve_streaming_chunk_size")?;

    for c in [0u32, 1, 8, 32, 256, 1024] {
        let v = classify(c);
        let ttft = estimated_ttft_ms(c, 100);
        println!("chunk={c:>4}  →  {v:?}  TTFT≈{ttft:?}ms");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn zero_invalid() {
        assert_eq!(classify(0), ChunkVerdict::InvalidZero);
    }

    #[test]
    fn one_token_passes() {
        // Per-token streaming is the most responsive.
        assert_eq!(classify(1), ChunkVerdict::Ok);
    }

    #[test]
    fn typical_8_to_32_passes() {
        assert_eq!(classify(8), ChunkVerdict::Ok);
        assert_eq!(classify(32), ChunkVerdict::Ok);
    }

    #[test]
    fn at_ceiling_passes() {
        assert_eq!(classify(MAX_CHUNK_TOKENS), ChunkVerdict::Ok);
    }

    #[test]
    fn above_ceiling_rejected() {
        let v = classify(1024);
        assert!(matches!(v, ChunkVerdict::AboveLatencyCap { .. }));
    }

    #[test]
    fn ttft_scales_with_chunk_size() {
        let small = estimated_ttft_ms(1, 100).unwrap();
        let big = estimated_ttft_ms(32, 100).unwrap();
        assert!(big > small);
    }

    #[test]
    fn ttft_inverse_scales_with_throughput() {
        // Faster model → lower TTFT for same chunk.
        let slow = estimated_ttft_ms(8, 50).unwrap();
        let fast = estimated_ttft_ms(8, 200).unwrap();
        assert!(fast < slow);
    }

    #[test]
    fn ttft_zero_chunk_or_throughput_yields_none() {
        assert!(estimated_ttft_ms(0, 100).is_none());
        assert!(estimated_ttft_ms(8, 0).is_none());
    }

    #[test]
    fn ttft_eight_at_one_hundred_eighty_ms() {
        // 8 tokens / 100 tps = 80 ms.
        let t = estimated_ttft_ms(8, 100).unwrap();
        assert!((t - 80.0).abs() < 1e-9);
    }
}
