//! # API Streaming Chunk Size Picker
//!
//! Server-Sent Events / WebSocket frame sizing trade-off:
//!   too small → header overhead dominates
//!   too large → first-token latency suffers
//!   middle → smooth streaming
//!
//! Heuristic:
//!   text completion: 64-256 bytes per chunk (1 token + framing)
//!   audio frames: 1 KiB - 4 KiB (one Opus frame)
//!   binary embedding: 4 KiB - 16 KiB (full frame batched)
//!
//! Demonstrates the **API.12** recipe for PMAT-143 (api round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTTP/2 frame-size best-practice (CHUNK_SIZE).
//!
//! Run with: cargo run --example api_streaming_chunk_size
//!
//! Added by PMAT-143 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamPayload {
    TextToken,
    AudioFrame,
    BinaryEmbedding,
}

#[derive(Debug, PartialEq)]
pub enum ChunkVerdict {
    Ok {
        chunk_bytes: u32,
        chunks_per_second_target: u32,
    },
    InvalidPayload,
}

pub fn pick(payload: StreamPayload, target_first_token_ms: u32) -> ChunkVerdict {
    if target_first_token_ms == 0 {
        return ChunkVerdict::InvalidPayload;
    }
    let (min_bytes, max_bytes) = match payload {
        StreamPayload::TextToken => (64u32, 256u32),
        StreamPayload::AudioFrame => (1024, 4096),
        StreamPayload::BinaryEmbedding => (4096, 16_384),
    };
    let chunk_bytes = if target_first_token_ms <= 100 {
        min_bytes
    } else if target_first_token_ms <= 500 {
        (min_bytes + max_bytes) / 2
    } else {
        max_bytes
    };
    let chunks_per_second_target = 1000_u32
        .checked_div(target_first_token_ms)
        .unwrap_or(1)
        .max(1);
    ChunkVerdict::Ok {
        chunk_bytes,
        chunks_per_second_target,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("api_streaming_chunk_size")?;

    println!("text 50ms: {:?}", pick(StreamPayload::TextToken, 50));
    println!("text 200ms: {:?}", pick(StreamPayload::TextToken, 200));
    println!("audio: {:?}", pick(StreamPayload::AudioFrame, 100));
    println!(
        "embedding: {:?}",
        pick(StreamPayload::BinaryEmbedding, 1000)
    );
    println!("invalid: {:?}", pick(StreamPayload::TextToken, 0));
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
    fn text_under_100ms_min() {
        let v = pick(StreamPayload::TextToken, 50);
        if let ChunkVerdict::Ok { chunk_bytes, .. } = v {
            assert_eq!(chunk_bytes, 64);
        }
    }

    #[test]
    fn text_under_500ms_mid() {
        let v = pick(StreamPayload::TextToken, 200);
        if let ChunkVerdict::Ok { chunk_bytes, .. } = v {
            assert_eq!(chunk_bytes, 160);
        }
    }

    #[test]
    fn text_above_500ms_max() {
        let v = pick(StreamPayload::TextToken, 1000);
        if let ChunkVerdict::Ok { chunk_bytes, .. } = v {
            assert_eq!(chunk_bytes, 256);
        }
    }

    #[test]
    fn audio_uses_audio_range() {
        let v = pick(StreamPayload::AudioFrame, 100);
        if let ChunkVerdict::Ok { chunk_bytes, .. } = v {
            assert!((1024..=4096).contains(&chunk_bytes));
        }
    }

    #[test]
    fn embedding_uses_embedding_range() {
        let v = pick(StreamPayload::BinaryEmbedding, 1000);
        if let ChunkVerdict::Ok { chunk_bytes, .. } = v {
            assert!((4096..=16_384).contains(&chunk_bytes));
        }
    }

    #[test]
    fn invalid_zero_target_rejected() {
        assert_eq!(
            pick(StreamPayload::TextToken, 0),
            ChunkVerdict::InvalidPayload
        );
    }

    #[test]
    fn higher_target_yields_smaller_cps() {
        let v_fast = pick(StreamPayload::TextToken, 50);
        let v_slow = pick(StreamPayload::TextToken, 1000);
        if let (
            ChunkVerdict::Ok {
                chunks_per_second_target: f,
                ..
            },
            ChunkVerdict::Ok {
                chunks_per_second_target: s,
                ..
            },
        ) = (v_fast, v_slow)
        {
            assert!(f > s);
        }
    }

    #[test]
    fn cps_at_least_one() {
        let v = pick(StreamPayload::BinaryEmbedding, 10_000);
        if let ChunkVerdict::Ok {
            chunks_per_second_target,
            ..
        } = v
        {
            assert!(chunks_per_second_target >= 1);
        }
    }

    #[test]
    fn text_smaller_than_embedding() {
        let text = pick(StreamPayload::TextToken, 100);
        let embed = pick(StreamPayload::BinaryEmbedding, 100);
        if let (ChunkVerdict::Ok { chunk_bytes: t, .. }, ChunkVerdict::Ok { chunk_bytes: e, .. }) =
            (text, embed)
        {
            assert!(t < e);
        }
    }

    #[test]
    fn boundary_at_100ms_min() {
        // exactly 100ms → still min.
        let v = pick(StreamPayload::TextToken, 100);
        if let ChunkVerdict::Ok { chunk_bytes, .. } = v {
            assert_eq!(chunk_bytes, 64);
        }
    }
}
