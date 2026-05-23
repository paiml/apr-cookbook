//! # Advanced Streaming Response Buffer
//!
//! Buffer streaming response chunks. On client disconnect / reconnect,
//! resume from last received chunk index.
//!
//! Demonstrates the **ADV.25** recipe for PMAT-154 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: HTTP/2 stream resumption + Server-Sent Events resume token.
//!
//! Run with: cargo run --example adv_chunked_response_buffer
//!
//! Added by PMAT-154 (catalog 1009→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ResumeVerdict {
    StartFresh,
    ResumeFromChunk {
        chunk_index: u32,
        bytes_skipped: u64,
    },
    InvalidChunkIndex {
        last_index: u32,
        requested: u32,
    },
    StreamComplete,
}

pub fn resume(
    last_received_chunk: Option<u32>,
    total_chunks_so_far: u32,
    is_complete: bool,
    bytes_per_chunk: u64,
) -> ResumeVerdict {
    if is_complete {
        return ResumeVerdict::StreamComplete;
    }
    let Some(last) = last_received_chunk else {
        return ResumeVerdict::StartFresh;
    };
    if last >= total_chunks_so_far {
        return ResumeVerdict::InvalidChunkIndex {
            last_index: total_chunks_so_far.saturating_sub(1),
            requested: last,
        };
    }
    let chunk_index = last + 1;
    let bytes_skipped = u64::from(chunk_index) * bytes_per_chunk;
    ResumeVerdict::ResumeFromChunk {
        chunk_index,
        bytes_skipped,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_chunked_response_buffer")?;

    println!("fresh start: {:?}", resume(None, 0, false, 256));
    println!("resume from 5: {:?}", resume(Some(5), 10, false, 256));
    println!("complete: {:?}", resume(Some(10), 10, true, 256));
    println!("invalid: {:?}", resume(Some(20), 10, false, 256));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_prior_chunk_starts_fresh() {
        assert_eq!(resume(None, 0, false, 256), ResumeVerdict::StartFresh);
    }

    #[test]
    fn complete_stream_signaled() {
        let v = resume(Some(10), 10, true, 256);
        assert_eq!(v, ResumeVerdict::StreamComplete);
    }

    #[test]
    fn resume_after_chunk_5() {
        let v = resume(Some(5), 10, false, 256);
        if let ResumeVerdict::ResumeFromChunk { chunk_index, .. } = v {
            assert_eq!(chunk_index, 6);
        }
    }

    #[test]
    fn invalid_index_too_high() {
        let v = resume(Some(20), 10, false, 256);
        assert!(matches!(v, ResumeVerdict::InvalidChunkIndex { .. }));
    }

    #[test]
    fn bytes_skipped_proportional() {
        let v = resume(Some(5), 10, false, 256);
        if let ResumeVerdict::ResumeFromChunk { bytes_skipped, .. } = v {
            // chunk_index = 6, bytes_skipped = 6 × 256 = 1536.
            assert_eq!(bytes_skipped, 1536);
        }
    }

    #[test]
    fn resume_at_last_chunk_invalid() {
        // last_received == total_chunks_so_far is invalid.
        let v = resume(Some(10), 10, false, 256);
        assert!(matches!(v, ResumeVerdict::InvalidChunkIndex { .. }));
    }

    #[test]
    fn resume_just_below_total_ok() {
        let v = resume(Some(9), 10, false, 256);
        assert!(matches!(v, ResumeVerdict::ResumeFromChunk { .. }));
    }

    #[test]
    fn complete_overrides_invalid_index() {
        // Even with invalid index, complete signal wins.
        let v = resume(Some(100), 10, true, 256);
        assert_eq!(v, ResumeVerdict::StreamComplete);
    }

    #[test]
    fn fresh_start_when_complete_no_prior() {
        let v = resume(None, 0, true, 256);
        assert_eq!(v, ResumeVerdict::StreamComplete);
    }

    #[test]
    fn deterministic() {
        let a = resume(Some(5), 10, false, 256);
        let b = resume(Some(5), 10, false, 256);
        assert_eq!(a, b);
    }
}
