//! # Speech Streaming Chunk Overlap Planner
//!
//! Streaming Whisper transcription windows audio into chunks (default
//! 30 s) with overlap (default 5 s) so the model sees boundary
//! context. Constraints: 5 s ≤ chunk_secs ≤ 30 s; 0 ≤ overlap_secs <
//! chunk_secs / 2 (avoid degenerate cases). This recipe builds the
//! validator + total-windows calculator.
//!
//! Demonstrates the **SPEECH.4** recipe for PMAT-123 (speech coverage —
//! closing F-invariant gap from 1 → 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Radford et al. (2022). Whisper §3.4 — Long-form transcription.
//!
//! Run with: cargo run --example speech_chunk_overlap_planner
//!
//! Added by PMAT-123 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ChunkVerdict {
    Ok { num_windows: u32 },
    ChunkTooShort,
    ChunkTooLong,
    OverlapNegative,
    OverlapTooLarge,
    InvalidAudioLength,
}

const MIN_CHUNK_SECS: f64 = 5.0;
const MAX_CHUNK_SECS: f64 = 30.0;

pub fn plan(audio_secs: f64, chunk_secs: f64, overlap_secs: f64) -> ChunkVerdict {
    if !audio_secs.is_finite() || audio_secs <= 0.0 {
        return ChunkVerdict::InvalidAudioLength;
    }
    if !chunk_secs.is_finite() || chunk_secs < MIN_CHUNK_SECS {
        return ChunkVerdict::ChunkTooShort;
    }
    if chunk_secs > MAX_CHUNK_SECS {
        return ChunkVerdict::ChunkTooLong;
    }
    if !overlap_secs.is_finite() || overlap_secs < 0.0 {
        return ChunkVerdict::OverlapNegative;
    }
    if overlap_secs >= chunk_secs / 2.0 {
        return ChunkVerdict::OverlapTooLarge;
    }
    let stride = chunk_secs - overlap_secs;
    let num_windows = if audio_secs <= chunk_secs {
        1
    } else {
        // ceil((audio - chunk) / stride) + 1
        let extra = audio_secs - chunk_secs;
        (extra / stride).ceil() as u32 + 1
    };
    ChunkVerdict::Ok { num_windows }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("speech_chunk_overlap_planner")?;

    let cases = [
        (60.0, 30.0, 5.0),
        (120.0, 30.0, 5.0),
        (10.0, 30.0, 5.0),
        (60.0, 3.0, 1.0),
        (60.0, 30.0, 20.0),
        (-1.0, 30.0, 5.0),
    ];
    for (audio, chunk, overlap) in cases {
        println!(
            "audio={audio}s chunk={chunk}s overlap={overlap}s  →  {:?}",
            plan(audio, chunk, overlap)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn audio_shorter_than_chunk_yields_one_window() {
        let v = plan(10.0, 30.0, 5.0);
        assert_eq!(v, ChunkVerdict::Ok { num_windows: 1 });
    }

    #[test]
    fn typical_60s_30s_chunk_5s_overlap() {
        // stride = 25; (60 - 30) / 25 = 1.2 → ceil = 2 → 3 windows.
        let v = plan(60.0, 30.0, 5.0);
        assert_eq!(v, ChunkVerdict::Ok { num_windows: 3 });
    }

    #[test]
    fn overlap_zero_no_redundant_compute() {
        // 60 / 30 = 2 windows.
        let v = plan(60.0, 30.0, 0.0);
        assert_eq!(v, ChunkVerdict::Ok { num_windows: 2 });
    }

    #[test]
    fn chunk_too_short_rejected() {
        assert_eq!(plan(60.0, 3.0, 1.0), ChunkVerdict::ChunkTooShort);
    }

    #[test]
    fn chunk_too_long_rejected() {
        assert_eq!(plan(60.0, 60.0, 5.0), ChunkVerdict::ChunkTooLong);
    }

    #[test]
    fn negative_overlap_rejected() {
        assert_eq!(plan(60.0, 30.0, -1.0), ChunkVerdict::OverlapNegative);
    }

    #[test]
    fn overlap_at_or_above_half_chunk_rejected() {
        // chunk=30, half=15; overlap ≥ 15 → reject.
        assert_eq!(plan(60.0, 30.0, 15.0), ChunkVerdict::OverlapTooLarge);
        assert_eq!(plan(60.0, 30.0, 20.0), ChunkVerdict::OverlapTooLarge);
    }

    #[test]
    fn audio_zero_or_negative_rejected() {
        assert_eq!(plan(0.0, 30.0, 5.0), ChunkVerdict::InvalidAudioLength);
        assert_eq!(plan(-5.0, 30.0, 5.0), ChunkVerdict::InvalidAudioLength);
    }

    #[test]
    fn long_audio_scales_linearly() {
        // 120 s audio with 30/5 → stride 25; (120 - 30) / 25 = 3.6 → ceil 4 → 5 windows.
        let v = plan(120.0, 30.0, 5.0);
        assert_eq!(v, ChunkVerdict::Ok { num_windows: 5 });
    }

    #[test]
    fn at_chunk_boundary_one_window() {
        // audio == chunk → 1 window (boundary).
        let v = plan(30.0, 30.0, 5.0);
        assert_eq!(v, ChunkVerdict::Ok { num_windows: 1 });
    }
}
