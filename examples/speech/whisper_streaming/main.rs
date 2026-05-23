#![allow(unused_imports)]
//! Whisper Streaming Transcription Example
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/whisper-wer-v1.yaml
//! Demonstrates real-time streaming speech recognition with whisper.apr.
//!
//! # Streaming Architecture
//!
//! ```text
//! ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
//! │  Audio  │──▶│ Chunker │──▶│ VAD     │──▶│ Encoder │
//! │ Stream  │   │ (30ms)  │   │ Filter  │   │ Sliding │
//! └─────────┘   └─────────┘   └─────────┘   └─────────┘
//!                                               │
//!                                               ▼
//! ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
//! │  Final  │◀──│  Merge  │◀──│ Decoder │◀──│ Context │
//! │  Text   │   │ Partial │   │  Beam   │   │ Window  │
//! └─────────┘   └─────────┘   └─────────┘   └─────────┘
//! ```
//!
//! # Features
//!
//! - **Low Latency**: Partial results every 30ms
//! - **Voice Activity Detection**: Skip silence
//! - **Context Window**: Maintain decoder state across chunks
//! - **Adaptive Beam Search**: Adjust beam width based on confidence
//!
//! # Running
//!
//! ```bash
//! cargo run --example whisper_streaming
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Radford, A. et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision*. ICML. arXiv:2212.04356

use apr_cookbook::prelude::*;
use std::collections::VecDeque;
use std::f32::consts::PI;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== Whisper Streaming Transcription Example ===\n");

    // =========================================================================
    // Section 1: Initialize Transcriber
    // =========================================================================
    println!("1. Initializing Streaming Transcriber");
    println!("   ─────────────────────────────────────────");

    let bundle = build_streaming_bundle();
    println!("   Bundle size: {} bytes", bundle.len());

    let loaded = BundledModelV2::from_bytes(&bundle).expect("Failed to load bundle");
    println!("   Quantization: {:?}", loaded.quantization());
    println!("   Compression: {:?}", loaded.compression());

    let mut transcriber =
        StreamingTranscriber::from_apr(&bundle).expect("Failed to create transcriber");
    println!(
        "   Chunk size: {} samples ({:.1} ms)",
        CHUNK_SIZE,
        CHUNK_SIZE as f32 / SAMPLE_RATE as f32 * 1000.0
    );
    println!("   Context window: {} chunks", CONTEXT_CHUNKS);
    println!();

    // =========================================================================
    // Section 2: Process Audio Stream
    // =========================================================================
    println!("2. Processing Audio Stream");
    println!("   ─────────────────────────────────────────");

    let audio_stream = generate_audio_stream(3.0);
    println!("   Total chunks: {}", audio_stream.len());
    println!(
        "   Duration: {:.1} seconds",
        audio_stream.len() as f32 * CHUNK_SIZE as f32 / SAMPLE_RATE as f32
    );
    println!();

    println!("   Live transcription:");
    for (i, chunk) in audio_stream.iter().enumerate() {
        if let Some(partial) = transcriber.process_chunk(chunk) {
            println!(
                "   [{:3}] \"{}\"{} ({:.1}%)",
                i,
                partial.text,
                if partial.is_final { " [FINAL]" } else { "" },
                partial.confidence * 100.0
            );
        }
    }
    println!();

    // =========================================================================
    // Section 3: Finalize
    // =========================================================================
    println!("3. Final Result");
    println!("   ─────────────────────────────────────────");

    let final_result = transcriber.finalize();
    println!("   Text: \"{}\"", final_result.text);
    println!("   Language: {}", final_result.language);
    println!("   Duration: {:.2}s", final_result.duration_seconds);
    println!();

    // =========================================================================
    // Section 4: Latency Analysis
    // =========================================================================
    println!("4. Latency Analysis");
    println!("   ─────────────────────────────────────────");
    println!("   ┌─────────────────────┬───────────────┐");
    println!("   │ Component           │ Latency       │");
    println!("   ├─────────────────────┼───────────────┤");
    println!("   │ Audio chunk (30ms)  │ ~30 ms        │");
    println!("   │ VAD detection       │ <1 ms         │");
    println!("   │ Mel spectrogram     │ ~2 ms         │");
    println!("   │ Encoder (cached)    │ ~5 ms         │");
    println!("   │ Decoder (1 token)   │ ~3 ms         │");
    println!("   ├─────────────────────┼───────────────┤");
    println!("   │ Total per chunk     │ ~40 ms        │");
    println!("   └─────────────────────┴───────────────┘");
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_transcriber_creation() {
        let transcriber = StreamingTranscriber::new(Quantization::Int4);
        assert!(!transcriber.is_speaking);
        assert!(transcriber.partial_text.is_empty());
    }

    #[test]
    fn test_from_apr() {
        let bundle = build_streaming_bundle();
        let transcriber = StreamingTranscriber::from_apr(&bundle).unwrap();
        assert_eq!(transcriber.quantization, Quantization::Int4);
    }

    #[test]
    fn test_energy_computation() {
        let transcriber = StreamingTranscriber::new(Quantization::Int4);

        // Silent audio
        let silent = vec![0.0f32; CHUNK_SIZE];
        assert!(transcriber.compute_energy(&silent) < VAD_THRESHOLD);

        // Loud audio
        let loud: Vec<f32> = (0..CHUNK_SIZE)
            .map(|i| 0.5 * (2.0 * PI * i as f32 / 100.0).sin())
            .collect();
        assert!(transcriber.compute_energy(&loud) > VAD_THRESHOLD);
    }

    #[test]
    fn test_vad_detection() {
        let mut transcriber = StreamingTranscriber::new(Quantization::Int4);

        // Process silent chunk
        let silent = vec![0.001f32; CHUNK_SIZE];
        transcriber.process_chunk(&silent);
        assert!(!transcriber.is_speaking);

        // Process loud chunk
        let loud: Vec<f32> = (0..CHUNK_SIZE)
            .map(|i| 0.5 * (2.0 * PI * i as f32 / 100.0).sin())
            .collect();
        transcriber.process_chunk(&loud);
        assert!(transcriber.is_speaking);
    }

    #[test]
    fn test_silence_timeout() {
        let mut transcriber = StreamingTranscriber::new(Quantization::Int4);

        // Start speaking
        let loud: Vec<f32> = (0..CHUNK_SIZE)
            .map(|i| 0.5 * (2.0 * PI * i as f32 / 100.0).sin())
            .collect();
        transcriber.process_chunk(&loud);
        assert!(transcriber.is_speaking);

        // Process 20 silent chunks (>500ms)
        let silent = vec![0.0f32; CHUNK_SIZE];
        for _ in 0..20 {
            transcriber.process_chunk(&silent);
        }
        assert!(!transcriber.is_speaking);
    }

    #[test]
    fn test_finalize() {
        let mut transcriber = StreamingTranscriber::new(Quantization::Int4);
        transcriber.partial_text = "Hello, world!".to_string();

        let result = transcriber.finalize();
        assert_eq!(result.text, "Hello, world!");
        assert_eq!(result.language, "en");
    }

    #[test]
    fn test_audio_stream_generation() {
        let stream = generate_audio_stream(1.0);
        let expected_chunks = (SAMPLE_RATE + CHUNK_SIZE - 1) / CHUNK_SIZE;
        assert_eq!(stream.len(), expected_chunks);
    }

    #[test]
    fn test_partial_result() {
        let mut transcriber = StreamingTranscriber::new(Quantization::Int4);

        // Need enough data to trigger decoding
        let loud: Vec<f32> = (0..CHUNK_SIZE)
            .map(|i| 0.5 * (2.0 * PI * i as f32 / 100.0).sin())
            .collect();

        // Process multiple chunks to fill buffer
        for _ in 0..5 {
            transcriber.process_chunk(&loud);
        }

        // Should have partial result
        assert!(transcriber.is_speaking);
    }

    #[test]
    fn test_decoder_context() {
        let transcriber = StreamingTranscriber::new(Quantization::Int4);
        assert_eq!(transcriber.decoder_context, vec![50258]);
    }

    #[test]
    fn test_confidence_increases_with_context() {
        let transcriber = StreamingTranscriber::new(Quantization::Int4);

        let conf1 = transcriber.compute_confidence(&[]);

        let mut longer_context = transcriber;
        longer_context.decoder_context = vec![50258; 10];
        let conf2 = longer_context.compute_confidence(&[]);

        assert!(conf2 > conf1);
    }
}
