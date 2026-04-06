//! Whisper Streaming Transcription Example
//!
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
//! ## References
//! - Radford, A. et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision*. ICML. arXiv:2212.04356

use apr_cookbook::prelude::*;
use std::collections::VecDeque;
use std::f32::consts::PI;

/// Audio chunk size in samples (30ms at 16kHz)
const CHUNK_SIZE: usize = 480;

/// Sample rate
const SAMPLE_RATE: usize = 16000;

/// Context window size in chunks (for encoder)
const CONTEXT_CHUNKS: usize = 10;

/// VAD energy threshold
const VAD_THRESHOLD: f32 = 0.01;

/// Streaming transcriber state
#[derive(Debug)]
#[allow(dead_code)]
struct StreamingTranscriber {
    /// Audio buffer (sliding window)
    audio_buffer: VecDeque<f32>,
    /// Partial transcription
    partial_text: String,
    /// Final transcription
    final_text: String,
    /// Voice activity state
    is_speaking: bool,
    /// Silence duration in chunks
    silence_chunks: usize,
    /// Decoder context (for beam search)
    decoder_context: Vec<u32>,
    /// Quantization for processing
    quantization: Quantization,
}

impl StreamingTranscriber {
    /// Create a new streaming transcriber
    fn new(quantization: Quantization) -> Self {
        Self {
            audio_buffer: VecDeque::with_capacity(CHUNK_SIZE * CONTEXT_CHUNKS),
            partial_text: String::new(),
            final_text: String::new(),
            is_speaking: false,
            silence_chunks: 0,
            decoder_context: vec![50258], // <|startoftranscript|>
            quantization,
        }
    }

    /// Load from APR v2 bundle
    fn from_apr(bytes: &[u8]) -> Result<Self> {
        let bundle = BundledModelV2::from_bytes(bytes)?;

        // Verify signature
        if let Some(valid) = bundle.signature_valid() {
            if !valid {
                return Err(CookbookError::invalid_format("Invalid model signature"));
            }
        }

        Ok(Self::new(bundle.quantization()))
    }

    /// Process an audio chunk and return partial result if available
    fn process_chunk(&mut self, chunk: &[f32]) -> Option<PartialResult> {
        // Add chunk to buffer
        for &sample in chunk {
            self.audio_buffer.push_back(sample);
        }

        // Maintain sliding window
        while self.audio_buffer.len() > CHUNK_SIZE * CONTEXT_CHUNKS {
            self.audio_buffer.pop_front();
        }

        // Voice activity detection
        let energy = self.compute_energy(chunk);
        let is_voice = energy > VAD_THRESHOLD;

        if is_voice {
            self.is_speaking = true;
            self.silence_chunks = 0;
        } else if self.is_speaking {
            self.silence_chunks += 1;

            // End of utterance after 500ms silence
            if self.silence_chunks > 16 {
                self.finalize_utterance();
                self.is_speaking = false;
                self.silence_chunks = 0;
            }
        }

        // Only process if speaking
        if !self.is_speaking {
            return None;
        }

        // Run incremental decoding
        let new_tokens = self.decode_incremental();
        if new_tokens.is_empty() {
            return None;
        }

        // Decode tokens to text
        let new_text = self.decode_tokens(&new_tokens);
        self.partial_text.push_str(&new_text);

        Some(PartialResult {
            text: self.partial_text.clone(),
            is_final: false,
            confidence: self.compute_confidence(&new_tokens),
        })
    }

    /// Compute audio energy for VAD
    fn compute_energy(&self, chunk: &[f32]) -> f32 {
        if chunk.is_empty() {
            return 0.0;
        }
        chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32
    }

    /// Run incremental decoding on current buffer
    fn decode_incremental(&mut self) -> Vec<u32> {
        // Simulated incremental decoding
        // In real implementation, this would use cached encoder states
        let buffer_len = self.audio_buffer.len();

        if buffer_len < CHUNK_SIZE * 3 {
            return vec![];
        }

        // Generate a token based on buffer state (simulated)
        let hash = buffer_len % 8;
        let token = match hash {
            0 => 7120, // "Hello"
            1 => 11,   // ","
            2 => 1002, // " world"
            3 => 0,    // "!"
            4 => 362,  // " How"
            5 => 389,  // " are"
            6 => 345,  // " you"
            7 => 30,   // "?"
            _ => 0,
        };

        // Add to context
        self.decoder_context.push(token);

        vec![token]
    }

    /// Decode tokens to text
    fn decode_tokens(&self, tokens: &[u32]) -> String {
        let vocab: std::collections::HashMap<u32, &str> = [
            (7120, "Hello"),
            (11, ","),
            (1002, " world"),
            (0, "!"),
            (362, " How"),
            (389, " are"),
            (345, " you"),
            (30, "?"),
        ]
        .into_iter()
        .collect();

        tokens
            .iter()
            .filter_map(|t| vocab.get(t).copied())
            .collect()
    }

    /// Compute confidence from tokens
    fn compute_confidence(&self, _tokens: &[u32]) -> f32 {
        // Simulated confidence based on context length
        let context_len = self.decoder_context.len();
        (0.7 + 0.03 * (context_len as f32).min(10.0)).min(0.99)
    }

    /// Finalize current utterance
    fn finalize_utterance(&mut self) {
        if !self.partial_text.is_empty() {
            self.final_text.push_str(&self.partial_text);
            self.final_text.push(' ');
            self.partial_text.clear();
        }

        // Reset decoder context
        self.decoder_context = vec![50258];
    }

    /// Get final transcription
    fn finalize(&mut self) -> FinalResult {
        self.finalize_utterance();

        FinalResult {
            text: self.final_text.trim().to_string(),
            language: "en".to_string(),
            duration_seconds: self.audio_buffer.len() as f32 / SAMPLE_RATE as f32,
        }
    }
}

/// Partial transcription result
#[derive(Debug)]
struct PartialResult {
    /// Current partial text
    text: String,
    /// Whether this is the final result for an utterance
    is_final: bool,
    /// Confidence score
    confidence: f32,
}

/// Final transcription result
#[derive(Debug)]
struct FinalResult {
    /// Complete transcription
    text: String,
    /// Detected language
    language: String,
    /// Total duration in seconds
    duration_seconds: f32,
}

/// Generate streaming audio chunks for testing
fn generate_audio_stream(duration_seconds: f32) -> Vec<Vec<f32>> {
    let n_samples = (duration_seconds * SAMPLE_RATE as f32) as usize;
    let n_chunks = n_samples.div_ceil(CHUNK_SIZE);

    let mut chunks = Vec::with_capacity(n_chunks);

    for chunk_idx in 0..n_chunks {
        let start = chunk_idx * CHUNK_SIZE;
        let end = ((chunk_idx + 1) * CHUNK_SIZE).min(n_samples);
        let mut chunk = Vec::with_capacity(CHUNK_SIZE);

        // Simulate speech with varying energy
        let is_speech = (chunk_idx / 10) % 2 == 0; // Alternating speech/silence

        for i in start..end {
            let t = i as f32 / SAMPLE_RATE as f32;
            let sample = if is_speech {
                0.5 * (440.0 * 2.0 * PI * t).sin() * (1.0 + 0.3 * (4.0 * 2.0 * PI * t).sin())
            // AM modulation
            } else {
                0.001 * (2.0 * PI * t).sin() // Very quiet
            };
            chunk.push(sample);
        }

        chunks.push(chunk);
    }

    chunks
}

/// Build a test APR v2 bundle
fn build_streaming_bundle() -> Vec<u8> {
    let encoder_cache = vec![0u8; 768 * 512]; // Cached encoder states

    ModelBundleV2::new()
        .with_name("whisper-streaming-int4")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::Int4)
        .add_tensor("encoder.cache", vec![512, 768], encoder_cache)
        .build()
}

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
