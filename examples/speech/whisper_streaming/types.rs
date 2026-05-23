//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use std::collections::VecDeque;
use std::f32::consts::PI;

/// Audio chunk size in samples (30ms at 16kHz)
pub const CHUNK_SIZE: usize = 480;

/// Sample rate
pub const SAMPLE_RATE: usize = 16000;

/// Context window size in chunks (for encoder)
pub const CONTEXT_CHUNKS: usize = 10;

/// VAD energy threshold
pub const VAD_THRESHOLD: f32 = 0.01;

/// Streaming transcriber state
#[derive(Debug)]
#[allow(dead_code)]
pub struct StreamingTranscriber {
    // Audio buffer (sliding window)
    pub audio_buffer: VecDeque<f32>,
    // Partial transcription
    pub partial_text: String,
    // Final transcription
    pub final_text: String,
    // Voice activity state
    pub is_speaking: bool,
    // Silence duration in chunks
    pub silence_chunks: usize,
    // Decoder context (for beam search)
    pub decoder_context: Vec<u32>,
    // Quantization for processing
    pub quantization: Quantization,
}

impl StreamingTranscriber {
    /// Create a new streaming transcriber
    pub fn new(quantization: Quantization) -> Self {
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
    pub fn from_apr(bytes: &[u8]) -> Result<Self> {
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
    pub fn process_chunk(&mut self, chunk: &[f32]) -> Option<PartialResult> {
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
    pub fn compute_energy(&self, chunk: &[f32]) -> f32 {
        if chunk.is_empty() {
            return 0.0;
        }
        chunk.iter().map(|s| s * s).sum::<f32>() / chunk.len() as f32
    }

    /// Run incremental decoding on current buffer
    pub fn decode_incremental(&mut self) -> Vec<u32> {
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
    pub fn decode_tokens(&self, tokens: &[u32]) -> String {
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
    pub fn compute_confidence(&self, _tokens: &[u32]) -> f32 {
        // Simulated confidence based on context length
        let context_len = self.decoder_context.len();
        (0.7 + 0.03 * (context_len as f32).min(10.0)).min(0.99)
    }

    /// Finalize current utterance
    pub fn finalize_utterance(&mut self) {
        if !self.partial_text.is_empty() {
            self.final_text.push_str(&self.partial_text);
            self.final_text.push(' ');
            self.partial_text.clear();
        }

        // Reset decoder context
        self.decoder_context = vec![50258];
    }

    /// Get final transcription
    pub fn finalize(&mut self) -> FinalResult {
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
pub struct PartialResult {
    // Current partial text
    pub text: String,
    // Whether this is the final result for an utterance
    pub is_final: bool,
    // Confidence score
    pub confidence: f32,
}

/// Final transcription result
#[derive(Debug)]
pub struct FinalResult {
    // Complete transcription
    pub text: String,
    // Detected language
    pub language: String,
    // Total duration in seconds
    pub duration_seconds: f32,
}

/// Generate streaming audio chunks for testing
pub fn generate_audio_stream(duration_seconds: f32) -> Vec<Vec<f32>> {
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
pub fn build_streaming_bundle() -> Vec<u8> {
    let encoder_cache = vec![0u8; 768 * 512]; // Cached encoder states

    ModelBundleV2::new()
        .with_name("whisper-streaming-int4")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::Int4)
        .add_tensor("encoder.cache", vec![512, 768], encoder_cache)
        .build()
}
