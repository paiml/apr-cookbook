//! Whisper Speech Transcription Example
//!
//! Demonstrates speech-to-text transcription using the whisper.apr model format.
//!
//! # APR v2 Format Features
//!
//! - **LZ4 Compression**: Fast decompression (≥3 GB/s)
//! - **Int4 Quantization**: 8x smaller models with <2% accuracy loss
//! - **Ed25519 Signatures**: Cryptographic model verification
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    whisper.apr Pipeline                         │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Audio Input → Mel Spectrogram → Encoder → Decoder → Text      │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  trueno (SIMD)  │  aprender (ML)  │  realizar (inference)      │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example whisper_transcribe
//! ```
//!
//! # Recipe Metadata
//!
//! - **Category**: Speech Recognition
//! - **Complexity**: Intermediate
//! - **Dependencies**: trueno 0.11+, aprender 0.21+
//! - **IIUR**: Isolated, Idempotent, Useful, Reproducible
//!
//! ## References
//! - Radford, A. et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision*. ICML. arXiv:2212.04356

use apr_cookbook::prelude::*;
use std::f32::consts::PI;

/// Audio sample rate for Whisper models (16kHz)
const SAMPLE_RATE: usize = 16000;

/// Mel spectrogram bins
const N_MELS: usize = 80;

/// FFT window size
const N_FFT: usize = 400;

/// Hop length between frames
const HOP_LENGTH: usize = 160;

/// Maximum audio length in seconds
#[allow(dead_code)]
const MAX_AUDIO_SECONDS: usize = 30;

/// Simulated Whisper model for demonstration
#[derive(Debug)]
#[allow(dead_code)]
struct WhisperModel {
    /// Model name
    name: String,
    /// Quantization type
    quantization: Quantization,
    /// Number of encoder layers
    n_encoder_layers: usize,
    /// Number of decoder layers
    n_decoder_layers: usize,
    /// Hidden dimension
    d_model: usize,
    /// Vocabulary size
    vocab_size: usize,
}

impl WhisperModel {
    /// Create a new Whisper model configuration
    fn new(name: &str, quantization: Quantization) -> Self {
        // Whisper small configuration
        Self {
            name: name.to_string(),
            quantization,
            n_encoder_layers: 12,
            n_decoder_layers: 12,
            d_model: 768,
            vocab_size: 51865, // Whisper vocabulary size
        }
    }

    /// Load model from APR v2 bundle
    fn from_apr(bytes: &[u8]) -> Result<Self> {
        let bundle = BundledModelV2::from_bytes(bytes)?;

        // Verify signature if present
        if let Some(valid) = bundle.signature_valid() {
            if !valid {
                return Err(CookbookError::invalid_format("Invalid model signature"));
            }
        }

        Ok(Self::new("whisper-small", bundle.quantization()))
    }

    /// Get model size in bytes (estimated)
    fn size_bytes(&self) -> usize {
        let bytes_per_param = match self.quantization {
            Quantization::FP32 => 4,
            Quantization::FP16 => 2,
            Quantization::Int8 => 1,
            Quantization::Int4 => 1, // Actually 0.5, but we round up
        };

        // Whisper small has ~244M parameters
        244_000_000 * bytes_per_param
    }

    /// Transcribe audio samples
    fn transcribe(&self, audio: &[f32]) -> TranscriptionResult {
        // Step 1: Compute mel spectrogram
        let mel = self.compute_mel_spectrogram(audio);

        // Step 2: Run encoder (simulated)
        let encoder_output = self.run_encoder(&mel);

        // Step 3: Run decoder with beam search (simulated)
        let tokens = self.run_decoder(&encoder_output);

        // Step 4: Decode tokens to text
        let text = self.decode_tokens(&tokens);

        // Step 5: Detect language
        let language = self.detect_language(&encoder_output);

        TranscriptionResult {
            text,
            language,
            confidence: 0.95,
            segments: vec![Segment {
                start: 0.0,
                end: audio.len() as f32 / SAMPLE_RATE as f32,
                text: "Hello, world!".to_string(),
                confidence: 0.95,
            }],
        }
    }

    /// Compute mel spectrogram from audio samples
    fn compute_mel_spectrogram(&self, audio: &[f32]) -> Vec<Vec<f32>> {
        let n_frames = (audio.len() / HOP_LENGTH).saturating_sub(1);
        let mut mel = vec![vec![0.0f32; N_MELS]; n_frames];

        for (frame_idx, frame) in mel.iter_mut().enumerate().take(n_frames) {
            let start = frame_idx * HOP_LENGTH;
            let end = (start + N_FFT).min(audio.len());

            // Apply Hann window and compute power spectrum
            let mut power = vec![0.0f32; N_FFT / 2 + 1];
            for (i, p) in power.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                for (j, &sample) in audio[start..end].iter().enumerate() {
                    let window = 0.5 * (1.0 - (2.0 * PI * j as f32 / N_FFT as f32).cos());
                    let angle = 2.0 * PI * i as f32 * j as f32 / N_FFT as f32;
                    sum += sample * window * angle.cos();
                }
                *p = sum * sum;
            }

            // Apply mel filterbank (simplified)
            for (mel_idx, mel_val) in frame.iter_mut().enumerate() {
                let center_freq = 700.0 * (10.0f32.powf(mel_idx as f32 / 2595.0) - 1.0);
                let bin = (center_freq * N_FFT as f32 / SAMPLE_RATE as f32) as usize;
                if bin < power.len() {
                    *mel_val = (power[bin] + 1e-10).log10();
                }
            }
        }

        mel
    }

    /// Run encoder on mel spectrogram (simulated)
    fn run_encoder(&self, mel: &[Vec<f32>]) -> Vec<Vec<f32>> {
        // Simulated encoder output
        let n_frames = mel.len();
        vec![vec![0.0f32; self.d_model]; n_frames]
    }

    /// Run decoder with beam search (simulated)
    fn run_decoder(&self, _encoder_output: &[Vec<f32>]) -> Vec<u32> {
        // Simulated token sequence
        // <|startoftranscript|><|en|><|transcribe|>Hello, world!<|endoftext|>
        vec![50258, 50259, 50360, 7120, 11, 1002, 0, 50257]
    }

    /// Decode tokens to text
    fn decode_tokens(&self, tokens: &[u32]) -> String {
        // Simulated vocabulary lookup
        let vocab: std::collections::HashMap<u32, &str> = [
            (50258, "<|startoftranscript|>"),
            (50259, "<|en|>"),
            (50360, "<|transcribe|>"),
            (7120, "Hello"),
            (11, ","),
            (1002, " world"),
            (0, "!"),
            (50257, "<|endoftext|>"),
        ]
        .into_iter()
        .collect();

        tokens
            .iter()
            .filter_map(|t| vocab.get(t).copied())
            .filter(|s| !s.starts_with("<|"))
            .collect::<Vec<_>>()
            .join("")
    }

    /// Detect language from encoder output
    fn detect_language(&self, _encoder_output: &[Vec<f32>]) -> String {
        "en".to_string()
    }
}

/// Transcription result
#[derive(Debug)]
struct TranscriptionResult {
    /// Full transcription text
    text: String,
    /// Detected language (ISO 639-1)
    language: String,
    /// Overall confidence score
    confidence: f32,
    /// Time-aligned segments
    segments: Vec<Segment>,
}

/// A transcription segment with timing
#[derive(Debug)]
struct Segment {
    /// Start time in seconds
    start: f32,
    /// End time in seconds
    end: f32,
    /// Segment text
    text: String,
    /// Segment confidence
    confidence: f32,
}

/// Generate synthetic audio for testing
fn generate_test_audio(duration_seconds: f32, sample_rate: usize) -> Vec<f32> {
    let n_samples = (duration_seconds * sample_rate as f32) as usize;
    let mut samples = vec![0.0f32; n_samples];

    // Generate a simple sine wave with some harmonics
    for (i, sample) in samples.iter_mut().enumerate() {
        let t = i as f32 / sample_rate as f32;
        *sample = 0.5 * (440.0 * 2.0 * PI * t).sin()
            + 0.3 * (880.0 * 2.0 * PI * t).sin()
            + 0.2 * (1320.0 * 2.0 * PI * t).sin();
    }

    samples
}

/// Build a whisper.apr model bundle for testing
fn build_whisper_bundle() -> Vec<u8> {
    // Create encoder weights (simulated)
    let encoder_weights = vec![0u8; 1024 * 768 * 4]; // Embedding layer
    let decoder_weights = vec![0u8; 1024 * 768 * 4]; // Output layer

    ModelBundleV2::new()
        .with_name("whisper-small-int8")
        .with_description("Whisper small model quantized to Int8")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::Int8)
        .add_tensor("encoder.embed", vec![1024, 768], encoder_weights)
        .add_tensor("decoder.output", vec![1024, 768], decoder_weights)
        .build()
}

fn main() {
    println!("=== Whisper Speech Transcription Example ===\n");

    // =========================================================================
    // Section 1: Model Loading
    // =========================================================================
    println!("1. Model Loading");
    println!("   ─────────────────────────────────────────");

    let bundle = build_whisper_bundle();
    println!("   Bundle size: {} bytes", bundle.len());

    let loaded = BundledModelV2::from_bytes(&bundle).expect("Failed to load bundle");
    println!("   Format: APR v2");
    println!("   Compression: {:?}", loaded.compression());
    println!("   Quantization: {:?}", loaded.quantization());
    println!("   Tensors: {}", loaded.tensor_count());

    let model = WhisperModel::from_apr(&bundle).expect("Failed to create model");
    println!(
        "   Model size (estimated): {} MB",
        model.size_bytes() / 1_000_000
    );
    println!();

    // =========================================================================
    // Section 2: Audio Preprocessing
    // =========================================================================
    println!("2. Audio Preprocessing");
    println!("   ─────────────────────────────────────────");

    let audio = generate_test_audio(3.0, SAMPLE_RATE);
    println!("   Sample rate: {} Hz", SAMPLE_RATE);
    println!(
        "   Duration: {:.1} seconds",
        audio.len() as f32 / SAMPLE_RATE as f32
    );
    println!("   Samples: {}", audio.len());

    let mel = model.compute_mel_spectrogram(&audio);
    println!("   Mel spectrogram: {} frames x {} bins", mel.len(), N_MELS);
    println!();

    // =========================================================================
    // Section 3: Transcription
    // =========================================================================
    println!("3. Transcription");
    println!("   ─────────────────────────────────────────");

    let result = model.transcribe(&audio);
    println!("   Text: \"{}\"", result.text);
    println!("   Language: {}", result.language);
    println!("   Confidence: {:.1}%", result.confidence * 100.0);
    println!();

    // =========================================================================
    // Section 4: Segments
    // =========================================================================
    println!("4. Time-aligned Segments");
    println!("   ─────────────────────────────────────────");

    for (i, segment) in result.segments.iter().enumerate() {
        println!(
            "   [{}] {:.2}s - {:.2}s: \"{}\" ({:.1}%)",
            i,
            segment.start,
            segment.end,
            segment.text,
            segment.confidence * 100.0
        );
    }
    println!();

    // =========================================================================
    // Section 5: Model Comparison
    // =========================================================================
    println!("5. Model Size Comparison");
    println!("   ─────────────────────────────────────────");
    println!("   ┌─────────────┬──────────┬───────────┬───────────┐");
    println!("   │ Model       │ FP32     │ Int8      │ Int4      │");
    println!("   ├─────────────┼──────────┼───────────┼───────────┤");

    for (name, params) in [
        ("Tiny", 39_000_000_u64),
        ("Base", 74_000_000),
        ("Small", 244_000_000),
        ("Medium", 769_000_000),
        ("Large", 1_550_000_000),
    ] {
        let fp32 = params * 4 / 1_000_000;
        let int8 = params / 1_000_000;
        let int4 = params / 2_000_000;
        println!(
            "   │ {:11} │ {:5} MB │ {:6} MB │ {:6} MB │",
            name, fp32, int8, int4
        );
    }
    println!("   └─────────────┴──────────┴───────────┴───────────┘");
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_model_creation() {
        let model = WhisperModel::new("test", Quantization::Int8);
        assert_eq!(model.name, "test");
        assert_eq!(model.n_encoder_layers, 12);
        assert_eq!(model.d_model, 768);
    }

    #[test]
    fn test_whisper_model_size() {
        let fp32 = WhisperModel::new("test", Quantization::FP32);
        let int8 = WhisperModel::new("test", Quantization::Int8);
        assert!(fp32.size_bytes() > int8.size_bytes());
    }

    #[test]
    fn test_generate_test_audio() {
        let audio = generate_test_audio(1.0, SAMPLE_RATE);
        assert_eq!(audio.len(), SAMPLE_RATE);

        // Check amplitude range
        for sample in &audio {
            assert!(*sample >= -1.0 && *sample <= 1.0);
        }
    }

    #[test]
    fn test_mel_spectrogram() {
        let model = WhisperModel::new("test", Quantization::Int8);
        let audio = generate_test_audio(1.0, SAMPLE_RATE);
        let mel = model.compute_mel_spectrogram(&audio);

        // Check dimensions
        assert!(!mel.is_empty());
        assert_eq!(mel[0].len(), N_MELS);
    }

    #[test]
    fn test_transcription_result() {
        let model = WhisperModel::new("test", Quantization::Int8);
        let audio = generate_test_audio(1.0, SAMPLE_RATE);
        let result = model.transcribe(&audio);

        assert!(!result.text.is_empty());
        assert_eq!(result.language, "en");
        assert!(result.confidence > 0.0);
        assert!(!result.segments.is_empty());
    }

    #[test]
    fn test_apr_v2_bundle() {
        let bundle = build_whisper_bundle();
        let loaded = BundledModelV2::from_bytes(&bundle).unwrap();

        assert_eq!(loaded.compression(), Compression::Lz4);
        assert_eq!(loaded.quantization(), Quantization::Int8);
        assert_eq!(loaded.tensor_count(), 2);
    }

    #[test]
    fn test_model_from_apr() {
        let bundle = build_whisper_bundle();
        let model = WhisperModel::from_apr(&bundle).unwrap();

        assert_eq!(model.name, "whisper-small");
        assert_eq!(model.quantization, Quantization::Int8);
    }

    #[test]
    fn test_token_decoding() {
        let model = WhisperModel::new("test", Quantization::Int8);
        let tokens = vec![7120, 11, 1002, 0];
        let text = model.decode_tokens(&tokens);

        assert_eq!(text, "Hello, world!");
    }

    #[test]
    fn test_language_detection() {
        let model = WhisperModel::new("test", Quantization::Int8);
        let encoder_output = vec![vec![0.0f32; 768]; 10];
        let language = model.detect_language(&encoder_output);

        assert_eq!(language, "en");
    }

    #[test]
    fn test_max_audio_length() {
        let audio = generate_test_audio(MAX_AUDIO_SECONDS as f32, SAMPLE_RATE);
        assert_eq!(audio.len(), SAMPLE_RATE * MAX_AUDIO_SECONDS);
    }
}
