#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports, clippy::wildcard_imports)]
use super::types::*;

use std::f32::consts::PI;

// ============================================================================
// Decoder (CTC-style)
// ============================================================================

/// Character vocabulary
pub const VOCAB: &[char] = &[
    ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
    's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '\'', '_',
];

/// CTC blank token index
pub const BLANK_TOKEN: usize = 28;

/// Decoder output probabilities
pub type FrameProbs = [f32; VOCAB_SIZE];

/// Transcription result
#[derive(Debug, Clone)]
pub struct Transcription {
    /// Decoded text
    pub text: String,
    /// Confidence score
    pub confidence: f32,
    /// Per-character confidences
    pub char_confidences: Vec<f32>,
}

impl Transcription {
    /// Create new transcription
    #[must_use]
    pub fn new(text: &str, confidence: f32) -> Self {
        Self {
            text: text.to_string(),
            confidence,
            char_confidences: Vec::new(),
        }
    }

    /// Word count
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }
}

/// Greedy CTC decoder
pub struct CTCDecoder {
    /// Minimum probability to consider
    pub min_prob: f32,
}

impl CTCDecoder {
    /// Create new decoder
    #[must_use]
    pub fn new() -> Self {
        Self { min_prob: 0.0 }
    }

    /// Set minimum probability threshold
    #[must_use]
    pub fn with_min_prob(mut self, min_prob: f32) -> Self {
        self.min_prob = min_prob;
        self
    }

    /// Decode frame probabilities to text
    #[must_use]
    pub fn decode(&self, frame_probs: &[FrameProbs]) -> Transcription {
        if frame_probs.is_empty() {
            return Transcription::new("", 0.0);
        }

        let mut result = String::new();
        let mut confidences = Vec::new();
        let mut prev_token = BLANK_TOKEN;
        let mut total_confidence = 0.0_f32;
        let mut num_chars = 0;

        for probs in frame_probs {
            // Find best token
            let (best_token, best_prob) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap_or((BLANK_TOKEN, &0.0));

            if *best_prob < self.min_prob {
                prev_token = BLANK_TOKEN;
                continue;
            }

            // CTC rules: emit character if not blank and different from previous
            if best_token != BLANK_TOKEN && best_token != prev_token && best_token < VOCAB.len() {
                result.push(VOCAB[best_token]);
                confidences.push(*best_prob);
                total_confidence += *best_prob;
                num_chars += 1;
            }

            prev_token = best_token;
        }

        let avg_confidence = if num_chars > 0 {
            total_confidence / num_chars as f32
        } else {
            0.0
        };

        let mut transcription = Transcription::new(&result, avg_confidence);
        transcription.char_confidences = confidences;
        transcription
    }
}

impl Default for CTCDecoder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Voice Recognition Model (Simulated)
// ============================================================================

/// Simulated voice recognition model
pub struct VoiceRecognizer {
    pub processor: AudioProcessor,
    pub decoder: CTCDecoder,
    /// Seed for deterministic simulation
    pub seed: u64,
}

impl VoiceRecognizer {
    /// Create new recognizer
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            processor: AudioProcessor::new(),
            decoder: CTCDecoder::new(),
            seed,
        }
    }

    /// Recognize speech from audio
    #[must_use]
    pub fn recognize(&self, audio: &AudioSamples) -> Transcription {
        // Check for silence
        if audio.is_silence(0.01) {
            return Transcription::new("", 1.0);
        }

        // Compute mel spectrogram
        let mel = self.processor.compute_mel_spectrogram(audio);

        if mel.num_frames == 0 {
            return Transcription::new("", 0.0);
        }

        // Simulate model output (in real implementation, this would be neural network)
        let frame_probs = self.simulate_model_output(&mel);

        // Decode to text
        self.decoder.decode(&frame_probs)
    }

    pub fn simulate_model_output(&self, mel: &MelSpectrogram) -> Vec<FrameProbs> {
        let mut rng = SimpleRng::new(self.seed);
        let mut probs = Vec::with_capacity(mel.num_frames);

        // Simulate output based on mel energy patterns
        for frame in &mel.frames {
            let mut frame_prob = [0.0_f32; VOCAB_SIZE];

            // Energy-based simulation
            let energy: f32 = frame.iter().map(|&x| x.exp()).sum();
            let is_speech = energy > 100.0;

            if is_speech {
                // Generate plausible character distribution
                let dominant_char = ((frame[0].abs() * 10.0) as usize) % 27;
                for (i, p) in frame_prob.iter_mut().enumerate() {
                    if i == dominant_char {
                        *p = 0.6 + rng.next_f32() * 0.3;
                    } else if i == BLANK_TOKEN {
                        *p = 0.1;
                    } else {
                        *p = rng.next_f32() * 0.1;
                    }
                }
            } else {
                // Silence → blank token
                frame_prob[BLANK_TOKEN] = 0.95;
                for (i, p) in frame_prob.iter_mut().enumerate() {
                    if i != BLANK_TOKEN {
                        *p = rng.next_f32() * 0.02;
                    }
                }
            }

            // Normalize to sum to 1
            let sum: f32 = frame_prob.iter().sum();
            if sum > 0.0 {
                for p in &mut frame_prob {
                    *p /= sum;
                }
            }

            probs.push(frame_prob);
        }

        probs
    }

    /// Recognize with streaming (chunk by chunk)
    pub fn recognize_streaming(
        &self,
        audio: &AudioSamples,
        chunk_size: usize,
    ) -> Vec<Transcription> {
        let mut results = Vec::new();

        for chunk_start in (0..audio.samples.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(audio.samples.len());
            let chunk_samples = audio.samples[chunk_start..chunk_end].to_vec();
            let chunk = AudioSamples::new(chunk_samples, audio.sample_rate);

            let transcription = self.recognize(&chunk);
            if !transcription.text.is_empty() {
                results.push(transcription);
            }
        }

        results
    }
}

// ============================================================================
// Audio Generation (for testing)
// ============================================================================

/// Generate synthetic audio for testing
pub struct AudioGenerator {
    pub rng: SimpleRng,
}

impl AudioGenerator {
    /// Create new generator
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SimpleRng::new(seed),
        }
    }

    /// Generate sine wave
    #[must_use]
    pub fn sine_wave(&self, freq: f32, duration: f32, sample_rate: u32) -> AudioSamples {
        let num_samples = (duration * sample_rate as f32) as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * PI * freq * t).sin() * 0.5
            })
            .collect();
        AudioSamples::new(samples, sample_rate)
    }

    /// Generate white noise
    pub fn white_noise(&mut self, duration: f32, sample_rate: u32) -> AudioSamples {
        let num_samples = (duration * sample_rate as f32) as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|_| self.rng.next_f32() * 2.0 - 1.0)
            .collect();
        AudioSamples::new(samples, sample_rate)
    }

    /// Generate silence
    #[must_use]
    pub fn silence(duration: f32, sample_rate: u32) -> AudioSamples {
        let num_samples = (duration * sample_rate as f32) as usize;
        AudioSamples::new(vec![0.0; num_samples], sample_rate)
    }

    /// Generate speech-like signal (amplitude modulated noise)
    pub fn speech_like(&mut self, duration: f32, sample_rate: u32) -> AudioSamples {
        let num_samples = (duration * sample_rate as f32) as usize;
        let samples: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                // Modulated noise
                let envelope = (2.0 * PI * 4.0 * t).sin().abs();
                let noise = self.rng.next_f32() * 2.0 - 1.0;
                envelope * noise * 0.5
            })
            .collect();
        AudioSamples::new(samples, sample_rate)
    }
}

// ============================================================================
// Utilities
// ============================================================================

pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }
}
