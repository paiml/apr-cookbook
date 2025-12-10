//! # Demo H: Voice Recognition Pipeline
//!
//! Simulates a speech-to-text pipeline with mel spectrogram processing
//! and CTC-style decoding. Demonstrates audio preprocessing concepts.
//!
//! ## Toyota Way Principles
//!
//! - **Jidoka**: Automatic silence/noise detection
//! - **Heijunka**: Consistent latency regardless of audio length
//! - **Genchi Genbutsu**: Process real audio patterns

use std::f32::consts::PI;

/// Audio sample rate (Hz)
pub const SAMPLE_RATE: u32 = 16000;

/// Mel spectrogram bins
pub const MEL_BINS: usize = 80;

/// FFT window size
pub const FFT_SIZE: usize = 512;

/// Hop length between frames
pub const HOP_LENGTH: usize = 160;

/// Vocabulary size (characters + blank + space)
pub const VOCAB_SIZE: usize = 29;

// ============================================================================
// Audio Processing
// ============================================================================

/// Raw audio samples
#[derive(Debug, Clone)]
pub struct AudioSamples {
    /// PCM samples (-1.0 to 1.0)
    pub samples: Vec<f32>,
    /// Sample rate
    pub sample_rate: u32,
}

impl AudioSamples {
    /// Create from samples
    #[must_use]
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Self {
        Self {
            samples,
            sample_rate,
        }
    }

    /// Duration in seconds
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32
    }

    /// Resample to target rate
    #[must_use]
    pub fn resample(&self, target_rate: u32) -> Self {
        if self.sample_rate == target_rate {
            return self.clone();
        }

        let ratio = target_rate as f32 / self.sample_rate as f32;
        let new_len = (self.samples.len() as f32 * ratio) as usize;
        let mut resampled = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_idx = i as f32 / ratio;
            let idx0 = src_idx.floor() as usize;
            let idx1 = (idx0 + 1).min(self.samples.len() - 1);
            let frac = src_idx - idx0 as f32;

            let sample = self.samples[idx0] * (1.0 - frac) + self.samples[idx1] * frac;
            resampled.push(sample);
        }

        Self::new(resampled, target_rate)
    }

    /// Normalize audio to [-1, 1]
    pub fn normalize(&mut self) {
        let max_abs = self.samples.iter().map(|s| s.abs()).fold(0.0_f32, f32::max);
        if max_abs > 0.0 {
            for s in &mut self.samples {
                *s /= max_abs;
            }
        }
    }

    /// Calculate RMS energy
    #[must_use]
    pub fn rms(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = self.samples.iter().map(|s| s * s).sum();
        (sum_sq / self.samples.len() as f32).sqrt()
    }

    /// Check if audio is mostly silence
    #[must_use]
    pub fn is_silence(&self, threshold: f32) -> bool {
        self.rms() < threshold
    }
}

/// Mel spectrogram frame
pub type MelFrame = [f32; MEL_BINS];

/// Mel spectrogram
#[derive(Debug, Clone)]
pub struct MelSpectrogram {
    /// Frames of mel bins
    pub frames: Vec<MelFrame>,
    /// Number of frames
    pub num_frames: usize,
}

impl MelSpectrogram {
    /// Create empty spectrogram
    #[must_use]
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            num_frames: 0,
        }
    }

    /// Create from audio
    #[must_use]
    pub fn from_audio(audio: &AudioSamples) -> Self {
        let processor = AudioProcessor::new();
        processor.compute_mel_spectrogram(audio)
    }

    /// Add a frame
    pub fn add_frame(&mut self, frame: MelFrame) {
        self.frames.push(frame);
        self.num_frames += 1;
    }

    /// Get frame at index
    #[must_use]
    pub fn get_frame(&self, idx: usize) -> Option<&MelFrame> {
        self.frames.get(idx)
    }

    /// Duration in seconds (approximate)
    #[must_use]
    pub fn duration(&self) -> f32 {
        self.num_frames as f32 * HOP_LENGTH as f32 / SAMPLE_RATE as f32
    }
}

impl Default for MelSpectrogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Audio preprocessor
pub struct AudioProcessor {
    /// Mel filterbank
    mel_filterbank: Vec<Vec<f32>>,
    /// Hann window
    window: Vec<f32>,
}

impl AudioProcessor {
    /// Create new processor
    #[must_use]
    pub fn new() -> Self {
        // Create Hann window
        let window: Vec<f32> = (0..FFT_SIZE)
            .map(|n| 0.5 * (1.0 - (2.0 * PI * n as f32 / (FFT_SIZE - 1) as f32).cos()))
            .collect();

        // Create mel filterbank (simplified)
        let mel_filterbank = create_mel_filterbank(FFT_SIZE / 2 + 1, MEL_BINS, SAMPLE_RATE);

        Self {
            mel_filterbank,
            window,
        }
    }

    /// Compute mel spectrogram from audio
    #[must_use]
    pub fn compute_mel_spectrogram(&self, audio: &AudioSamples) -> MelSpectrogram {
        let mut spec = MelSpectrogram::new();

        if audio.samples.len() < FFT_SIZE {
            return spec;
        }

        let num_frames = (audio.samples.len() - FFT_SIZE) / HOP_LENGTH + 1;

        for frame_idx in 0..num_frames {
            let start = frame_idx * HOP_LENGTH;
            let end = start + FFT_SIZE;

            if end > audio.samples.len() {
                break;
            }

            // Apply window
            let windowed: Vec<f32> = audio.samples[start..end]
                .iter()
                .zip(self.window.iter())
                .map(|(s, w)| s * w)
                .collect();

            // Compute power spectrum (simplified FFT approximation)
            let power_spectrum = compute_power_spectrum(&windowed);

            // Apply mel filterbank
            let mut mel_frame = [0.0_f32; MEL_BINS];
            for (mel_idx, filter) in self.mel_filterbank.iter().enumerate() {
                let energy: f32 = filter
                    .iter()
                    .zip(power_spectrum.iter())
                    .map(|(f, p)| f * p)
                    .sum();
                mel_frame[mel_idx] = (energy + 1e-10).ln();
            }

            spec.add_frame(mel_frame);
        }

        spec
    }

    /// Compute features for a single frame
    #[must_use]
    pub fn process_frame(&self, samples: &[f32]) -> Option<MelFrame> {
        if samples.len() < FFT_SIZE {
            return None;
        }

        let windowed: Vec<f32> = samples[..FFT_SIZE]
            .iter()
            .zip(self.window.iter())
            .map(|(s, w)| s * w)
            .collect();

        let power_spectrum = compute_power_spectrum(&windowed);

        let mut mel_frame = [0.0_f32; MEL_BINS];
        for (mel_idx, filter) in self.mel_filterbank.iter().enumerate() {
            let energy: f32 = filter
                .iter()
                .zip(power_spectrum.iter())
                .map(|(f, p)| f * p)
                .sum();
            mel_frame[mel_idx] = (energy + 1e-10).ln();
        }

        Some(mel_frame)
    }
}

impl Default for AudioProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(clippy::needless_range_loop)]
fn create_mel_filterbank(
    num_fft_bins: usize,
    num_mel_bins: usize,
    sample_rate: u32,
) -> Vec<Vec<f32>> {
    let mel_low = hz_to_mel(0.0);
    let mel_high = hz_to_mel(sample_rate as f32 / 2.0);

    let mel_points: Vec<f32> = (0..=num_mel_bins + 1)
        .map(|i| mel_low + (mel_high - mel_low) * i as f32 / (num_mel_bins + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&hz| ((hz / (sample_rate as f32 / 2.0)) * (num_fft_bins - 1) as f32) as usize)
        .collect();

    let mut filterbank = Vec::with_capacity(num_mel_bins);

    for i in 0..num_mel_bins {
        let mut filter = vec![0.0_f32; num_fft_bins];

        let start = bin_points[i];
        let center = bin_points[i + 1];
        let end = bin_points[i + 2];

        // Rising edge
        for k in start..center {
            if center > start {
                filter[k] = (k - start) as f32 / (center - start) as f32;
            }
        }

        // Falling edge
        for k in center..end {
            if end > center {
                filter[k] = (end - k) as f32 / (end - center) as f32;
            }
        }

        filterbank.push(filter);
    }

    filterbank
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

#[allow(clippy::needless_range_loop)]
fn compute_power_spectrum(samples: &[f32]) -> Vec<f32> {
    // Simplified DFT (real implementation would use FFT)
    let n = samples.len();
    let num_bins = n / 2 + 1;
    let mut spectrum = vec![0.0_f32; num_bins];

    for k in 0..num_bins {
        let mut real = 0.0_f32;
        let mut imag = 0.0_f32;

        for (n_idx, &sample) in samples.iter().enumerate() {
            let angle = -2.0 * PI * k as f32 * n_idx as f32 / n as f32;
            real += sample * angle.cos();
            imag += sample * angle.sin();
        }

        spectrum[k] = real * real + imag * imag;
    }

    spectrum
}

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
    min_prob: f32,
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
    processor: AudioProcessor,
    decoder: CTCDecoder,
    /// Seed for deterministic simulation
    seed: u64,
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

    fn simulate_model_output(&self, mel: &MelSpectrogram) -> Vec<FrameProbs> {
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
    rng: SimpleRng,
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

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f64 / u64::MAX as f64) as f32
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("=== Demo H: Voice Recognition Pipeline ===\n");

    let mut generator = AudioGenerator::new(42);

    // Generate different audio types
    let silence = AudioGenerator::silence(1.0, SAMPLE_RATE);
    let sine = generator.sine_wave(440.0, 1.0, SAMPLE_RATE);
    let speech = generator.speech_like(2.0, SAMPLE_RATE);

    let recognizer = VoiceRecognizer::new(42);

    println!("--- Processing Silence ---");
    let result = recognizer.recognize(&silence);
    println!(
        "Text: \"{}\" (conf: {:.2})\n",
        result.text, result.confidence
    );

    println!("--- Processing Sine Wave (440 Hz) ---");
    let result = recognizer.recognize(&sine);
    println!(
        "Text: \"{}\" (conf: {:.2})\n",
        result.text, result.confidence
    );

    println!("--- Processing Speech-like Signal ---");
    let result = recognizer.recognize(&speech);
    println!("Text: \"{}\" (conf: {:.2})", result.text, result.confidence);
    println!("Words: {}", result.word_count());

    println!("\n--- Mel Spectrogram Stats ---");
    let mel = MelSpectrogram::from_audio(&speech);
    println!("Frames: {}", mel.num_frames);
    println!("Duration: {:.2}s", mel.duration());

    println!("\n=== Demo H Complete ===");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_samples_new() {
        let audio = AudioSamples::new(vec![0.0; 100], 16000);
        assert_eq!(audio.samples.len(), 100);
        assert_eq!(audio.sample_rate, 16000);
    }

    #[test]
    fn test_audio_duration() {
        let audio = AudioSamples::new(vec![0.0; 16000], 16000);
        assert!((audio.duration() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_audio_rms_silence() {
        let audio = AudioSamples::new(vec![0.0; 100], 16000);
        assert!((audio.rms() - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_audio_rms_signal() {
        let audio = AudioSamples::new(vec![1.0; 100], 16000);
        assert!((audio.rms() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_audio_is_silence() {
        let silence = AudioSamples::new(vec![0.0; 100], 16000);
        assert!(silence.is_silence(0.01));

        let loud = AudioSamples::new(vec![0.5; 100], 16000);
        assert!(!loud.is_silence(0.01));
    }

    #[test]
    fn test_audio_normalize() {
        let mut audio = AudioSamples::new(vec![0.0, 0.5, 1.0, -0.5], 16000);
        audio.normalize();
        assert!((audio.samples[2] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_audio_resample() {
        let audio = AudioSamples::new(vec![0.0; 16000], 16000);
        let resampled = audio.resample(8000);
        assert_eq!(resampled.sample_rate, 8000);
        assert_eq!(resampled.samples.len(), 8000);
    }

    #[test]
    fn test_mel_spectrogram_new() {
        let mel = MelSpectrogram::new();
        assert_eq!(mel.num_frames, 0);
    }

    #[test]
    fn test_mel_spectrogram_add_frame() {
        let mut mel = MelSpectrogram::new();
        mel.add_frame([0.0; MEL_BINS]);
        assert_eq!(mel.num_frames, 1);
    }

    #[test]
    fn test_mel_from_audio() {
        let audio = AudioSamples::new(vec![0.1; 2000], SAMPLE_RATE);
        let mel = MelSpectrogram::from_audio(&audio);
        assert!(mel.num_frames > 0);
    }

    #[test]
    fn test_audio_processor_new() {
        let processor = AudioProcessor::new();
        assert_eq!(processor.window.len(), FFT_SIZE);
    }

    #[test]
    fn test_hz_to_mel() {
        assert!((hz_to_mel(0.0) - 0.0).abs() < 0.1);
        assert!(hz_to_mel(1000.0) > 0.0);
    }

    #[test]
    fn test_mel_to_hz() {
        let hz = 1000.0;
        let mel = hz_to_mel(hz);
        let back = mel_to_hz(mel);
        assert!((back - hz).abs() < 1.0);
    }

    #[test]
    fn test_ctc_decoder_new() {
        let decoder = CTCDecoder::new();
        assert!((decoder.min_prob - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_ctc_decoder_empty() {
        let decoder = CTCDecoder::new();
        let result = decoder.decode(&[]);
        assert!(result.text.is_empty());
    }

    #[test]
    fn test_ctc_decoder_blank_only() {
        let decoder = CTCDecoder::new();
        let mut frame = [0.0_f32; VOCAB_SIZE];
        frame[BLANK_TOKEN] = 1.0;
        let result = decoder.decode(&[frame]);
        assert!(result.text.is_empty());
    }

    #[test]
    fn test_ctc_decoder_character() {
        let decoder = CTCDecoder::new();
        let mut frame = [0.0_f32; VOCAB_SIZE];
        frame[1] = 1.0; // 'a'
        let result = decoder.decode(&[frame]);
        assert_eq!(result.text, "a");
    }

    #[test]
    fn test_transcription_new() {
        let t = Transcription::new("hello", 0.9);
        assert_eq!(t.text, "hello");
        assert!((t.confidence - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_transcription_word_count() {
        let t = Transcription::new("hello world test", 0.9);
        assert_eq!(t.word_count(), 3);
    }

    #[test]
    fn test_voice_recognizer_new() {
        let recognizer = VoiceRecognizer::new(42);
        // Just verify it creates successfully
        assert!(true);
    }

    #[test]
    fn test_voice_recognizer_silence() {
        let recognizer = VoiceRecognizer::new(42);
        let silence = AudioSamples::new(vec![0.0; 16000], 16000);
        let result = recognizer.recognize(&silence);
        assert!(result.text.is_empty());
    }

    #[test]
    fn test_audio_generator_sine() {
        let generator = AudioGenerator::new(42);
        let audio = generator.sine_wave(440.0, 1.0, 16000);
        assert_eq!(audio.samples.len(), 16000);
        assert!(!audio.is_silence(0.01));
    }

    #[test]
    fn test_audio_generator_silence() {
        let audio = AudioGenerator::silence(1.0, 16000);
        assert!(audio.is_silence(0.01));
    }

    #[test]
    fn test_audio_generator_noise() {
        let mut generator = AudioGenerator::new(42);
        let audio = generator.white_noise(1.0, 16000);
        assert!(!audio.is_silence(0.01));
    }

    #[test]
    fn test_vocab_size() {
        assert_eq!(VOCAB.len(), VOCAB_SIZE);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(30))]

        #[test]
        fn prop_audio_duration_positive(len in 100usize..10000, rate in 8000u32..48000) {
            let audio = AudioSamples::new(vec![0.0; len], rate);
            prop_assert!(audio.duration() > 0.0);
        }

        #[test]
        fn prop_rms_non_negative(len in 10usize..1000) {
            let samples: Vec<f32> = (0..len).map(|i| (i as f32 * 0.01).sin()).collect();
            let audio = AudioSamples::new(samples, 16000);
            prop_assert!(audio.rms() >= 0.0);
        }

        #[test]
        fn prop_mel_hz_roundtrip(hz in 20.0f32..8000.0) {
            let mel = hz_to_mel(hz);
            let back = mel_to_hz(mel);
            prop_assert!((back - hz).abs() < 1.0);
        }

        #[test]
        fn prop_resample_changes_length(len in 1000usize..5000, ratio in 1u32..3) {
            let audio = AudioSamples::new(vec![0.0; len], 16000);
            let target = 16000 * ratio;
            let resampled = audio.resample(target);
            let expected_len = (len as f32 * ratio as f32) as usize;
            prop_assert!((resampled.samples.len() as i32 - expected_len as i32).abs() <= 1);
        }

        #[test]
        fn prop_ctc_decode_deterministic(seed in 0u64..1000) {
            let decoder = CTCDecoder::new();
            let mut rng = SimpleRng::new(seed);
            let mut frame = [0.0_f32; VOCAB_SIZE];
            for p in &mut frame {
                *p = rng.next_f32();
            }
            let r1 = decoder.decode(&[frame]);
            let r2 = decoder.decode(&[frame]);
            prop_assert_eq!(r1.text, r2.text);
        }

        #[test]
        fn prop_transcription_word_count(n in 1usize..10) {
            let words: Vec<&str> = vec!["hello", "world", "test", "foo", "bar", "baz", "qux", "abc", "def", "ghi"];
            let text = words[..n].join(" ");
            let t = Transcription::new(&text, 0.9);
            prop_assert_eq!(t.word_count(), n);
        }
    }
}
