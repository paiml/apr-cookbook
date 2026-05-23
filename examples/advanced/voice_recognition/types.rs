#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
//! # Demo H: Voice Recognition Pipeline
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Simulates a speech-to-text pipeline with mel spectrogram processing
//! and CTC-style decoding. Demonstrates audio preprocessing concepts.
//!
//! ## Toyota Way Principles
//!
//! - **Jidoka**: Automatic silence/noise detection
//! - **Heijunka**: Consistent latency regardless of audio length
//! - **Genchi Genbutsu**: Process real audio patterns
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
    pub fn add_frame(&mut self, frame: &MelFrame) {
        self.frames.push(*frame);
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
    pub mel_filterbank: Vec<Vec<f32>>,
    /// Hann window
    pub window: Vec<f32>,
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

            spec.add_frame(&mel_frame);
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
pub fn create_mel_filterbank(
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

pub fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

pub fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

#[allow(clippy::needless_range_loop)]
pub fn compute_power_spectrum(samples: &[f32]) -> Vec<f32> {
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
