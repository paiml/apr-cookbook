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
use rand::Rng;
use rand::SeedableRng;
use std::f32::consts::PI;

/// Audio sample rate (16kHz, standard for speech)
pub const SAMPLE_RATE: usize = 16000;

/// Total audio duration in seconds
pub const AUDIO_DURATION_SECS: usize = 10;

/// Total number of audio samples
pub const TOTAL_SAMPLES: usize = SAMPLE_RATE * AUDIO_DURATION_SECS;

/// Frame size in samples (30ms at 16kHz)
pub const FRAME_SIZE: usize = 480;

/// VAD configuration parameters.
#[derive(Debug, Clone)]
pub struct VadConfig {
    // Number of samples per analysis frame
    pub frame_size: usize,
    // Minimum RMS energy to consider as speech
    pub energy_threshold: f32,
    // Maximum zero-crossing rate for speech (noise has higher ZCR)
    pub zc_threshold: f32,
    // Window size for median smoothing (must be odd)
    pub smooth_window: usize,
}

impl VadConfig {
    /// Create a default VAD configuration tuned for 16kHz audio.
    pub fn default_16khz() -> Self {
        Self {
            frame_size: FRAME_SIZE,
            energy_threshold: 0.05,
            zc_threshold: 0.45,
            smooth_window: 5,
        }
    }
}

/// Per-frame VAD analysis result.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct VadFrame {
    // Frame index (0-based)
    pub index: usize,
    // RMS energy of the frame
    pub energy: f32,
    // Zero-crossing rate (0.0 to 1.0)
    pub zero_crossings: f32,
    // Whether this frame is classified as speech
    pub is_speech: bool,
}

/// A detected speech segment with timing information.
#[derive(Debug, Clone)]
pub struct SpeechSegment {
    // Start time in milliseconds
    pub start_ms: f64,
    // End time in milliseconds
    pub end_ms: f64,
    // Duration in milliseconds
    pub duration_ms: f64,
    // Average RMS energy across segment frames
    pub avg_energy: f32,
}

// Generate a synthetic 10-second audio signal with known speech/silence regions.
//
// Layout:
//   0-2s: silence (noise floor ~0.01)
//   2-5s: speech (sine + harmonics + noise)
//   5-6s: silence
//   6-9s: speech
///   9-10s: silence
pub fn generate_synthetic_audio(rng: &mut impl Rng) -> Vec<f32> {
    let mut samples = Vec::with_capacity(TOTAL_SAMPLES);

    for i in 0..TOTAL_SAMPLES {
        let t = i as f32 / SAMPLE_RATE as f32;
        let second = t as usize;

        let sample = match second {
            // Silence regions: low-amplitude noise
            0..=1 | 5 | 9 => rng.gen_range(-0.01..0.01),
            // Speech regions: fundamental + harmonics + noise
            2..=4 | 6..=8 => {
                let fundamental = 0.3 * (150.0 * 2.0 * PI * t).sin();
                let harmonic2 = 0.15 * (300.0 * 2.0 * PI * t).sin();
                let harmonic3 = 0.08 * (450.0 * 2.0 * PI * t).sin();
                let noise = rng.gen_range(-0.05..0.05);
                fundamental + harmonic2 + harmonic3 + noise
            }
            _ => rng.gen_range(-0.01..0.01),
        };

        samples.push(sample.clamp(-1.0, 1.0));
    }

    samples
}

/// Compute RMS (root mean square) energy for a frame of audio samples.
pub fn compute_rms_energy(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = frame.iter().map(|&s| s * s).sum();
    (sum_sq / frame.len() as f32).sqrt()
}

/// Compute zero-crossing rate for a frame (ratio of sign changes to total transitions).
pub fn compute_zero_crossing_rate(frame: &[f32]) -> f32 {
    if frame.len() < 2 {
        return 0.0;
    }
    let crossings = frame
        .windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    crossings as f32 / (frame.len() - 1) as f32
}

/// Analyze all frames in the audio signal, producing raw VAD decisions.
pub fn analyze_frames(audio: &[f32], config: &VadConfig) -> Vec<VadFrame> {
    let n_frames = audio.len() / config.frame_size;
    let mut frames = Vec::with_capacity(n_frames);

    for i in 0..n_frames {
        let start = i * config.frame_size;
        let end = start + config.frame_size;
        let frame_data = &audio[start..end];

        let energy = compute_rms_energy(frame_data);
        let zcr = compute_zero_crossing_rate(frame_data);

        let is_speech = energy > config.energy_threshold && zcr < config.zc_threshold;

        frames.push(VadFrame {
            index: i,
            energy,
            zero_crossings: zcr,
            is_speech,
        });
    }

    frames
}

// Apply median smoothing to VAD decisions to remove spurious transitions.
//
// Uses a sliding window of `window_size` frames. Each frame's decision is
/// replaced by the majority vote within its window.
pub fn smooth_predictions(frames: &mut [VadFrame], window_size: usize) {
    if frames.is_empty() {
        return;
    }

    let half = window_size / 2;
    let threshold = window_size / 2 + 1;

    // Snapshot the raw decisions before smoothing
    let raw: Vec<bool> = frames.iter().map(|f| f.is_speech).collect();

    for (i, frame) in frames.iter_mut().enumerate() {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(raw.len());

        let speech_count = raw[lo..hi].iter().filter(|&&v| v).count();
        frame.is_speech = speech_count >= threshold;
    }
}

/// Detect contiguous speech segments from smoothed frame decisions.
pub fn detect_segments(frames: &[VadFrame], config: &VadConfig) -> Vec<SpeechSegment> {
    let ms_per_frame = (config.frame_size as f64 / SAMPLE_RATE as f64) * 1000.0;
    let mut segments = Vec::new();
    let mut seg_start: Option<usize> = None;

    for (i, frame) in frames.iter().enumerate() {
        match (seg_start, frame.is_speech) {
            (None, true) => seg_start = Some(i),
            (Some(start), false) => {
                let avg_energy = compute_segment_avg_energy(frames, start, i);
                let start_ms = start as f64 * ms_per_frame;
                let end_ms = i as f64 * ms_per_frame;
                segments.push(SpeechSegment {
                    start_ms,
                    end_ms,
                    duration_ms: end_ms - start_ms,
                    avg_energy,
                });
                seg_start = None;
            }
            _ => {}
        }
    }

    // Handle segment that extends to the end
    if let Some(start) = seg_start {
        let end = frames.len();
        let avg_energy = compute_segment_avg_energy(frames, start, end);
        let start_ms = start as f64 * ms_per_frame;
        let end_ms = end as f64 * ms_per_frame;
        segments.push(SpeechSegment {
            start_ms,
            end_ms,
            duration_ms: end_ms - start_ms,
            avg_energy,
        });
    }

    segments
}

/// Compute average energy for a range of frames.
pub fn compute_segment_avg_energy(frames: &[VadFrame], start: usize, end: usize) -> f32 {
    let slice = &frames[start..end];
    if slice.is_empty() {
        return 0.0;
    }
    let total: f32 = slice.iter().map(|f| f.energy).sum();
    total / slice.len() as f32
}

// Render a timeline visualization string from frame decisions.
//
/// Each character represents one frame: `S` = speech, `.` = silence.
pub fn render_timeline(frames: &[VadFrame], width: usize) -> String {
    if frames.is_empty() {
        return String::new();
    }

    let ratio = frames.len() as f64 / width as f64;
    let mut timeline = String::with_capacity(width + 2);
    timeline.push('[');

    for col in 0..width {
        let frame_start = (col as f64 * ratio) as usize;
        let frame_end = (((col + 1) as f64 * ratio) as usize).min(frames.len());

        let speech_count = frames[frame_start..frame_end]
            .iter()
            .filter(|f| f.is_speech)
            .count();
        let total = frame_end - frame_start;

        if total > 0 && speech_count * 2 >= total {
            timeline.push('S');
        } else {
            timeline.push('.');
        }
    }

    timeline.push(']');
    timeline
}
