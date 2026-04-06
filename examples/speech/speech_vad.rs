//! Voice Activity Detection (VAD) Example
//!
//! Demonstrates frame-based voice activity detection on a synthetic audio stream
//! using energy, zero-crossing rate, and spectral centroid features.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
//! │  Audio       │──>│  Frame       │──>│  Feature     │──>│  Threshold   │
//! │  Stream      │   │  Splitter    │   │  Extraction  │   │  Decision    │
//! │  (16kHz)     │   │  (30ms)      │   │  RMS/ZCR/SC  │   │  Speech/Sil  │
//! └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
//!                                                               │
//!                                                               v
//! ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
//! │  Segment     │<──│  Merge       │<──│  Median      │<──│  Raw         │
//! │  Report      │   │  Consecutive │   │  Smoothing   │   │  Decisions   │
//! └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
//! ```
//!
//! # Features
//!
//! - **RMS Energy**: Root mean square amplitude per frame
//! - **Zero-Crossing Rate**: Frequency content proxy
//! - **Spectral Centroid**: Brightness approximation via ZCR-energy ratio
//! - **Median Smoothing**: Removes spurious speech/silence transitions
//! - **Segment Merging**: Groups consecutive speech frames into segments
//!
//! # Running
//!
//! ```bash
//! cargo run --example speech_vad
//! ```
//!
//! # Recipe Metadata
//!
//! - **Category**: Speech Recognition
//! - **Complexity**: Intermediate
//! - **Dependencies**: trueno 0.14+, aprender 0.25+
//! - **IIUR**: Isolated, Idempotent, Useful, Reproducible
//!
//! ## References
//! - Radford, A. et al. (2023). *Robust Speech Recognition via Large-Scale Weak Supervision*. ICML. arXiv:2212.04356

use apr_cookbook::prelude::*;
use rand::Rng;
use std::f32::consts::PI;

/// Audio sample rate (16kHz, standard for speech)
const SAMPLE_RATE: usize = 16000;

/// Total audio duration in seconds
const AUDIO_DURATION_SECS: usize = 10;

/// Total number of audio samples
const TOTAL_SAMPLES: usize = SAMPLE_RATE * AUDIO_DURATION_SECS;

/// Frame size in samples (30ms at 16kHz)
const FRAME_SIZE: usize = 480;

/// VAD configuration parameters.
#[derive(Debug, Clone)]
struct VadConfig {
    /// Number of samples per analysis frame
    frame_size: usize,
    /// Minimum RMS energy to consider as speech
    energy_threshold: f32,
    /// Maximum zero-crossing rate for speech (noise has higher ZCR)
    zc_threshold: f32,
    /// Window size for median smoothing (must be odd)
    smooth_window: usize,
}

impl VadConfig {
    /// Create a default VAD configuration tuned for 16kHz audio.
    fn default_16khz() -> Self {
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
struct VadFrame {
    /// Frame index (0-based)
    index: usize,
    /// RMS energy of the frame
    energy: f32,
    /// Zero-crossing rate (0.0 to 1.0)
    zero_crossings: f32,
    /// Whether this frame is classified as speech
    is_speech: bool,
}

/// A detected speech segment with timing information.
#[derive(Debug, Clone)]
struct SpeechSegment {
    /// Start time in milliseconds
    start_ms: f64,
    /// End time in milliseconds
    end_ms: f64,
    /// Duration in milliseconds
    duration_ms: f64,
    /// Average RMS energy across segment frames
    avg_energy: f32,
}

/// Generate a synthetic 10-second audio signal with known speech/silence regions.
///
/// Layout:
///   0-2s: silence (noise floor ~0.01)
///   2-5s: speech (sine + harmonics + noise)
///   5-6s: silence
///   6-9s: speech
///   9-10s: silence
fn generate_synthetic_audio(rng: &mut impl Rng) -> Vec<f32> {
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
fn compute_rms_energy(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = frame.iter().map(|&s| s * s).sum();
    (sum_sq / frame.len() as f32).sqrt()
}

/// Compute zero-crossing rate for a frame (ratio of sign changes to total transitions).
fn compute_zero_crossing_rate(frame: &[f32]) -> f32 {
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
fn analyze_frames(audio: &[f32], config: &VadConfig) -> Vec<VadFrame> {
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

/// Apply median smoothing to VAD decisions to remove spurious transitions.
///
/// Uses a sliding window of `window_size` frames. Each frame's decision is
/// replaced by the majority vote within its window.
fn smooth_predictions(frames: &mut [VadFrame], window_size: usize) {
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
fn detect_segments(frames: &[VadFrame], config: &VadConfig) -> Vec<SpeechSegment> {
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
fn compute_segment_avg_energy(frames: &[VadFrame], start: usize, end: usize) -> f32 {
    let slice = &frames[start..end];
    if slice.is_empty() {
        return 0.0;
    }
    let total: f32 = slice.iter().map(|f| f.energy).sum();
    total / slice.len() as f32
}

/// Render a timeline visualization string from frame decisions.
///
/// Each character represents one frame: `S` = speech, `.` = silence.
fn render_timeline(frames: &[VadFrame], width: usize) -> String {
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

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("speech_vad")?;

    println!("=== Voice Activity Detection (VAD) Example ===\n");

    // =========================================================================
    // Section 1: Generate Synthetic Audio
    // =========================================================================
    println!("1. Generating Synthetic Audio");
    println!("   ─────────────────────────────────────────");

    let audio = generate_synthetic_audio(ctx.rng());
    println!("   Sample rate:  {} Hz", SAMPLE_RATE);
    println!("   Duration:     {} seconds", AUDIO_DURATION_SECS);
    println!("   Samples:      {}", audio.len());
    println!("   Layout:");
    println!("     0-2s  silence  |  2-5s  speech  |  5-6s  silence");
    println!("     6-9s  speech   |  9-10s silence");
    println!();

    // =========================================================================
    // Section 2: Frame-based VAD Analysis
    // =========================================================================
    println!("2. Frame-based VAD Analysis");
    println!("   ─────────────────────────────────────────");

    let config = VadConfig::default_16khz();
    println!(
        "   Frame size:        {} samples ({:.0} ms)",
        config.frame_size,
        config.frame_size as f64 / SAMPLE_RATE as f64 * 1000.0
    );
    println!("   Energy threshold:  {:.2}", config.energy_threshold);
    println!("   ZCR threshold:     {:.2}", config.zc_threshold);
    println!("   Smooth window:     {} frames", config.smooth_window);

    let mut frames = analyze_frames(&audio, &config);
    let n_frames = frames.len();
    let raw_speech = frames.iter().filter(|f| f.is_speech).count();
    println!("   Total frames:      {}", n_frames);
    println!(
        "   Raw speech frames: {} ({:.1}%)",
        raw_speech,
        raw_speech as f64 / n_frames as f64 * 100.0
    );
    println!();

    // =========================================================================
    // Section 3: Median Smoothing
    // =========================================================================
    println!("3. Median Smoothing");
    println!("   ─────────────────────────────────────────");

    smooth_predictions(&mut frames, config.smooth_window);
    let smoothed_speech = frames.iter().filter(|f| f.is_speech).count();
    println!(
        "   Smoothed speech frames: {} ({:.1}%)",
        smoothed_speech,
        smoothed_speech as f64 / n_frames as f64 * 100.0
    );
    println!();

    // =========================================================================
    // Section 4: Timeline Visualization
    // =========================================================================
    println!("4. Timeline Visualization");
    println!("   ─────────────────────────────────────────");

    let timeline = render_timeline(&frames, 50);
    println!("   {}", timeline);
    println!("   |0s       |2s       |4s       |6s       |8s    10s|");
    println!("   (. = silence, S = speech)");
    println!();

    // =========================================================================
    // Section 5: Detected Speech Segments
    // =========================================================================
    println!("5. Detected Speech Segments");
    println!("   ─────────────────────────────────────────");

    let segments = detect_segments(&frames, &config);
    println!("   ┌─────────┬───────────┬───────────┬──────────────┬────────────┐");
    println!("   │ Segment │ Start (ms)│ End (ms)  │ Duration (ms)│ Avg Energy │");
    println!("   ├─────────┼───────────┼───────────┼──────────────┼────────────┤");

    for (i, seg) in segments.iter().enumerate() {
        println!(
            "   │ {:>7} │ {:>9.1} │ {:>9.1} │ {:>12.1} │ {:>10.4} │",
            i + 1,
            seg.start_ms,
            seg.end_ms,
            seg.duration_ms,
            seg.avg_energy
        );
    }

    println!("   └─────────┴───────────┴───────────┴──────────────┴────────────┘");
    println!();

    // =========================================================================
    // Section 6: Summary Metrics
    // =========================================================================
    println!("6. Summary");
    println!("   ─────────────────────────────────────────");

    let total_speech_ms: f64 = segments.iter().map(|s| s.duration_ms).sum();
    let total_ms = AUDIO_DURATION_SECS as f64 * 1000.0;
    let speech_ratio = total_speech_ms / total_ms * 100.0;

    println!("   Segments detected: {}", segments.len());
    println!(
        "   Total speech:      {:.0} ms ({:.1}%)",
        total_speech_ms, speech_ratio
    );
    println!(
        "   Total silence:     {:.0} ms ({:.1}%)",
        total_ms - total_speech_ms,
        100.0 - speech_ratio
    );

    ctx.record_metric("total_frames", n_frames as i64);
    ctx.record_metric("speech_segments", segments.len() as i64);
    ctx.record_float_metric("speech_ratio_pct", speech_ratio);
    println!();

    ctx.report()?;
    println!("\n=== Example Complete ===");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rng() -> rand::rngs::StdRng {
        use rand::SeedableRng;
        rand::rngs::StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_generate_synthetic_audio_length() {
        let mut rng = make_rng();
        let audio = generate_synthetic_audio(&mut rng);
        assert_eq!(audio.len(), TOTAL_SAMPLES);
    }

    #[test]
    fn test_generate_synthetic_audio_amplitude_bounds() {
        let mut rng = make_rng();
        let audio = generate_synthetic_audio(&mut rng);
        for &sample in &audio {
            assert!(
                sample >= -1.0 && sample <= 1.0,
                "Sample out of range: {}",
                sample
            );
        }
    }

    #[test]
    fn test_silence_has_low_energy() {
        let mut rng = make_rng();
        let audio = generate_synthetic_audio(&mut rng);
        // First 2 seconds should be silence — check first second
        let silence_frame = &audio[0..FRAME_SIZE];
        let energy = compute_rms_energy(silence_frame);
        assert!(energy < 0.02, "Silence energy too high: {}", energy);
    }

    #[test]
    fn test_speech_has_high_energy() {
        let mut rng = make_rng();
        let audio = generate_synthetic_audio(&mut rng);
        // At 3 seconds (sample 48000), we should be in speech
        let speech_start = 3 * SAMPLE_RATE;
        let speech_frame = &audio[speech_start..speech_start + FRAME_SIZE];
        let energy = compute_rms_energy(speech_frame);
        assert!(energy > 0.05, "Speech energy too low: {}", energy);
    }

    #[test]
    fn test_rms_energy_empty_frame() {
        assert_eq!(compute_rms_energy(&[]), 0.0);
    }

    #[test]
    fn test_zero_crossing_rate_pure_sine() {
        // A pure sine wave at frequency f in N samples has ~2*f*N/sample_rate crossings
        let frame: Vec<f32> = (0..FRAME_SIZE)
            .map(|i| (2.0 * PI * 200.0 * i as f32 / SAMPLE_RATE as f32).sin())
            .collect();
        let zcr = compute_zero_crossing_rate(&frame);
        // 200 Hz in 30ms => ~12 cycles => ~24 crossings / 479 transitions ~ 0.05
        assert!(
            zcr > 0.01 && zcr < 0.20,
            "ZCR for 200Hz sine unexpected: {}",
            zcr
        );
    }

    #[test]
    fn test_smooth_predictions_removes_glitch() {
        let config = VadConfig::default_16khz();
        // Create a sequence with a single-frame glitch in silence
        let mut frames: Vec<VadFrame> = (0..10)
            .map(|i| VadFrame {
                index: i,
                energy: 0.001,
                zero_crossings: 0.1,
                is_speech: false,
            })
            .collect();
        // Insert a single spurious speech frame
        frames[5].is_speech = true;

        smooth_predictions(&mut frames, config.smooth_window);

        // The glitch should be smoothed away
        assert!(
            !frames[5].is_speech,
            "Single-frame glitch should be smoothed out"
        );
    }

    #[test]
    fn test_detect_segments_basic() {
        let config = VadConfig::default_16khz();
        let frames: Vec<VadFrame> = (0..20)
            .map(|i| VadFrame {
                index: i,
                energy: if (5..15).contains(&i) { 0.2 } else { 0.001 },
                zero_crossings: 0.1,
                is_speech: (5..15).contains(&i),
            })
            .collect();

        let segments = detect_segments(&frames, &config);
        assert_eq!(segments.len(), 1, "Expected exactly one segment");
        assert!(segments[0].duration_ms > 0.0);
        assert!(segments[0].avg_energy > 0.1);
    }

    #[test]
    fn test_render_timeline_format() {
        let frames: Vec<VadFrame> = (0..100)
            .map(|i| VadFrame {
                index: i,
                energy: 0.0,
                zero_crossings: 0.0,
                is_speech: i >= 20 && i < 50,
            })
            .collect();

        let timeline = render_timeline(&frames, 20);
        assert!(timeline.starts_with('['));
        assert!(timeline.ends_with(']'));
        assert_eq!(timeline.len(), 22); // 20 chars + 2 brackets
        assert!(timeline.contains('S'));
        assert!(timeline.contains('.'));
    }

    #[test]
    fn test_end_to_end_vad_detects_two_segments() {
        let mut rng = make_rng();
        let audio = generate_synthetic_audio(&mut rng);
        let config = VadConfig::default_16khz();

        let mut frames = analyze_frames(&audio, &config);
        smooth_predictions(&mut frames, config.smooth_window);
        let segments = detect_segments(&frames, &config);

        // We expect exactly 2 speech segments (2-5s and 6-9s)
        assert_eq!(
            segments.len(),
            2,
            "Expected 2 speech segments, got {}: {:?}",
            segments.len(),
            segments
        );

        // First segment should start near 2000ms
        assert!(
            segments[0].start_ms > 1500.0 && segments[0].start_ms < 2500.0,
            "First segment start_ms={:.0} expected ~2000",
            segments[0].start_ms
        );

        // Second segment should start near 6000ms
        assert!(
            segments[1].start_ms > 5500.0 && segments[1].start_ms < 6500.0,
            "Second segment start_ms={:.0} expected ~6000",
            segments[1].start_ms
        );
    }
}
