#![allow(unused_imports)]
//! Voice Activity Detection (VAD) Example
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/whisper-wer-v1.yaml
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
use rand::Rng;
use std::f32::consts::PI;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
