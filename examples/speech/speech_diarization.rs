//! # Recipe: Speaker Diarization
//!
//! **Category**: Speech Recognition
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## Learning Objective
//! Identify "who spoke when" in a multi-speaker audio stream using simplified
//! speaker embeddings and k-means clustering. Demonstrates the full pipeline:
//! audio generation, per-frame feature extraction, clustering, and turn merging.
//!
//! ## Run Command
//! ```bash
//! cargo run --example speech_diarization
//! ```
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────┐   ┌────────────┐   ┌──────────┐   ┌─────────┐
//! │  Audio   │──>│  Feature   │──>│ K-Means  │──>│  Merge  │
//! │  Stream  │   │ Extraction │   │ Cluster  │   │  Turns  │
//! └──────────┘   └────────────┘   └──────────┘   └─────────┘
//!    16kHz          f0, centroid,    k=3            Consecutive
//!    3 speakers     energy           10 iter        same-speaker
//! ```
//!
//! ## Toyota Way Principles
//! - **Genchi Genbutsu** (Go and see): Analyze actual audio features, not assumptions
//! - **Jidoka** (Quality built-in): Validated embeddings and deterministic clustering
//! - **Muda** (Waste elimination): Minimal 3-dim embeddings, no heavy ML stack
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
use std::f64::consts::PI;

/// Audio sample rate in Hz.
const SAMPLE_RATE: usize = 16_000;

/// Frame size in samples (30ms at 16kHz).
const FRAME_SIZE: usize = 480;

/// Total audio duration in seconds.
const TOTAL_DURATION_SECS: usize = 12;

/// Number of speakers to detect.
const NUM_SPEAKERS: usize = 3;

/// Number of k-means iterations.
const KMEANS_ITERATIONS: usize = 10;

// ============================================================================
// Data Structures
// ============================================================================

/// A 3-dimensional speaker embedding extracted from one audio frame.
#[derive(Debug, Clone, Copy)]
struct SpeakerEmbedding {
    /// Features: [fundamental_frequency, spectral_centroid, energy].
    features: [f64; 3],
}

/// A contiguous segment where one speaker is talking.
#[derive(Debug, Clone)]
struct DiarizationTurn {
    /// Cluster-assigned speaker identifier (0-indexed).
    speaker_id: usize,
    /// Start time in milliseconds.
    start_ms: u64,
    /// End time in milliseconds.
    end_ms: u64,
    /// Duration in milliseconds.
    duration_ms: u64,
}

/// State for the k-means clustering algorithm.
#[derive(Debug, Clone)]
struct KMeansState {
    /// Current centroid positions (one per cluster).
    centroids: Vec<[f64; 3]>,
    /// Cluster assignment for each data point.
    assignments: Vec<usize>,
}

/// Speaker segment definition for audio generation.
#[derive(Debug, Clone, Copy)]
struct SpeakerSegment {
    /// Fundamental frequency in Hz.
    frequency_hz: f64,
    /// Start time in seconds.
    start_sec: f64,
    /// End time in seconds.
    end_sec: f64,
    /// Label for display.
    label: char,
}

// ============================================================================
// Audio Generation
// ============================================================================

/// Generate a synthetic 3-speaker conversation.
///
/// Layout:
/// - Speaker A (0-3s): 200 Hz fundamental
/// - Speaker B (3-6s): 350 Hz fundamental
/// - Speaker A (6-8s): 200 Hz (same as before)
/// - Speaker C (8-11s): 275 Hz fundamental
/// - Silence  (11-12s)
fn generate_conversation(ctx: &mut RecipeContext) -> Vec<f64> {
    let n_samples = TOTAL_DURATION_SECS * SAMPLE_RATE;
    let mut audio = vec![0.0f64; n_samples];

    let segments = [
        SpeakerSegment {
            frequency_hz: 200.0,
            start_sec: 0.0,
            end_sec: 3.0,
            label: 'A',
        },
        SpeakerSegment {
            frequency_hz: 350.0,
            start_sec: 3.0,
            end_sec: 6.0,
            label: 'B',
        },
        SpeakerSegment {
            frequency_hz: 200.0,
            start_sec: 6.0,
            end_sec: 8.0,
            label: 'A',
        },
        SpeakerSegment {
            frequency_hz: 275.0,
            start_sec: 8.0,
            end_sec: 11.0,
            label: 'C',
        },
    ];

    for seg in &segments {
        let start_sample = (seg.start_sec * SAMPLE_RATE as f64) as usize;
        let end_sample = (seg.end_sec * SAMPLE_RATE as f64) as usize;

        for (i, sample) in audio
            .iter_mut()
            .enumerate()
            .take(end_sample.min(n_samples))
            .skip(start_sample)
        {
            let t = i as f64 / SAMPLE_RATE as f64;
            let fundamental = (2.0 * PI * seg.frequency_hz * t).sin();
            let harmonic2 = 0.5 * (2.0 * PI * seg.frequency_hz * 2.0 * t).sin();
            let harmonic3 = 0.25 * (2.0 * PI * seg.frequency_hz * 3.0 * t).sin();
            let noise: f64 = ctx.rng().gen_range(-0.05..0.05);
            *sample = 0.6 * (fundamental + harmonic2 + harmonic3) + noise;
        }
    }

    println!("   Generated {} speaker segments:", segments.len());
    for seg in &segments {
        println!(
            "     Speaker {}: {:.0}-{:.0}s at {} Hz",
            seg.label, seg.start_sec, seg.end_sec, seg.frequency_hz
        );
    }
    println!("     Silence: 11-12s");

    audio
}

// ============================================================================
// Feature Extraction
// ============================================================================

/// Estimate fundamental frequency via autocorrelation peak detection.
///
/// Searches for the first significant autocorrelation peak (above 40%
/// of the zero-lag energy) in the plausible speech range (80-500 Hz).
/// Using the first peak avoids octave errors from sub-harmonic lags.
fn estimate_fundamental_frequency(frame: &[f64]) -> f64 {
    let min_lag = SAMPLE_RATE / 500; // 500 Hz upper bound
    let max_lag = SAMPLE_RATE / 80; // 80 Hz lower bound
    let max_lag = max_lag.min(frame.len() - 1);

    // Compute zero-lag autocorrelation (frame energy) for normalization
    let energy: f64 = frame.iter().map(|s| s * s).sum();
    if energy < 1e-10 {
        return 0.0;
    }

    // Find the first peak above threshold (shortest lag = highest frequency)
    let threshold = 0.4 * energy;
    let mut prev_corr = 0.0f64;
    let mut rising = false;

    for lag in min_lag..=max_lag {
        let mut corr = 0.0f64;
        let limit = frame.len() - lag;
        for i in 0..limit {
            corr += frame[i] * frame[i + lag];
        }

        // Detect first peak: was rising, now falling, and above threshold
        if corr < prev_corr && rising && prev_corr > threshold {
            return SAMPLE_RATE as f64 / (lag - 1) as f64;
        }

        rising = corr > prev_corr;
        prev_corr = corr;
    }

    0.0
}

/// Compute a simplified spectral centroid for a frame.
///
/// Uses energy-weighted frequency bin average from a basic DFT magnitude
/// estimate at a handful of probe frequencies.
fn compute_spectral_centroid(frame: &[f64]) -> f64 {
    let probe_frequencies: Vec<f64> = (1..=20).map(|i| f64::from(i) * 50.0).collect();
    let mut weighted_sum = 0.0f64;
    let mut total_energy = 0.0f64;

    for &freq in &probe_frequencies {
        let mut real = 0.0f64;
        let mut imag = 0.0f64;
        for (i, &sample) in frame.iter().enumerate() {
            let angle = 2.0 * PI * freq * i as f64 / SAMPLE_RATE as f64;
            real += sample * angle.cos();
            imag += sample * angle.sin();
        }
        let magnitude = (real * real + imag * imag).sqrt();
        weighted_sum += freq * magnitude;
        total_energy += magnitude;
    }

    if total_energy < 1e-12 {
        return 0.0;
    }

    weighted_sum / total_energy
}

/// Compute frame energy (root mean square).
fn compute_energy(frame: &[f64]) -> f64 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = frame.iter().map(|s| s * s).sum();
    (sum_sq / frame.len() as f64).sqrt()
}

/// Normalize a 3-dim feature vector to unit length.
fn normalize_embedding(features: [f64; 3]) -> [f64; 3] {
    let norm =
        (features[0] * features[0] + features[1] * features[1] + features[2] * features[2]).sqrt();

    if norm < 1e-12 {
        return [0.0; 3];
    }

    [features[0] / norm, features[1] / norm, features[2] / norm]
}

/// Extract per-frame speaker embeddings from audio.
///
/// Each frame of `FRAME_SIZE` samples yields a 3-dim embedding:
/// `[fundamental_frequency, spectral_centroid, energy]`, normalized.
fn extract_embeddings(audio: &[f64]) -> Vec<SpeakerEmbedding> {
    let n_frames = audio.len() / FRAME_SIZE;
    let mut embeddings = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * FRAME_SIZE;
        let end = start + FRAME_SIZE;
        let frame = &audio[start..end];

        let f0 = estimate_fundamental_frequency(frame);
        let centroid = compute_spectral_centroid(frame);
        let energy = compute_energy(frame);

        let features = normalize_embedding([f0, centroid, energy]);
        embeddings.push(SpeakerEmbedding { features });
    }

    embeddings
}

// ============================================================================
// K-Means Clustering
// ============================================================================

/// Compute squared Euclidean distance between two 3-dim points.
fn distance_sq(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

/// Initialize centroids from the first frame of each known segment.
///
/// Segment boundaries (in frames): A starts at 0, B at 100, C at ~267.
fn initialize_centroids(embeddings: &[SpeakerEmbedding]) -> Vec<[f64; 3]> {
    let frames_per_sec = SAMPLE_RATE / FRAME_SIZE;
    let centroid_indices = [
        0,                  // Speaker A at 0s
        3 * frames_per_sec, // Speaker B at 3s
        8 * frames_per_sec, // Speaker C at 8s
    ];

    centroid_indices
        .iter()
        .map(|&idx| {
            if idx < embeddings.len() {
                embeddings[idx].features
            } else {
                [0.0; 3]
            }
        })
        .collect()
}

/// Run k-means clustering on speaker embeddings.
///
/// Returns the final `KMeansState` after the specified number of iterations.
fn run_kmeans(embeddings: &[SpeakerEmbedding], k: usize, iterations: usize) -> KMeansState {
    let mut centroids = initialize_centroids(embeddings);
    centroids.truncate(k);

    let mut assignments = vec![0usize; embeddings.len()];

    for _ in 0..iterations {
        // Assignment step: assign each point to nearest centroid
        for (i, emb) in embeddings.iter().enumerate() {
            let mut best_cluster = 0;
            let mut best_dist = f64::MAX;
            for (c, centroid) in centroids.iter().enumerate() {
                let dist = distance_sq(&emb.features, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = c;
                }
            }
            assignments[i] = best_cluster;
        }

        // Update step: recompute centroids
        let mut sums = vec![[0.0f64; 3]; k];
        let mut counts = vec![0usize; k];

        for (i, emb) in embeddings.iter().enumerate() {
            let c = assignments[i];
            sums[c][0] += emb.features[0];
            sums[c][1] += emb.features[1];
            sums[c][2] += emb.features[2];
            counts[c] += 1;
        }

        for (c, centroid) in centroids.iter_mut().enumerate() {
            if counts[c] > 0 {
                let n = counts[c] as f64;
                centroid[0] = sums[c][0] / n;
                centroid[1] = sums[c][1] / n;
                centroid[2] = sums[c][2] / n;
            }
        }
    }

    KMeansState {
        centroids,
        assignments,
    }
}

// ============================================================================
// Turn Merging
// ============================================================================

/// Merge consecutive same-speaker frame assignments into diarization turns.
fn merge_turns(assignments: &[usize]) -> Vec<DiarizationTurn> {
    if assignments.is_empty() {
        return Vec::new();
    }

    let ms_per_frame = (FRAME_SIZE as u64 * 1000) / SAMPLE_RATE as u64;
    let mut turns = Vec::new();
    let mut current_speaker = assignments[0];
    let mut segment_start: usize = 0;

    for (i, &speaker) in assignments.iter().enumerate().skip(1) {
        if speaker != current_speaker {
            let start_ms = segment_start as u64 * ms_per_frame;
            let end_ms = i as u64 * ms_per_frame;
            turns.push(DiarizationTurn {
                speaker_id: current_speaker,
                start_ms,
                end_ms,
                duration_ms: end_ms - start_ms,
            });
            current_speaker = speaker;
            segment_start = i;
        }
    }

    // Final segment
    let start_ms = segment_start as u64 * ms_per_frame;
    let end_ms = assignments.len() as u64 * ms_per_frame;
    turns.push(DiarizationTurn {
        speaker_id: current_speaker,
        start_ms,
        end_ms,
        duration_ms: end_ms - start_ms,
    });

    turns
}

// ============================================================================
// Display
// ============================================================================

/// Map cluster ID to a speaker label for display.
fn speaker_label(id: usize) -> char {
    (b'A' + id as u8) as char
}

/// Print the speaker timeline as a visual bar.
fn print_timeline(assignments: &[usize], total_frames: usize) {
    let bar_width: usize = 60;
    let mut bar = String::with_capacity(bar_width);

    for col in 0..bar_width {
        let frame_idx = col * total_frames / bar_width;
        if frame_idx < assignments.len() {
            bar.push(speaker_label(assignments[frame_idx]));
        } else {
            bar.push('.');
        }
    }

    println!(
        "   Timeline (each char = {:.0}ms):",
        (total_frames as f64 / bar_width as f64)
            * (FRAME_SIZE as f64 / SAMPLE_RATE as f64)
            * 1000.0
    );
    println!("   |{}|", bar);
    println!("   0s                    6s                    12s");
}

/// Print the turn table.
fn print_turn_table(turns: &[DiarizationTurn]) {
    println!("   ┌───────┬─────────┬──────────┬──────────┬─────────────┐");
    println!("   │ Turn  │ Speaker │ Start_ms │  End_ms  │ Duration_ms │");
    println!("   ├───────┼─────────┼──────────┼──────────┼─────────────┤");
    for (i, turn) in turns.iter().enumerate() {
        println!(
            "   │ {:>5} │    {}    │ {:>8} │ {:>8} │ {:>11} │",
            i + 1,
            speaker_label(turn.speaker_id),
            turn.start_ms,
            turn.end_ms,
            turn.duration_ms,
        );
    }
    println!("   └───────┴─────────┴──────────┴──────────┴─────────────┘");
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("=== Speaker Diarization Example ===\n");

    let mut ctx =
        RecipeContext::new("speech_diarization").expect("Failed to create recipe context");

    // =========================================================================
    // Section 1: Generate synthetic conversation
    // =========================================================================
    println!("1. Generating Synthetic Conversation");
    println!("   ─────────────────────────────────────────");
    println!("   Sample rate: {} Hz", SAMPLE_RATE);
    println!("   Duration: {} seconds", TOTAL_DURATION_SECS);
    println!("   Speakers: {}", NUM_SPEAKERS);

    let audio = generate_conversation(&mut ctx);
    println!("   Total samples: {}", audio.len());
    println!();

    // =========================================================================
    // Section 2: Extract speaker embeddings
    // =========================================================================
    println!("2. Extracting Speaker Embeddings");
    println!("   ─────────────────────────────────────────");
    println!(
        "   Frame size: {} samples ({:.1} ms)",
        FRAME_SIZE,
        FRAME_SIZE as f64 / SAMPLE_RATE as f64 * 1000.0
    );

    let embeddings = extract_embeddings(&audio);
    println!("   Total frames: {}", embeddings.len());
    println!("   Embedding dim: 3 (f0, centroid, energy)");

    // Show a few sample embeddings
    let sample_indices = [
        0,
        embeddings.len() / 4,
        embeddings.len() / 2,
        3 * embeddings.len() / 4,
    ];
    for &idx in &sample_indices {
        if idx < embeddings.len() {
            let e = &embeddings[idx];
            let time_ms = idx as u64 * (FRAME_SIZE as u64 * 1000 / SAMPLE_RATE as u64);
            println!(
                "   Frame {:>4} ({:>5}ms): [{:.3}, {:.3}, {:.3}]",
                idx, time_ms, e.features[0], e.features[1], e.features[2]
            );
        }
    }
    println!();

    // =========================================================================
    // Section 3: Cluster with k-means
    // =========================================================================
    println!(
        "3. K-Means Clustering (k={}, {} iterations)",
        NUM_SPEAKERS, KMEANS_ITERATIONS
    );
    println!("   ─────────────────────────────────────────");

    let state = run_kmeans(&embeddings, NUM_SPEAKERS, KMEANS_ITERATIONS);

    println!("   Final centroids:");
    for (i, c) in state.centroids.iter().enumerate() {
        println!(
            "     Cluster {} (Speaker {}): [{:.4}, {:.4}, {:.4}]",
            i,
            speaker_label(i),
            c[0],
            c[1],
            c[2]
        );
    }

    // Count frames per cluster
    let mut cluster_counts = [0usize; NUM_SPEAKERS];
    for &a in &state.assignments {
        if a < NUM_SPEAKERS {
            cluster_counts[a] += 1;
        }
    }
    println!("   Frames per cluster:");
    for (i, &count) in cluster_counts.iter().enumerate() {
        println!(
            "     Speaker {}: {} frames ({:.1}%)",
            speaker_label(i),
            count,
            100.0 * count as f64 / embeddings.len() as f64
        );
    }
    println!();

    // =========================================================================
    // Section 4: Merge turns and display results
    // =========================================================================
    println!("4. Diarization Results");
    println!("   ─────────────────────────────────────────");

    let turns = merge_turns(&state.assignments);
    println!("   Detected {} speaker turns", turns.len());
    println!();

    print_timeline(&state.assignments, embeddings.len());
    println!();

    print_turn_table(&turns);
    println!();

    // =========================================================================
    // Section 5: Summary metrics
    // =========================================================================
    println!("5. Summary");
    println!("   ─────────────────────────────────────────");

    let total_speech_ms: u64 = turns
        .iter()
        .filter(|t| {
            // Check if this turn has significant energy (not silence)
            let frame_idx = (t.start_ms as usize * SAMPLE_RATE) / (1000 * FRAME_SIZE);
            frame_idx < embeddings.len() && embeddings[frame_idx].features[2] > 0.01
        })
        .map(|t| t.duration_ms)
        .sum();

    ctx.record_metric("total_frames", embeddings.len() as i64);
    ctx.record_metric("num_turns", turns.len() as i64);
    ctx.record_metric("total_speech_ms", total_speech_ms as i64);

    println!("   Total frames analyzed: {}", embeddings.len());
    println!("   Speaker turns detected: {}", turns.len());
    println!("   Total speech duration: {} ms", total_speech_ms);
    println!("   Clustering iterations: {}", KMEANS_ITERATIONS);
    println!();

    println!("=== Example Complete ===");
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_conversation_length() {
        let mut ctx = RecipeContext::new("test_gen_conv").expect("ctx");
        let audio = generate_conversation(&mut ctx);
        assert_eq!(audio.len(), TOTAL_DURATION_SECS * SAMPLE_RATE);
    }

    #[test]
    fn test_generate_conversation_has_speech_and_silence() {
        let mut ctx = RecipeContext::new("test_speech_silence").expect("ctx");
        let audio = generate_conversation(&mut ctx);

        // Speaker A region (0-3s) should have significant energy
        let speech_energy = compute_energy(&audio[0..SAMPLE_RATE]);
        assert!(
            speech_energy > 0.01,
            "Speech region should have energy > 0.01, got {speech_energy}"
        );

        // Silence region (11-12s) should have near-zero energy
        let silence_start = 11 * SAMPLE_RATE;
        let silence_energy = compute_energy(&audio[silence_start..silence_start + SAMPLE_RATE]);
        assert!(
            silence_energy < 0.1,
            "Silence region should have low energy, got {silence_energy}"
        );
    }

    #[test]
    fn test_fundamental_frequency_estimation() {
        // Generate a pure 200 Hz tone
        let n = FRAME_SIZE;
        let tone: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 200.0 * i as f64 / SAMPLE_RATE as f64).sin())
            .collect();

        let f0 = estimate_fundamental_frequency(&tone);
        // Should be close to 200 Hz (within 20% tolerance due to discrete lag)
        assert!((f0 - 200.0).abs() < 40.0, "Expected ~200 Hz, got {f0} Hz");
    }

    #[test]
    fn test_silence_fundamental_frequency() {
        let silent = vec![0.0f64; FRAME_SIZE];
        let f0 = estimate_fundamental_frequency(&silent);
        assert!(f0.abs() < 1e-6, "Silent frame should yield ~0 Hz f0");
    }

    #[test]
    fn test_spectral_centroid_higher_for_higher_pitch() {
        let n = FRAME_SIZE;
        let low: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 100.0 * i as f64 / SAMPLE_RATE as f64).sin())
            .collect();
        let high: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 400.0 * i as f64 / SAMPLE_RATE as f64).sin())
            .collect();

        let centroid_low = compute_spectral_centroid(&low);
        let centroid_high = compute_spectral_centroid(&high);
        assert!(
            centroid_high > centroid_low,
            "Higher pitch should have higher centroid: low={centroid_low}, high={centroid_high}"
        );
    }

    #[test]
    fn test_normalize_embedding() {
        let features = [3.0, 4.0, 0.0];
        let normed = normalize_embedding(features);
        let len = (normed[0] * normed[0] + normed[1] * normed[1] + normed[2] * normed[2]).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-9,
            "Normalized vector should have unit length, got {len}"
        );
    }

    #[test]
    fn test_normalize_zero_vector() {
        let zero = [0.0, 0.0, 0.0];
        let normed = normalize_embedding(zero);
        assert_eq!(normed, [0.0; 3], "Zero vector should remain zero");
    }

    #[test]
    fn test_kmeans_assigns_all_points() {
        let mut ctx = RecipeContext::new("test_kmeans_assign").expect("ctx");
        let audio = generate_conversation(&mut ctx);
        let embeddings = extract_embeddings(&audio);
        let state = run_kmeans(&embeddings, NUM_SPEAKERS, KMEANS_ITERATIONS);

        assert_eq!(state.assignments.len(), embeddings.len());
        for &a in &state.assignments {
            assert!(a < NUM_SPEAKERS, "Assignment {a} out of range");
        }
    }

    #[test]
    fn test_merge_turns_single_speaker() {
        let assignments = vec![0, 0, 0, 0, 0];
        let turns = merge_turns(&assignments);
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].speaker_id, 0);
    }

    #[test]
    fn test_merge_turns_alternating() {
        let assignments = vec![0, 0, 1, 1, 0, 0];
        let turns = merge_turns(&assignments);
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0].speaker_id, 0);
        assert_eq!(turns[1].speaker_id, 1);
        assert_eq!(turns[2].speaker_id, 0);

        // Verify timing continuity
        for i in 1..turns.len() {
            assert_eq!(
                turns[i].start_ms,
                turns[i - 1].end_ms,
                "Turns should be contiguous"
            );
        }
    }

    #[test]
    fn test_merge_turns_empty() {
        let turns = merge_turns(&[]);
        assert!(turns.is_empty());
    }
}
