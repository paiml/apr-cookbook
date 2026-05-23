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
use std::f64::consts::PI;

/// Audio sample rate in Hz.
pub const SAMPLE_RATE: usize = 16_000;

/// Frame size in samples (30ms at 16kHz).
pub const FRAME_SIZE: usize = 480;

/// Total audio duration in seconds.
pub const TOTAL_DURATION_SECS: usize = 12;

/// Number of speakers to detect.
pub const NUM_SPEAKERS: usize = 3;

/// Number of k-means iterations.
pub const KMEANS_ITERATIONS: usize = 10;

// ============================================================================
// Data Structures
// ============================================================================

/// A 3-dimensional speaker embedding extracted from one audio frame.
#[derive(Debug, Clone, Copy)]
pub struct SpeakerEmbedding {
    // Features: [fundamental_frequency, spectral_centroid, energy].
    pub features: [f64; 3],
}

/// A contiguous segment where one speaker is talking.
#[derive(Debug, Clone)]
pub struct DiarizationTurn {
    // Cluster-assigned speaker identifier (0-indexed).
    pub speaker_id: usize,
    // Start time in milliseconds.
    pub start_ms: u64,
    // End time in milliseconds.
    pub end_ms: u64,
    // Duration in milliseconds.
    pub duration_ms: u64,
}

/// State for the k-means clustering algorithm.
#[derive(Debug, Clone)]
pub struct KMeansState {
    // Current centroid positions (one per cluster).
    pub centroids: Vec<[f64; 3]>,
    // Cluster assignment for each data point.
    pub assignments: Vec<usize>,
}

/// Speaker segment definition for audio generation.
#[derive(Debug, Clone, Copy)]
pub struct SpeakerSegment {
    // Fundamental frequency in Hz.
    pub frequency_hz: f64,
    // Start time in seconds.
    pub start_sec: f64,
    // End time in seconds.
    pub end_sec: f64,
    // Label for display.
    pub label: char,
}

// ============================================================================
// Audio Generation
// ============================================================================

// Generate a synthetic 3-speaker conversation.
//
// Layout:
// - Speaker A (0-3s): 200 Hz fundamental
// - Speaker B (3-6s): 350 Hz fundamental
// - Speaker A (6-8s): 200 Hz (same as before)
// - Speaker C (8-11s): 275 Hz fundamental
/// - Silence  (11-12s)
pub fn generate_conversation(ctx: &mut RecipeContext) -> Vec<f64> {
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

// Estimate fundamental frequency via autocorrelation peak detection.
//
// Searches for the first significant autocorrelation peak (above 40%
// of the zero-lag energy) in the plausible speech range (80-500 Hz).
/// Using the first peak avoids octave errors from sub-harmonic lags.
pub fn estimate_fundamental_frequency(frame: &[f64]) -> f64 {
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

// Compute a simplified spectral centroid for a frame.
//
// Uses energy-weighted frequency bin average from a basic DFT magnitude
/// estimate at a handful of probe frequencies.
pub fn compute_spectral_centroid(frame: &[f64]) -> f64 {
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
pub fn compute_energy(frame: &[f64]) -> f64 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = frame.iter().map(|s| s * s).sum();
    (sum_sq / frame.len() as f64).sqrt()
}

/// Normalize a 3-dim feature vector to unit length.
pub fn normalize_embedding(features: [f64; 3]) -> [f64; 3] {
    let norm =
        (features[0] * features[0] + features[1] * features[1] + features[2] * features[2]).sqrt();

    if norm < 1e-12 {
        return [0.0; 3];
    }

    [features[0] / norm, features[1] / norm, features[2] / norm]
}

// Extract per-frame speaker embeddings from audio.
//
// Each frame of `FRAME_SIZE` samples yields a 3-dim embedding:
/// `[fundamental_frequency, spectral_centroid, energy]`, normalized.
pub fn extract_embeddings(audio: &[f64]) -> Vec<SpeakerEmbedding> {
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
pub fn distance_sq(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
}

// Initialize centroids from the first frame of each known segment.
//
/// Segment boundaries (in frames): A starts at 0, B at 100, C at ~267.
pub fn initialize_centroids(embeddings: &[SpeakerEmbedding]) -> Vec<[f64; 3]> {
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

// Run k-means clustering on speaker embeddings.
//
/// Returns the final `KMeansState` after the specified number of iterations.
pub fn run_kmeans(embeddings: &[SpeakerEmbedding], k: usize, iterations: usize) -> KMeansState {
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
pub fn merge_turns(assignments: &[usize]) -> Vec<DiarizationTurn> {
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
pub fn speaker_label(id: usize) -> char {
    (b'A' + id as u8) as char
}

/// Print the speaker timeline as a visual bar.
pub fn print_timeline(assignments: &[usize], total_frames: usize) {
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
pub fn print_turn_table(turns: &[DiarizationTurn]) {
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
