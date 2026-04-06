#![allow(unused_imports)]
//! # Recipe: Speaker Diarization
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/whisper-wer-v1.yaml
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

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
