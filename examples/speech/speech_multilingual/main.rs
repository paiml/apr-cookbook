#![allow(unused_imports)]
//! # Recipe: Multilingual Speech Processing
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/whisper-wer-v1.yaml
//! **Category**: Speech Recognition
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## Learning Objective
//! Multi-language speech processing: language identification from acoustic
//! features, confidence scoring, and language-specific transcription routing.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐
//! │ Audio Segment │──▶│ Feature      │──▶│ Language ID           │
//! │ (3s, 16kHz)  │   │ Extraction   │   │ (cosine similarity)   │
//! └──────────────┘   └──────────────┘   └──────────┬────────────┘
//!                                                   │
//!                         ┌─────────────────────────┼──────────┐
//!                         ▼              ▼          ▼          ▼
//!                    ┌─────────┐   ┌──────────┐ ┌────────┐ ┌──────┐
//!                    │ English │   │ Spanish  │ │ 中文   │ │ 日本 │ ...
//!                    │ Engine  │   │ Engine   │ │ Engine │ │ 語   │
//!                    └─────────┘   └──────────┘ └────────┘ └──────┘
//! ```
//!
//! ## Run Command
//! ```bash
//! cargo run --example speech_multilingual
//! ```
//!
//! ## Toyota Way Principles
//! - **Genchi Genbutsu** (Go and see): Measure actual acoustic features, not assumptions
//! - **Jidoka** (Quality built-in): Confidence thresholds gate transcription routing
//! - **Heijunka** (Level production): Uniform processing pipeline for all languages
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

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    println!("=== Multilingual Speech Processing Example ===\n");

    let mut ctx = RecipeContext::new("speech_multilingual")?;
    let profiles = build_language_profiles();
    let sample_rate = 16000_usize;

    // =========================================================================
    // Section 1: Generate Synthetic Audio Segments
    // =========================================================================
    println!("1. Generating Audio Segments");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {} segments x {:.1}s each @ {} Hz\n",
        NUM_SEGMENTS, SEGMENT_DURATION_SECS, sample_rate
    );

    let mut segments: Vec<(Vec<f32>, String)> = Vec::with_capacity(NUM_SEGMENTS);
    for (i, profile) in profiles.iter().enumerate().take(NUM_SEGMENTS) {
        let audio = generate_language_audio(ctx.rng(), profile, SEGMENT_DURATION_SECS, sample_rate);
        println!(
            "   Segment {}: {} ({}) - {} samples",
            i,
            profile.name,
            profile.code,
            audio.len()
        );
        segments.push((audio, profile.code.clone()));
    }
    println!();

    // =========================================================================
    // Section 2: Feature Extraction
    // =========================================================================
    println!("2. Feature Extraction");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>3} {:>12} {:>14} {:>10} {:>10} {:>10}",
        "Seg", "Syll. Rate", "Spec. Tilt", "E_low", "E_mid", "E_high"
    );

    let mut all_features: Vec<AcousticFeatures> = Vec::with_capacity(NUM_SEGMENTS);
    for (i, (audio, _)) in segments.iter().enumerate() {
        let features = extract_features(audio, sample_rate);
        println!(
            "   {:>3} {:>10.2}/s {:>11.2} dB {:>10.3} {:>10.3} {:>10.3}",
            i,
            features.syllable_rate,
            features.spectral_tilt,
            features.energy_low,
            features.energy_mid,
            features.energy_high,
        );
        all_features.push(features);
    }
    println!();

    // =========================================================================
    // Section 3: Language Identification
    // =========================================================================
    println!("3. Language Identification");
    println!("   ─────────────────────────────────────────");

    let mut detection_results: Vec<DetectionResult> = Vec::with_capacity(NUM_SEGMENTS);
    for (i, (features, (_, ground_truth))) in all_features.iter().zip(segments.iter()).enumerate() {
        let scores = identify_language(features, &profiles);
        let top = &scores[0];
        let correct = top.language == *ground_truth;

        println!(
            "   Segment {}: detected={} (conf={:.3}, matched={}/{}) truth={} {}",
            i,
            top.language,
            top.confidence,
            top.features_matched,
            FEATURE_DIM,
            ground_truth,
            if correct { "[OK]" } else { "[MISS]" },
        );

        detection_results.push(DetectionResult {
            segment_id: i,
            detected_language: top.language.clone(),
            confidence: top.confidence,
            ground_truth: ground_truth.clone(),
            correct,
        });
    }
    println!();

    // =========================================================================
    // Section 4: Confidence Scores Table
    // =========================================================================
    println!("4. Confidence Score Matrix");
    println!("   ─────────────────────────────────────────");

    // Header row
    print!("   {:>6}", "Seg");
    for profile in &profiles {
        print!("  {:>6}", profile.code);
    }
    println!();

    for (i, features) in all_features.iter().enumerate() {
        let scores = identify_language(features, &profiles);
        print!("   {:>6}", i);
        for profile in &profiles {
            let score = scores
                .iter()
                .find(|s| s.language == profile.code)
                .map_or(0.0, |s| s.confidence);
            print!("  {:>6.3}", score);
        }
        println!();
    }
    println!();

    // =========================================================================
    // Section 5: Transcription Routing
    // =========================================================================
    println!("5. Transcription Routing");
    println!("   ─────────────────────────────────────────");

    let mut routing_decisions: Vec<RoutingDecision> = Vec::with_capacity(NUM_SEGMENTS);
    for result in &detection_results {
        let decision = route_to_engine(
            result.segment_id,
            &result.detected_language,
            result.confidence,
        );
        println!(
            "   Segment {} -> {} (lang={}, conf={:.3})",
            decision.segment_id, decision.engine, decision.language, decision.confidence
        );
        routing_decisions.push(decision);
    }
    println!();

    // =========================================================================
    // Section 6: Confusion Matrix
    // =========================================================================
    println!("6. Confusion Matrix");
    println!("   ─────────────────────────────────────────");

    let labels: Vec<String> = profiles.iter().map(|p| p.code.clone()).collect();
    let matrix = build_confusion_matrix(&detection_results, &labels);

    // Print header
    print!("   {:>8}", "Truth\\Pred");
    for label in &labels {
        print!("  {:>4}", label);
    }
    println!();

    // Print rows
    for (i, label) in labels.iter().enumerate() {
        print!("   {:>8}", label);
        for &count in &matrix[i] {
            print!("  {:>4}", count);
        }
        println!();
    }

    let total_correct = detection_results.iter().filter(|r| r.correct).count();
    let accuracy = total_correct as f64 / detection_results.len().max(1) as f64;
    println!(
        "\n   Overall accuracy: {}/{} ({:.1}%)",
        total_correct,
        detection_results.len(),
        accuracy * 100.0
    );
    println!();

    // Record metrics
    ctx.record_float_metric("accuracy", accuracy);
    ctx.record_metric("segments_processed", detection_results.len() as i64);
    ctx.record_metric("languages", NUM_LANGUAGES as i64);

    println!("=== Example Complete ===");
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_profiles_count() {
        let profiles = build_language_profiles();
        assert_eq!(profiles.len(), NUM_LANGUAGES);
    }

    #[test]
    fn test_language_profiles_unique_codes() {
        let profiles = build_language_profiles();
        let mut codes: Vec<&str> = profiles.iter().map(|p| p.code.as_str()).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), NUM_LANGUAGES, "language codes must be unique");
    }

    #[test]
    fn test_generate_audio_length() {
        let mut ctx = RecipeContext::new("test_audio_len").expect("context");
        let profile = &build_language_profiles()[0];
        let audio = generate_language_audio(ctx.rng(), profile, 3.0, 16000);
        assert_eq!(audio.len(), 48000);
    }

    #[test]
    fn test_generate_audio_amplitude_range() {
        let mut ctx = RecipeContext::new("test_audio_amp").expect("context");
        let profile = &build_language_profiles()[0];
        let audio = generate_language_audio(ctx.rng(), profile, 1.0, 16000);
        for &sample in &audio {
            assert!(
                (-1.0..=1.0).contains(&sample),
                "sample out of range: {}",
                sample
            );
        }
    }

    #[test]
    fn test_extract_features_valid_ranges() {
        let mut ctx = RecipeContext::new("test_feat_ranges").expect("context");
        let profile = &build_language_profiles()[0];
        let audio = generate_language_audio(ctx.rng(), profile, 3.0, 16000);
        let features = extract_features(&audio, 16000);

        assert!(
            features.syllable_rate > 0.0,
            "syllable rate must be positive"
        );
        assert!(
            features.spectral_tilt >= -10.0 && features.spectral_tilt <= 0.0,
            "spectral tilt out of range: {}",
            features.spectral_tilt
        );
        let energy_sum = features.energy_low + features.energy_mid + features.energy_high;
        assert!(
            (energy_sum - 1.0).abs() < 0.01,
            "energy ratios must sum to ~1.0, got {}",
            energy_sum
        );
    }

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let v = [0.5, 0.3, 0.6, 0.2, 0.1];
        let sim = cosine_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "identical vectors should have similarity ~1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = [1.0, 0.0, 0.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-5,
            "orthogonal vectors should have similarity ~0.0, got {}",
            sim
        );
    }

    #[test]
    fn test_identify_language_returns_all_profiles() {
        let profiles = build_language_profiles();
        let features = AcousticFeatures {
            syllable_rate: 4.3,
            spectral_tilt: -6.0,
            energy_low: 0.5,
            energy_mid: 0.3,
            energy_high: 0.2,
        };
        let scores = identify_language(&features, &profiles);
        assert_eq!(scores.len(), NUM_LANGUAGES);
    }

    #[test]
    fn test_identify_language_english_profile() {
        let profiles = build_language_profiles();
        // Features matching English profile closely
        let en_profile = &profiles[0];
        let prof_vec = profile_to_vec(en_profile);
        let features = AcousticFeatures {
            syllable_rate: prof_vec[0] * 10.0,
            spectral_tilt: prof_vec[1] * 10.0 - 10.0,
            energy_low: prof_vec[2],
            energy_mid: prof_vec[3],
            energy_high: prof_vec[4],
        };
        let scores = identify_language(&features, &profiles);
        assert_eq!(
            scores[0].language, "en",
            "English features should identify as English, got {}",
            scores[0].language
        );
    }

    #[test]
    fn test_route_to_engine_known_languages() {
        let cases = [
            ("en", "whisper-en-v3"),
            ("es", "whisper-es-v2"),
            ("zh", "whisper-zh-tonal-v2"),
            ("ja", "whisper-ja-mora-v2"),
            ("de", "whisper-de-v2"),
        ];
        for (lang, expected_engine) in &cases {
            let decision = route_to_engine(0, lang, 0.95);
            assert_eq!(
                decision.engine, *expected_engine,
                "lang {} should route to {}",
                lang, expected_engine
            );
        }
    }

    #[test]
    fn test_confusion_matrix_dimensions() {
        let labels = vec!["en".to_string(), "es".to_string(), "zh".to_string()];
        let results = vec![
            DetectionResult {
                segment_id: 0,
                detected_language: "en".to_string(),
                confidence: 0.9,
                ground_truth: "en".to_string(),
                correct: true,
            },
            DetectionResult {
                segment_id: 1,
                detected_language: "es".to_string(),
                confidence: 0.8,
                ground_truth: "zh".to_string(),
                correct: false,
            },
        ];
        let matrix = build_confusion_matrix(&results, &labels);
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[0].len(), 3);
        // en->en should be 1
        assert_eq!(matrix[0][0], 1);
        // zh->es should be 1
        assert_eq!(matrix[2][1], 1);
    }
}
