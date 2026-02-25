//! # Recipe: Multilingual Speech Processing
//!
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

use apr_cookbook::prelude::*;
use rand::Rng;

// ============================================================================
// Constants
// ============================================================================

/// Number of language profiles
const NUM_LANGUAGES: usize = 5;

/// Audio segment duration in seconds
const SEGMENT_DURATION_SECS: f32 = 3.0;

/// Number of test segments to generate
const NUM_SEGMENTS: usize = 5;

/// Feature vector dimension (syllable_rate, spectral_tilt, energy_low, energy_mid, energy_high)
const FEATURE_DIM: usize = 5;

// ============================================================================
// Data Structures
// ============================================================================

/// Acoustic profile for a language.
///
/// Characterizes a language by its prosodic and spectral properties,
/// used as a reference template for language identification.
#[derive(Debug, Clone)]
struct LanguageProfile {
    /// Human-readable language name
    name: String,
    /// ISO 639-1 language code
    code: String,
    /// Average syllable rate (syllables per second)
    syllable_rate: f32,
    /// Spectral tilt in dB/octave (negative = falling spectrum)
    spectral_tilt: f32,
    /// Whether the language is tonal (e.g., Mandarin)
    tonal: bool,
}

/// Score for a language candidate during identification.
#[derive(Debug, Clone)]
struct LanguageScore {
    /// Language code that was scored
    language: String,
    /// Confidence score in [0, 1]
    confidence: f32,
    /// Number of features that closely matched the profile
    features_matched: usize,
}

/// Result of language detection for a single audio segment.
#[derive(Debug, Clone)]
struct DetectionResult {
    /// Segment index (0-based)
    segment_id: usize,
    /// Detected language code
    detected_language: String,
    /// Confidence of the detection
    confidence: f32,
    /// Actual language (ground truth)
    ground_truth: String,
    /// Whether the detection was correct
    correct: bool,
}

/// Extracted acoustic features from an audio segment.
#[derive(Debug, Clone)]
struct AcousticFeatures {
    /// Estimated syllable rate (syllables/second)
    syllable_rate: f32,
    /// Estimated spectral tilt (dB/octave)
    spectral_tilt: f32,
    /// Low-frequency energy ratio (0-500 Hz band)
    energy_low: f32,
    /// Mid-frequency energy ratio (500-2000 Hz band)
    energy_mid: f32,
    /// High-frequency energy ratio (2000+ Hz band)
    energy_high: f32,
}

/// Routing decision for a detected language.
#[derive(Debug, Clone)]
struct RoutingDecision {
    /// Segment index
    segment_id: usize,
    /// Target transcription engine name
    engine: String,
    /// Language code being routed
    language: String,
    /// Confidence that triggered the routing
    confidence: f32,
}

// ============================================================================
// Language Profile Definitions
// ============================================================================

/// Build the set of reference language profiles.
///
/// Each profile captures characteristic acoustic features:
/// - Syllable rate: temporal rhythm (syllables/second)
/// - Spectral tilt: energy distribution across frequencies (dB/octave)
/// - Tonal: whether pitch contour carries lexical meaning
fn build_language_profiles() -> Vec<LanguageProfile> {
    vec![
        LanguageProfile {
            name: "English".to_string(),
            code: "en".to_string(),
            syllable_rate: 4.3,
            spectral_tilt: -6.0,
            tonal: false,
        },
        LanguageProfile {
            name: "Spanish".to_string(),
            code: "es".to_string(),
            syllable_rate: 7.8,
            spectral_tilt: -4.0,
            tonal: false,
        },
        LanguageProfile {
            name: "Mandarin".to_string(),
            code: "zh".to_string(),
            syllable_rate: 5.2,
            spectral_tilt: -5.0,
            tonal: true,
        },
        LanguageProfile {
            name: "Japanese".to_string(),
            code: "ja".to_string(),
            syllable_rate: 7.0,
            spectral_tilt: -3.0,
            tonal: false,
        },
        LanguageProfile {
            name: "German".to_string(),
            code: "de".to_string(),
            syllable_rate: 5.0,
            spectral_tilt: -7.0,
            tonal: false,
        },
    ]
}

// ============================================================================
// Audio Generation
// ============================================================================

/// Generate a synthetic audio segment matching a language profile.
///
/// Produces samples with statistical properties (syllable rate, spectral
/// characteristics) consistent with the target language profile, plus
/// controlled Gaussian noise for realism.
fn generate_language_audio(
    rng: &mut impl Rng,
    profile: &LanguageProfile,
    duration_secs: f32,
    sample_rate: usize,
) -> Vec<f32> {
    let n_samples = (duration_secs * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(n_samples);

    // Derive energy distribution from spectral tilt
    // More negative tilt = more low-frequency energy
    let tilt_norm = (profile.spectral_tilt + 7.0) / 4.0; // Normalize to ~[0, 1]
    let low_weight = 0.6 - 0.2 * tilt_norm;
    let mid_weight = 0.3 + 0.1 * tilt_norm;
    let high_weight = 0.1 + 0.1 * tilt_norm;

    // Syllable modulation frequency derived from syllable rate
    let syllable_mod_hz = profile.syllable_rate;

    for i in 0..n_samples {
        let t = i as f32 / sample_rate as f32;

        // Syllable envelope (amplitude modulation at syllable rate)
        let envelope = 0.5 + 0.5 * (2.0 * std::f32::consts::PI * syllable_mod_hz * t).sin();

        // Low-frequency component (~200 Hz fundamental)
        let low = low_weight * (2.0 * std::f32::consts::PI * 200.0 * t).sin();

        // Mid-frequency component (~1000 Hz)
        let mid = mid_weight * (2.0 * std::f32::consts::PI * 1000.0 * t).sin();

        // High-frequency component (~3000 Hz)
        let high = high_weight * (2.0 * std::f32::consts::PI * 3000.0 * t).sin();

        // Tonal languages get pitch variation at ~5 Hz (F0 modulation)
        let tonal_mod = if profile.tonal {
            0.15 * (2.0 * std::f32::consts::PI * 5.0 * t).sin()
        } else {
            0.0
        };

        // Combine with envelope and add noise
        let noise: f32 = rng.gen_range(-0.05..0.05);
        let sample = envelope * (low + mid + high + tonal_mod) + noise;
        samples.push(sample.clamp(-1.0, 1.0));
    }

    samples
}

// ============================================================================
// Feature Extraction
// ============================================================================

/// Extract acoustic features from an audio segment.
///
/// Computes five features:
/// 1. Syllable rate estimate via energy envelope peak counting
/// 2. Spectral tilt from low/high frequency energy ratio
/// 3. Low-frequency energy ratio
/// 4. Mid-frequency energy ratio
/// 5. High-frequency energy ratio
fn extract_features(audio: &[f32], sample_rate: usize) -> AcousticFeatures {
    let duration = audio.len() as f32 / sample_rate as f32;

    // Estimate syllable rate from energy envelope zero-crossings
    let frame_size = sample_rate / 100; // 10ms frames
    let energies: Vec<f32> = audio
        .chunks(frame_size)
        .map(|frame| frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32)
        .collect();

    let mean_energy = energies.iter().sum::<f32>() / energies.len().max(1) as f32;
    let mut peaks = 0u32;
    let mut prev_above = false;
    for &e in &energies {
        let above = e > mean_energy;
        if above && !prev_above {
            peaks += 1;
        }
        prev_above = above;
    }
    let syllable_rate = peaks as f32 / duration;

    // Spectral energy analysis via simple band filtering
    // Low band: samples that change slowly (moving average proxy)
    // High band: samples that change rapidly (difference proxy)
    let mut low_energy = 0.0_f32;
    let mut mid_energy = 0.0_f32;
    let mut high_energy = 0.0_f32;

    for i in 2..audio.len() {
        let slow = (audio[i] + audio[i - 1] + audio[i - 2]) / 3.0;
        low_energy += slow * slow;

        let mid = audio[i] - audio[i - 1];
        mid_energy += mid * mid;

        let fast = audio[i] - 2.0 * audio[i - 1] + audio[i - 2];
        high_energy += fast * fast;
    }

    let total = (low_energy + mid_energy + high_energy).max(1e-10);
    let low_ratio = low_energy / total;
    let mid_ratio = mid_energy / total;
    let high_ratio = high_energy / total;

    // Spectral tilt from low/high ratio (log scale)
    let spectral_tilt = if high_ratio > 1e-10 {
        -3.0 * (low_ratio / high_ratio).ln()
    } else {
        -7.0
    };

    AcousticFeatures {
        syllable_rate,
        spectral_tilt: spectral_tilt.clamp(-10.0, 0.0),
        energy_low: low_ratio,
        energy_mid: mid_ratio,
        energy_high: high_ratio,
    }
}

/// Convert acoustic features to a normalized feature vector.
fn features_to_vec(features: &AcousticFeatures) -> [f32; FEATURE_DIM] {
    [
        features.syllable_rate / 10.0,          // Normalize syllable rate
        (features.spectral_tilt + 10.0) / 10.0, // Normalize tilt to [0, 1]
        features.energy_low,
        features.energy_mid,
        features.energy_high,
    ]
}

/// Convert a language profile to a normalized feature vector.
fn profile_to_vec(profile: &LanguageProfile) -> [f32; FEATURE_DIM] {
    // Derive expected energy distribution from spectral tilt
    let tilt_norm = (profile.spectral_tilt + 7.0) / 4.0;
    let low_weight = 0.6 - 0.2 * tilt_norm;
    let mid_weight = 0.3 + 0.1 * tilt_norm;
    let high_weight = 0.1 + 0.1 * tilt_norm;
    let total = low_weight + mid_weight + high_weight;

    [
        profile.syllable_rate / 10.0,
        (profile.spectral_tilt + 10.0) / 10.0,
        low_weight / total,
        mid_weight / total,
        high_weight / total,
    ]
}

// ============================================================================
// Language Identification
// ============================================================================

/// Compute cosine similarity between two feature vectors.
#[cfg(test)]
fn cosine_similarity(a: &[f32; FEATURE_DIM], b: &[f32; FEATURE_DIM]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = mag_a * mag_b;
    if denom < 1e-10 {
        return 0.0;
    }
    (dot / denom).clamp(0.0, 1.0)
}

/// Compute weighted Euclidean distance between two feature vectors.
///
/// Syllable rate and spectral tilt are weighted more heavily as they
/// are the most discriminative features for language identification.
fn weighted_distance(a: &[f32; FEATURE_DIM], b: &[f32; FEATURE_DIM]) -> f32 {
    // Weights: syllable_rate=4.0, spectral_tilt=3.0, energy bands=1.0
    let weights = [4.0_f32, 3.0, 1.0, 1.0, 1.0];
    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .zip(weights.iter())
        .map(|((x, y), w)| w * (x - y) * (x - y))
        .sum();
    sum.sqrt()
}

/// Count how many individual features are within threshold of the profile.
fn count_matched_features(
    features: &[f32; FEATURE_DIM],
    profile: &[f32; FEATURE_DIM],
    threshold: f32,
) -> usize {
    features
        .iter()
        .zip(profile.iter())
        .filter(|(a, b)| (*a - *b).abs() < threshold)
        .count()
}

/// Identify the language of an audio segment by scoring against all profiles.
///
/// Uses weighted Euclidean distance converted to a confidence score via
/// softmax-like normalization. Returns scores sorted by confidence descending.
fn identify_language(
    features: &AcousticFeatures,
    profiles: &[LanguageProfile],
) -> Vec<LanguageScore> {
    let feat_vec = features_to_vec(features);

    // Compute distances to each profile
    let distances: Vec<f32> = profiles
        .iter()
        .map(|profile| {
            let prof_vec = profile_to_vec(profile);
            weighted_distance(&feat_vec, &prof_vec)
        })
        .collect();

    // Convert distances to confidence via softmax over negative distances
    // (shorter distance = higher confidence)
    let neg_dists: Vec<f32> = distances.iter().map(|d| (-8.0 * d).exp()).collect();
    let total: f32 = neg_dists.iter().sum();

    let mut scores: Vec<LanguageScore> = profiles
        .iter()
        .zip(neg_dists.iter())
        .map(|(profile, &nd)| {
            let prof_vec = profile_to_vec(profile);
            let matched = count_matched_features(&feat_vec, &prof_vec, 0.10);
            let confidence = if total > 1e-10 { nd / total } else { 0.0 };

            LanguageScore {
                language: profile.code.clone(),
                confidence,
                features_matched: matched,
            }
        })
        .collect();

    // Sort by confidence descending
    scores.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scores
}

// ============================================================================
// Transcription Routing
// ============================================================================

/// Route a detected language to its transcription engine.
///
/// Each language maps to a specialized engine optimized for that language's
/// phonetics, vocabulary, and acoustic model.
fn route_to_engine(segment_id: usize, language: &str, confidence: f32) -> RoutingDecision {
    let engine = match language {
        "en" => "whisper-en-v3",
        "es" => "whisper-es-v2",
        "zh" => "whisper-zh-tonal-v2",
        "ja" => "whisper-ja-mora-v2",
        "de" => "whisper-de-v2",
        _ => "whisper-multilingual-v1",
    };

    RoutingDecision {
        segment_id,
        engine: engine.to_string(),
        language: language.to_string(),
        confidence,
    }
}

// ============================================================================
// Confusion Matrix
// ============================================================================

/// Build and display a confusion matrix from detection results.
///
/// Rows = ground truth, Columns = predicted.
fn build_confusion_matrix(results: &[DetectionResult], labels: &[String]) -> Vec<Vec<u32>> {
    let n = labels.len();
    let mut matrix = vec![vec![0u32; n]; n];

    for result in results {
        let true_idx = labels.iter().position(|l| *l == result.ground_truth);
        let pred_idx = labels.iter().position(|l| *l == result.detected_language);
        if let (Some(ti), Some(pi)) = (true_idx, pred_idx) {
            matrix[ti][pi] += 1;
        }
    }

    matrix
}

// ============================================================================
// Main Entry Point
// ============================================================================

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
