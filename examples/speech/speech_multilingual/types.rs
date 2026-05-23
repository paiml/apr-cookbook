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

// ============================================================================
// Constants
// ============================================================================

/// Number of language profiles
pub const NUM_LANGUAGES: usize = 5;

/// Audio segment duration in seconds
pub const SEGMENT_DURATION_SECS: f32 = 3.0;

/// Number of test segments to generate
pub const NUM_SEGMENTS: usize = 5;

/// Feature vector dimension (syllable_rate, spectral_tilt, energy_low, energy_mid, energy_high)
pub const FEATURE_DIM: usize = 5;

// ============================================================================
// Data Structures
// ============================================================================

// Acoustic profile for a language.
//
// Characterizes a language by its prosodic and spectral properties,
/// used as a reference template for language identification.
#[derive(Debug, Clone)]
pub struct LanguageProfile {
    // Human-readable language name
    pub name: String,
    // ISO 639-1 language code
    pub code: String,
    // Average syllable rate (syllables per second)
    pub syllable_rate: f32,
    // Spectral tilt in dB/octave (negative = falling spectrum)
    pub spectral_tilt: f32,
    // Whether the language is tonal (e.g., Mandarin)
    pub tonal: bool,
}

/// Score for a language candidate during identification.
#[derive(Debug, Clone)]
pub struct LanguageScore {
    // Language code that was scored
    pub language: String,
    // Confidence score in [0, 1]
    pub confidence: f32,
    // Number of features that closely matched the profile
    pub features_matched: usize,
}

/// Result of language detection for a single audio segment.
#[derive(Debug, Clone)]
pub struct DetectionResult {
    // Segment index (0-based)
    pub segment_id: usize,
    // Detected language code
    pub detected_language: String,
    // Confidence of the detection
    pub confidence: f32,
    // Actual language (ground truth)
    pub ground_truth: String,
    // Whether the detection was correct
    pub correct: bool,
}

/// Extracted acoustic features from an audio segment.
#[derive(Debug, Clone)]
pub struct AcousticFeatures {
    // Estimated syllable rate (syllables/second)
    pub syllable_rate: f32,
    // Estimated spectral tilt (dB/octave)
    pub spectral_tilt: f32,
    // Low-frequency energy ratio (0-500 Hz band)
    pub energy_low: f32,
    // Mid-frequency energy ratio (500-2000 Hz band)
    pub energy_mid: f32,
    // High-frequency energy ratio (2000+ Hz band)
    pub energy_high: f32,
}

/// Routing decision for a detected language.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    // Segment index
    pub segment_id: usize,
    // Target transcription engine name
    pub engine: String,
    // Language code being routed
    pub language: String,
    // Confidence that triggered the routing
    pub confidence: f32,
}

// ============================================================================
// Language Profile Definitions
// ============================================================================

// Build the set of reference language profiles.
//
// Each profile captures characteristic acoustic features:
// - Syllable rate: temporal rhythm (syllables/second)
// - Spectral tilt: energy distribution across frequencies (dB/octave)
/// - Tonal: whether pitch contour carries lexical meaning
pub fn build_language_profiles() -> Vec<LanguageProfile> {
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

// Generate a synthetic audio segment matching a language profile.
//
// Produces samples with statistical properties (syllable rate, spectral
// characteristics) consistent with the target language profile, plus
/// controlled Gaussian noise for realism.
pub fn generate_language_audio(
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

// Extract acoustic features from an audio segment.
//
// Computes five features:
// 1. Syllable rate estimate via energy envelope peak counting
// 2. Spectral tilt from low/high frequency energy ratio
// 3. Low-frequency energy ratio
// 4. Mid-frequency energy ratio
/// 5. High-frequency energy ratio
pub fn extract_features(audio: &[f32], sample_rate: usize) -> AcousticFeatures {
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
pub fn features_to_vec(features: &AcousticFeatures) -> [f32; FEATURE_DIM] {
    [
        features.syllable_rate / 10.0,          // Normalize syllable rate
        (features.spectral_tilt + 10.0) / 10.0, // Normalize tilt to [0, 1]
        features.energy_low,
        features.energy_mid,
        features.energy_high,
    ]
}

/// Convert a language profile to a normalized feature vector.
pub fn profile_to_vec(profile: &LanguageProfile) -> [f32; FEATURE_DIM] {
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
pub fn cosine_similarity(a: &[f32; FEATURE_DIM], b: &[f32; FEATURE_DIM]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    let denom = mag_a * mag_b;
    if denom < 1e-10 {
        return 0.0;
    }
    (dot / denom).clamp(0.0, 1.0)
}

// Compute weighted Euclidean distance between two feature vectors.
//
// Syllable rate and spectral tilt are weighted more heavily as they
/// are the most discriminative features for language identification.
pub fn weighted_distance(a: &[f32; FEATURE_DIM], b: &[f32; FEATURE_DIM]) -> f32 {
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
pub fn count_matched_features(
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

// Identify the language of an audio segment by scoring against all profiles.
//
// Uses weighted Euclidean distance converted to a confidence score via
/// softmax-like normalization. Returns scores sorted by confidence descending.
pub fn identify_language(
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

// Route a detected language to its transcription engine.
//
// Each language maps to a specialized engine optimized for that language's
/// phonetics, vocabulary, and acoustic model.
pub fn route_to_engine(segment_id: usize, language: &str, confidence: f32) -> RoutingDecision {
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

// Build and display a confusion matrix from detection results.
//
/// Rows = ground truth, Columns = predicted.
pub fn build_confusion_matrix(results: &[DetectionResult], labels: &[String]) -> Vec<Vec<u32>> {
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
