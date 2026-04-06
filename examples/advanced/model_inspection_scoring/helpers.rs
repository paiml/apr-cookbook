#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports, clippy::wildcard_imports)]
use super::types::*;
use apr_cookbook::prelude::*;
use std::collections::HashMap;
use std::f32;

// ============================================================================
// Core Implementation
// ============================================================================

/// Demonstrate model inspection on a generated test model
pub fn demonstrate_inspection(ctx: &RecipeContext) -> Result<InspectionResult> {
    println!("📊 Creating test model for inspection...");

    // Generate test weights with known properties
    let weights = generate_test_weights(ctx, 1000, false, false)?;
    let mut model = SimpleModel::new(100, 10); // 100x10 = 1000 params
    model.weights = weights.clone();

    println!("🔍 Inspecting model...");

    // Inspect the model
    let result = inspect_model(&model, &weights, "test_model.apr")?;

    Ok(result)
}

/// Demonstrate model diff between two versions
pub fn demonstrate_model_diff(ctx: &RecipeContext) -> Result<ModelDiff> {
    println!("📊 Creating two model versions for diff...");

    // Version A: Original weights
    let weights_a = generate_test_weights(ctx, 500, false, false)?;

    // Version B: Slightly modified weights (simulating training update)
    let mut weights_b = weights_a.clone();
    for w in weights_b.iter_mut().take(100) {
        *w += 0.01; // Small perturbation
    }

    println!("🔍 Computing model diff...");

    let diff = compute_model_diff(&weights_a, &weights_b, "model_v1.apr", "model_v2.apr")?;

    Ok(diff)
}

/// Inspect a model and return comprehensive results
pub fn inspect_model(
    _model: &SimpleModel,
    weights: &[f32],
    model_path: &str,
) -> Result<InspectionResult> {
    // Parse header (simulated for demo)
    let header = parse_header(weights)?;

    // Extract metadata
    let metadata = extract_metadata(weights)?;

    // Compute weight statistics
    let weight_stats = compute_weight_stats(weights)?;

    // Calculate quality score
    let quality_score = calculate_quality_score(&header, &weight_stats)?;

    // Determine health status
    let health_status = determine_health_status(quality_score.total);

    Ok(InspectionResult {
        header,
        metadata: MetadataInfo {
            model_type: metadata.model_type,
            model_name: model_path.to_string(),
            framework: metadata.framework,
            created_at: metadata.created_at,
            parameters: weights.len(),
            hyperparameters: HashMap::new(),
        },
        weight_stats: vec![weight_stats],
        quality_score,
        health_status,
    })
}

/// Parse APR header (simulated)
pub fn parse_header(weights: &[f32]) -> Result<HeaderInfo> {
    // Compute CRC32 checksum
    let checksum = compute_crc32(weights);

    Ok(HeaderInfo {
        magic: "APRN".to_string(),
        version: (1, 0),
        flags: FeatureFlags {
            compressed: false,
            signed: false,
            encrypted: false,
            streaming: false,
            licensed: false,
            quantized: false,
        },
        compression_ratio: 1.0,
        checksum,
    })
}

/// Extract model metadata
pub fn extract_metadata(_weights: &[f32]) -> Result<MetadataInfo> {
    Ok(MetadataInfo {
        model_type: "LinearRegression".to_string(),
        model_name: "test_model".to_string(),
        framework: "aprender".to_string(),
        created_at: "2025-12-08T00:00:00Z".to_string(),
        parameters: 0,
        hyperparameters: HashMap::new(),
    })
}

/// Accumulator for weight statistics computation
#[derive(Default)]
pub struct WeightAccumulator {
    pub nan_count: usize,
    pub inf_count: usize,
    pub zero_count: usize,
    pub sum: f64,
    pub sum_sq: f64,
    pub min: f32,
    pub max: f32,
}

impl WeightAccumulator {
    pub fn new() -> Self {
        Self {
            min: f32::MAX,
            max: f32::MIN,
            ..Default::default()
        }
    }

    pub fn process(&mut self, w: f32) {
        if w.is_nan() {
            self.nan_count += 1;
        } else if w.is_infinite() {
            self.inf_count += 1;
        } else {
            self.process_valid(w);
        }
    }

    pub fn process_valid(&mut self, w: f32) {
        if w == 0.0 {
            self.zero_count += 1;
        }
        self.sum += f64::from(w);
        self.sum_sq += f64::from(w) * f64::from(w);
        self.min = self.min.min(w);
        self.max = self.max.max(w);
    }

    pub fn finalize_range(&self) -> (f32, f32) {
        let min = if (self.min - f32::MAX).abs() < f32::EPSILON {
            0.0
        } else {
            self.min
        };
        let max = if (self.max - f32::MIN).abs() < f32::EPSILON {
            0.0
        } else {
            self.max
        };
        (min, max)
    }
}

/// Compute comprehensive weight statistics
pub fn compute_weight_stats(weights: &[f32]) -> Result<LayerStats> {
    if weights.is_empty() {
        return Err(CookbookError::invalid_format("Empty weights"));
    }

    let mut acc = WeightAccumulator::new();
    for &w in weights {
        acc.process(w);
    }

    let valid_count = weights.len() - acc.nan_count - acc.inf_count;
    let mean = if valid_count > 0 {
        (acc.sum / valid_count as f64) as f32
    } else {
        0.0
    };

    let variance = if valid_count > 1 {
        let mean_sq = (acc.sum_sq / valid_count as f64) as f32;
        (mean_sq - mean * mean).max(0.0)
    } else {
        0.0
    };

    let std = variance.sqrt();
    let sparsity = acc.zero_count as f32 / weights.len() as f32;
    let (final_min, final_max) = acc.finalize_range();

    Ok(LayerStats {
        name: "weights".to_string(),
        shape: vec![weights.len()],
        dtype: "f32".to_string(),
        min: final_min,
        max: final_max,
        mean,
        std,
        nan_count: acc.nan_count,
        inf_count: acc.inf_count,
        zero_count: acc.zero_count,
        sparsity,
    })
}

/// Score structural integrity (max 25 pts)
pub fn score_structural(header: &HeaderInfo) -> u8 {
    let mut score: u8 = 25;
    if header.magic != "APRN" {
        score = score.saturating_sub(25);
    }
    if header.checksum == 0 {
        score = score.saturating_sub(5);
    }
    score
}

/// Score numerical stability (max 25 pts)
pub fn score_numerical(stats: &LayerStats) -> u8 {
    let mut score: u8 = 25;
    if stats.nan_count > 0 {
        score = score.saturating_sub(15);
    }
    if stats.inf_count > 0 {
        score = score.saturating_sub(10);
    }
    if stats.max > 1e6 || stats.min < -1e6 {
        score = score.saturating_sub(5);
    }
    score
}

/// Score compression efficiency (max 25 pts)
pub fn score_compression(header: &HeaderInfo) -> u8 {
    if header.compression_ratio >= 2.0 {
        25
    } else if header.compression_ratio >= 1.5 {
        20
    } else if header.flags.compressed {
        15
    } else {
        10
    }
}

/// Score security compliance (max 25 pts)
pub fn score_security(header: &HeaderInfo) -> u8 {
    let mut score: u8 = 10;
    if header.flags.signed {
        score += 10;
    }
    if header.flags.encrypted {
        score += 5;
    }
    score
}

/// Map total score to letter grade
pub fn score_to_grade(total: u8) -> &'static str {
    match total {
        95..=100 => "A+",
        90..=94 => "A",
        80..=89 => "B",
        70..=79 => "C",
        60..=69 => "D",
        _ => "F",
    }
}

/// Calculate quality score (100-point scale)
pub fn calculate_quality_score(header: &HeaderInfo, stats: &LayerStats) -> Result<QualityScore> {
    let structural = score_structural(header);
    let numerical = score_numerical(stats);
    let compression = score_compression(header);
    let security = score_security(header);

    let total = structural + numerical + compression + security;
    let grade = score_to_grade(total).to_string();

    Ok(QualityScore {
        structural,
        numerical,
        compression,
        security,
        total,
        grade,
    })
}

/// Determine health status from score
pub fn determine_health_status(score: u8) -> HealthStatus {
    match score {
        85..=100 => HealthStatus::Healthy,
        60..=84 => HealthStatus::Warning,
        _ => HealthStatus::Critical,
    }
}

/// Compute model diff between two weight vectors
pub fn compute_model_diff(
    weights_a: &[f32],
    weights_b: &[f32],
    name_a: &str,
    name_b: &str,
) -> Result<ModelDiff> {
    if weights_a.len() != weights_b.len() {
        return Err(CookbookError::invalid_format(
            "Weight vectors must have same length for diff",
        ));
    }

    let mut sum_sq_diff = 0.0_f64;
    let mut dot_product = 0.0_f64;
    let mut norm_a = 0.0_f64;
    let mut norm_b = 0.0_f64;
    let mut max_abs_diff: f32 = 0.0;

    for (&a, &b) in weights_a.iter().zip(weights_b.iter()) {
        let diff = f64::from(a) - f64::from(b);
        sum_sq_diff += diff * diff;
        dot_product += f64::from(a) * f64::from(b);
        norm_a += f64::from(a) * f64::from(a);
        norm_b += f64::from(b) * f64::from(b);
        let abs_diff = (a - b).abs();
        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
        }
    }

    let l2_distance = sum_sq_diff.sqrt();
    let cosine_similarity = if norm_a > 0.0 && norm_b > 0.0 {
        dot_product / (norm_a.sqrt() * norm_b.sqrt())
    } else {
        0.0
    };

    // Drift threshold: L2 > 1.0 or cosine < 0.99
    let drift_detected = l2_distance > 1.0 || cosine_similarity < 0.99;

    let layer_diff = LayerDiff {
        name: "weights".to_string(),
        l2_distance,
        cosine_similarity,
        max_abs_diff,
        changed: max_abs_diff > 1e-6,
    };

    Ok(ModelDiff {
        model_a: name_a.to_string(),
        model_b: name_b.to_string(),
        layer_diffs: vec![layer_diff],
        total_l2_distance: l2_distance,
        cosine_similarity,
        drift_detected,
    })
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Generate test weights with optional NaN/Inf injection
pub fn generate_test_weights(
    ctx: &RecipeContext,
    count: usize,
    inject_nan: bool,
    inject_inf: bool,
) -> Result<Vec<f32>> {
    let seed = hash_name_to_seed(ctx.name());
    let mut weights = generate_test_data(seed, count);

    if inject_nan && !weights.is_empty() {
        weights[0] = f32::NAN;
    }
    if inject_inf && weights.len() > 1 {
        weights[1] = f32::INFINITY;
    }

    Ok(weights)
}

/// Compute CRC32 checksum of weights
pub fn compute_crc32(weights: &[f32]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &w in weights {
        let bytes = w.to_le_bytes();
        for &byte in &bytes {
            crc ^= u32::from(byte);
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB8_8320;
                } else {
                    crc >>= 1;
                }
            }
        }
    }
    !crc
}
