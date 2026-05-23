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
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub const NUM_FEATURES: usize = 8;
pub const NUM_CLASSES: usize = 3;

/// A single data sample
#[derive(Clone, Debug)]
pub struct Sample {
    pub features: Vec<f32>,
    pub label: usize,
}

/// Statistics for a feature column
#[derive(Clone, Debug)]
pub struct FeatureStats {
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
}

/// Normalization strategy
#[derive(Clone, Copy)]
pub enum NormStrategy {
    MinMax,
    ZScore,
    RobustScale,
}

impl NormStrategy {
    pub fn name(self) -> &'static str {
        match self {
            NormStrategy::MinMax => "MinMax [0,1]",
            NormStrategy::ZScore => "Z-Score",
            NormStrategy::RobustScale => "Robust Scale",
        }
    }
}

/// Compute per-feature statistics
pub fn compute_stats(data: &[Sample]) -> Vec<FeatureStats> {
    let n = data.len() as f32;
    (0..NUM_FEATURES)
        .map(|f| {
            let values: Vec<f32> = data.iter().map(|s| s.features[f]).collect();
            let mean = values.iter().sum::<f32>() / n;
            let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
            let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            FeatureStats {
                mean,
                std_dev: var.sqrt(),
                min,
                max,
            }
        })
        .collect()
}

/// Normalize data using a given strategy
pub fn normalize(data: &[Sample], stats: &[FeatureStats], strategy: NormStrategy) -> Vec<Sample> {
    data.iter()
        .map(|sample| {
            let features = sample
                .features
                .iter()
                .enumerate()
                .map(|(f, &val)| {
                    let s = &stats[f];
                    match strategy {
                        NormStrategy::MinMax => {
                            let range = (s.max - s.min).max(1e-8);
                            (val - s.min) / range
                        }
                        NormStrategy::ZScore => (val - s.mean) / s.std_dev.max(1e-8),
                        NormStrategy::RobustScale => {
                            // Use mean as center, std as scale (simplified robust)
                            (val - s.mean) / (s.std_dev * 1.35).max(1e-8)
                        }
                    }
                })
                .collect();
            Sample {
                features,
                label: sample.label,
            }
        })
        .collect()
}

/// Train/test split with stratification
pub fn stratified_split(data: &[Sample], test_ratio: f32, seed: u64) -> (Vec<Sample>, Vec<Sample>) {
    let mut by_class: Vec<Vec<&Sample>> = vec![Vec::new(); NUM_CLASSES];
    for sample in data {
        by_class[sample.label].push(sample);
    }

    let mut train = Vec::new();
    let mut test = Vec::new();

    for (class_idx, class_samples) in by_class.iter().enumerate() {
        // Deterministic shuffle using hash
        let mut indices: Vec<usize> = (0..class_samples.len()).collect();
        indices.sort_by_key(|&i| {
            let mut h = DefaultHasher::new();
            (seed, class_idx, i).hash(&mut h);
            h.finish()
        });

        let split_point = (class_samples.len() as f32 * (1.0 - test_ratio)) as usize;
        for (pos, &idx) in indices.iter().enumerate() {
            if pos < split_point {
                train.push(class_samples[idx].clone());
            } else {
                test.push(class_samples[idx].clone());
            }
        }
    }

    (train, test)
}

/// Data augmentation: add noise to features
pub fn augment_with_noise(
    data: &[Sample],
    noise_level: f32,
    copies: usize,
    seed: u64,
) -> Vec<Sample> {
    let mut augmented = data.to_vec();
    for copy in 0..copies {
        for (i, sample) in data.iter().enumerate() {
            let features = sample
                .features
                .iter()
                .enumerate()
                .map(|(f, &val)| {
                    let mut h = DefaultHasher::new();
                    (seed, "noise", copy, i, f).hash(&mut h);
                    let noise = (h.finish() as f32 / u64::MAX as f32 - 0.5) * 2.0 * noise_level;
                    val + noise
                })
                .collect();
            augmented.push(Sample {
                features,
                label: sample.label,
            });
        }
    }
    augmented
}

/// Validate data for common issues
pub fn validate_data(data: &[Sample]) -> Vec<String> {
    let mut issues = Vec::new();

    if data.is_empty() {
        issues.push("Dataset is empty".to_string());
        return issues;
    }

    // Check for NaN/Inf
    let nan_count = data
        .iter()
        .flat_map(|s| &s.features)
        .filter(|v| !v.is_finite())
        .count();
    if nan_count > 0 {
        issues.push(format!("{nan_count} NaN/Inf values found"));
    }

    // Check class balance
    let mut class_counts = [0usize; NUM_CLASSES];
    for sample in data {
        if sample.label < NUM_CLASSES {
            class_counts[sample.label] += 1;
        } else {
            issues.push(format!("Invalid label: {}", sample.label));
        }
    }

    let max_count = class_counts.iter().max().copied().unwrap_or(0);
    let min_count = class_counts.iter().min().copied().unwrap_or(0);
    if max_count > min_count * 3 {
        issues.push(format!(
            "Class imbalance: min={min_count}, max={max_count} (ratio {:.1}x)",
            max_count as f64 / min_count.max(1) as f64
        ));
    }

    // Check feature dimensions
    let expected_dim = data[0].features.len();
    let bad_dims = data
        .iter()
        .filter(|s| s.features.len() != expected_dim)
        .count();
    if bad_dims > 0 {
        issues.push(format!(
            "{bad_dims} samples with inconsistent feature dimensions"
        ));
    }

    issues
}

/// Generate synthetic dataset
pub fn generate_dataset(n_per_class: usize, seed: u64) -> Vec<Sample> {
    let mut data = Vec::with_capacity(n_per_class * NUM_CLASSES);
    for class in 0..NUM_CLASSES {
        for i in 0..n_per_class {
            let features: Vec<f32> = (0..NUM_FEATURES)
                .map(|f| {
                    let mut h = DefaultHasher::new();
                    (seed, class, i, f).hash(&mut h);
                    let base = h.finish() as f32 / u64::MAX as f32;
                    // Class-conditional distribution
                    base * 10.0 + class as f32 * 5.0 + f as f32 * 0.5
                })
                .collect();
            data.push(Sample {
                features,
                label: class,
            });
        }
    }
    data
}
