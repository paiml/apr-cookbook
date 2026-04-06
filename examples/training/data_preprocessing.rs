//! Data Preprocessing Pipeline Example
//! **CLI Equivalent**: `apr data`
//!
//! Demonstrates common data preprocessing steps for ML training:
//! normalization, standardization, train/test splitting, data augmentation,
//! and feature engineering.
//!
//! # Pipeline
//!
//! ```text
//! Raw Data → [Validation] → [Split] → [Normalize] → [Augment] → Ready
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example data_preprocessing
//! ```

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const NUM_FEATURES: usize = 8;
const NUM_CLASSES: usize = 3;

/// A single data sample
#[derive(Clone, Debug)]
struct Sample {
    features: Vec<f32>,
    label: usize,
}

/// Statistics for a feature column
#[derive(Clone, Debug)]
struct FeatureStats {
    mean: f32,
    std_dev: f32,
    min: f32,
    max: f32,
}

/// Normalization strategy
#[derive(Clone, Copy)]
enum NormStrategy {
    MinMax,
    ZScore,
    RobustScale,
}

impl NormStrategy {
    fn name(self) -> &'static str {
        match self {
            NormStrategy::MinMax => "MinMax [0,1]",
            NormStrategy::ZScore => "Z-Score",
            NormStrategy::RobustScale => "Robust Scale",
        }
    }
}

/// Compute per-feature statistics
fn compute_stats(data: &[Sample]) -> Vec<FeatureStats> {
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
fn normalize(data: &[Sample], stats: &[FeatureStats], strategy: NormStrategy) -> Vec<Sample> {
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
fn stratified_split(data: &[Sample], test_ratio: f32, seed: u64) -> (Vec<Sample>, Vec<Sample>) {
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
fn augment_with_noise(data: &[Sample], noise_level: f32, copies: usize, seed: u64) -> Vec<Sample> {
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
fn validate_data(data: &[Sample]) -> Vec<String> {
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
fn generate_dataset(n_per_class: usize, seed: u64) -> Vec<Sample> {
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

fn main() {
    println!("=== Data Preprocessing Pipeline ===\n");

    let seed = 42;

    // =========================================================================
    // Section 1: Data Generation & Validation
    // =========================================================================
    println!("1. Data Generation & Validation");
    println!("   ─────────────────────────────────────────");

    let raw_data = generate_dataset(100, seed);
    println!("   Samples: {}", raw_data.len());
    println!("   Features: {NUM_FEATURES}");
    println!("   Classes: {NUM_CLASSES}");

    let issues = validate_data(&raw_data);
    if issues.is_empty() {
        println!("   Validation: PASSED (no issues)");
    } else {
        for issue in &issues {
            println!("   Issue: {issue}");
        }
    }
    println!();

    // =========================================================================
    // Section 2: Feature Statistics
    // =========================================================================
    println!("2. Feature Statistics (raw)");
    println!("   ─────────────────────────────────────────");

    let stats = compute_stats(&raw_data);
    println!(
        "   {:>4} {:>10} {:>10} {:>10} {:>10}",
        "Feat", "Mean", "StdDev", "Min", "Max"
    );
    println!("   {}", "─".repeat(46));

    for (i, s) in stats.iter().enumerate() {
        println!(
            "   {:>4} {:>10.3} {:>10.3} {:>10.3} {:>10.3}",
            i, s.mean, s.std_dev, s.min, s.max
        );
    }
    println!();

    // =========================================================================
    // Section 3: Train/Test Split
    // =========================================================================
    println!("3. Stratified Train/Test Split");
    println!("   ─────────────────────────────────────────");

    let (train, test) = stratified_split(&raw_data, 0.2, seed);

    println!("   Train: {} samples", train.len());
    println!("   Test:  {} samples", test.len());

    // Verify stratification
    let mut train_counts = [0usize; NUM_CLASSES];
    let mut test_counts = [0usize; NUM_CLASSES];
    for s in &train {
        train_counts[s.label] += 1;
    }
    for s in &test {
        test_counts[s.label] += 1;
    }

    println!("   Class distribution:");
    for c in 0..NUM_CLASSES {
        println!(
            "     Class {}: train={}, test={}",
            c, train_counts[c], test_counts[c]
        );
    }
    println!();

    // =========================================================================
    // Section 4: Normalization Comparison
    // =========================================================================
    println!("4. Normalization Comparison");
    println!("   ─────────────────────────────────────────");

    let train_stats = compute_stats(&train);
    let strategies = [
        NormStrategy::MinMax,
        NormStrategy::ZScore,
        NormStrategy::RobustScale,
    ];

    for strategy in strategies {
        let normalized = normalize(&train, &train_stats, strategy);
        let norm_stats = compute_stats(&normalized);

        let avg_mean: f32 = norm_stats.iter().map(|s| s.mean).sum::<f32>() / NUM_FEATURES as f32;
        let avg_std: f32 = norm_stats.iter().map(|s| s.std_dev).sum::<f32>() / NUM_FEATURES as f32;
        let avg_min: f32 = norm_stats.iter().map(|s| s.min).sum::<f32>() / NUM_FEATURES as f32;
        let avg_max: f32 = norm_stats.iter().map(|s| s.max).sum::<f32>() / NUM_FEATURES as f32;

        println!(
            "   {:<14} mean={:>7.3}, std={:>7.3}, range=[{:>6.3}, {:>6.3}]",
            strategy.name(),
            avg_mean,
            avg_std,
            avg_min,
            avg_max
        );
    }
    println!();

    // =========================================================================
    // Section 5: Data Augmentation
    // =========================================================================
    println!("5. Data Augmentation");
    println!("   ─────────────────────────────────────────");

    let noise_levels = [0.01, 0.05, 0.1, 0.2];
    println!(
        "   {:>8} {:>8} {:>12} {:>12}",
        "Noise", "Copies", "Total", "Mean Shift"
    );
    println!("   {}", "─".repeat(44));

    for &noise in &noise_levels {
        let augmented = augment_with_noise(&train, noise, 2, seed);
        let aug_stats = compute_stats(&augmented);

        let mean_shift: f32 = aug_stats
            .iter()
            .zip(train_stats.iter())
            .map(|(a, t)| (a.mean - t.mean).abs())
            .sum::<f32>()
            / NUM_FEATURES as f32;

        println!(
            "   {:>8.2} {:>8} {:>12} {:>12.5}",
            noise,
            2,
            augmented.len(),
            mean_shift
        );
    }
    println!();

    // =========================================================================
    // Section 6: Full Pipeline
    // =========================================================================
    println!("6. Full Pipeline Summary");
    println!("   ─────────────────────────────────────────");

    let augmented_train = augment_with_noise(&train, 0.05, 1, seed);
    let normalized_train = normalize(&augmented_train, &train_stats, NormStrategy::ZScore);
    let normalized_test = normalize(&test, &train_stats, NormStrategy::ZScore);

    println!("   Steps:");
    println!("   1. Generated {} raw samples", raw_data.len());
    println!("   2. Split: {} train / {} test", train.len(), test.len());
    println!(
        "   3. Augmented: {} → {} (1 noise copy)",
        train.len(),
        augmented_train.len()
    );
    println!("   4. Normalized: Z-Score (fit on train)");
    println!(
        "   Ready: {} train, {} test",
        normalized_train.len(),
        normalized_test.len()
    );
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_dataset_size() {
        let data = generate_dataset(10, 42);
        assert_eq!(data.len(), 10 * NUM_CLASSES);
    }

    #[test]
    fn test_generate_dataset_labels() {
        let data = generate_dataset(5, 42);
        for class in 0..NUM_CLASSES {
            let count = data.iter().filter(|s| s.label == class).count();
            assert_eq!(count, 5);
        }
    }

    #[test]
    fn test_compute_stats_dimensions() {
        let data = generate_dataset(10, 42);
        let stats = compute_stats(&data);
        assert_eq!(stats.len(), NUM_FEATURES);
    }

    #[test]
    fn test_minmax_normalization_bounds() {
        let data = generate_dataset(20, 42);
        let stats = compute_stats(&data);
        let normed = normalize(&data, &stats, NormStrategy::MinMax);
        for sample in &normed {
            for &val in &sample.features {
                assert!(val >= -0.01 && val <= 1.01, "MinMax out of range: {val}");
            }
        }
    }

    #[test]
    fn test_zscore_normalization_mean() {
        let data = generate_dataset(50, 42);
        let stats = compute_stats(&data);
        let normed = normalize(&data, &stats, NormStrategy::ZScore);
        let normed_stats = compute_stats(&normed);
        for s in &normed_stats {
            assert!(
                s.mean.abs() < 0.1,
                "Z-score mean should be ~0, got {}",
                s.mean
            );
        }
    }

    #[test]
    fn test_stratified_split_preserves_total() {
        let data = generate_dataset(10, 42);
        let (train, test) = stratified_split(&data, 0.2, 42);
        assert_eq!(train.len() + test.len(), data.len());
    }

    #[test]
    fn test_augmentation_increases_size() {
        let data = generate_dataset(10, 42);
        let augmented = augment_with_noise(&data, 0.1, 2, 42);
        assert_eq!(augmented.len(), data.len() * 3);
    }

    #[test]
    fn test_validate_clean_data() {
        let data = generate_dataset(10, 42);
        let issues = validate_data(&data);
        assert!(
            issues.is_empty(),
            "Clean data should have no issues: {issues:?}"
        );
    }

    #[test]
    fn test_validate_empty_data() {
        let issues = validate_data(&[]);
        assert!(!issues.is_empty());
    }
}
