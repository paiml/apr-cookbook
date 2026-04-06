#![allow(unused_imports)]
//! Data Preprocessing Pipeline Example
//! **CLI Equivalent**: `apr data`
//! Contract: contracts/recipe-iiur-v1.yaml
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
//!
//!
//! ## Format Variants
//! ```bash
//! apr finetune model.apr          # APR native format
//! apr finetune model.gguf         # GGUF (llama.cpp compatible)
//! apr finetune model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
