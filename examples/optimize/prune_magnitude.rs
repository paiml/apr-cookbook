//! # Recipe: Magnitude-Based Unstructured Pruning
//!
//! **Category**: optimize
//! **CLI Equivalent**: `apr prune --method magnitude --target 0.5`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Demonstrates magnitude-based unstructured pruning: zeroing out the
//! smallest-magnitude weights to achieve a target sparsity. This is the
//! simplest and most widely used pruning strategy.
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Clippy clean
//! 6. [x] No `unwrap()` in logic
//!
//!
//! ## Format Variants
//! ```bash
//! apr prune model.apr          # APR native format
//! apr prune model.gguf         # GGUF (llama.cpp compatible)
//! apr prune model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Frantar, E. & Alistarh, D. (2023). *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot*. ICML. arXiv:2301.00774

use apr_cookbook::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Generate deterministic weights using a hash-based PRNG.
fn det_weights(seed: u64, count: usize) -> Vec<f32> {
    (0..count)
        .map(|i| {
            let mut h = DefaultHasher::new();
            (seed, i as u64).hash(&mut h);
            let bits = h.finish();
            // Map to [-1.0, 1.0] range with roughly normal-ish distribution
            let u = (bits & 0xFFFF_FFFF) as f64 / f64::from(u32::MAX);
            let v = ((bits >> 32) & 0xFFFF_FFFF) as f64 / f64::from(u32::MAX);
            // Box-Muller approximation via simple mapping
            let centered = (u - 0.5) * 2.0;
            let scaled = centered * (1.0 + v * 0.3);
            scaled as f32
        })
        .collect()
}

/// Prune weights by magnitude: zero out the smallest weights to reach target sparsity.
///
/// Returns a new weight vector with the smallest-magnitude weights set to zero.
fn prune_magnitude(weights: &[f32], target_sparsity: f64) -> Vec<f32> {
    if weights.is_empty() {
        return Vec::new();
    }

    let num_to_prune = (weights.len() as f64 * target_sparsity).round() as usize;
    let num_to_prune = num_to_prune.min(weights.len());

    // Sort indices by absolute magnitude (ascending)
    let mut indices_by_mag: Vec<usize> = (0..weights.len()).collect();
    indices_by_mag.sort_by(|&a, &b| {
        weights[a]
            .abs()
            .partial_cmp(&weights[b].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut pruned = weights.to_vec();
    for &idx in indices_by_mag.iter().take(num_to_prune) {
        pruned[idx] = 0.0;
    }

    pruned
}

/// Compute sparsity (fraction of zero weights).
fn compute_sparsity(weights: &[f32]) -> f64 {
    if weights.is_empty() {
        return 0.0;
    }
    let zeros = weights.iter().filter(|&&w| w == 0.0).count();
    zeros as f64 / weights.len() as f64
}

/// Compute RMSE between original and pruned weights.
fn compute_rmse(original: &[f32], pruned: &[f32]) -> f64 {
    assert_eq!(original.len(), pruned.len());
    if original.is_empty() {
        return 0.0;
    }
    let mse: f64 = original
        .iter()
        .zip(pruned.iter())
        .map(|(a, b)| {
            let diff = f64::from(*a) - f64::from(*b);
            diff * diff
        })
        .sum::<f64>()
        / original.len() as f64;
    mse.sqrt()
}

/// Render a simple ASCII histogram of weight magnitudes.
fn weight_histogram(weights: &[f32], bins: usize, label: &str) {
    let max_mag = weights.iter().map(|w| w.abs()).fold(0.0_f32, f32::max);

    if max_mag == 0.0 {
        println!("  [{label}] All weights are zero");
        return;
    }

    let bin_width = f64::from(max_mag) / bins as f64;
    let mut counts = vec![0usize; bins];

    for &w in weights {
        let mag = f64::from(w.abs());
        let bin = ((mag / bin_width) as usize).min(bins - 1);
        counts[bin] += 1;
    }

    let max_count = counts.iter().copied().max().unwrap_or(1);
    let bar_max = 40;

    println!(
        "  [{label}] Weight magnitude histogram ({} weights):",
        weights.len()
    );
    for (i, &count) in counts.iter().enumerate() {
        let lo = i as f64 * bin_width;
        let hi = (i + 1) as f64 * bin_width;
        let bar_len = if max_count > 0 {
            count * bar_max / max_count
        } else {
            0
        };
        let bar: String = "#".repeat(bar_len);
        println!("    [{lo:>5.2}, {hi:>5.2}) | {bar:<40} ({count})");
    }
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("prune_magnitude")?;

    // --- Section 1: Weight Distribution ---
    println!("=== Magnitude-Based Unstructured Pruning ===\n");
    println!("--- Weight Distribution ---");

    let weights = det_weights(42, 1024);
    let mean: f64 = weights.iter().map(|w| f64::from(*w)).sum::<f64>() / weights.len() as f64;
    let variance: f64 = weights
        .iter()
        .map(|w| {
            let d = f64::from(*w) - mean;
            d * d
        })
        .sum::<f64>()
        / weights.len() as f64;
    let std_dev = variance.sqrt();

    println!("  Weights: {} parameters", weights.len());
    println!("  Mean: {mean:.4}");
    println!("  Std Dev: {std_dev:.4}");
    println!(
        "  Min: {:.4}",
        weights.iter().copied().fold(f32::INFINITY, f32::min)
    );
    println!(
        "  Max: {:.4}",
        weights.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    );
    println!();

    weight_histogram(&weights, 8, "Original");
    println!();

    ctx.record_metric("weight_count", weights.len() as i64);

    // --- Section 2: Pruning at Multiple Sparsities ---
    println!("--- Pruning at Multiple Sparsities ---");
    let sparsities = [0.1, 0.3, 0.5, 0.7, 0.9];

    for &target in &sparsities {
        let pruned = prune_magnitude(&weights, target);
        let actual = compute_sparsity(&pruned);
        let nonzero = pruned.iter().filter(|&&w| w != 0.0).count();
        println!(
            "  Target: {:.0}% | Actual: {:.1}% | Non-zero: {}/{} | Zeros: {}",
            target * 100.0,
            actual * 100.0,
            nonzero,
            pruned.len(),
            pruned.len() - nonzero
        );
    }
    println!();

    // --- Section 3: Histogram After 50% Pruning ---
    println!("--- After 50% Pruning ---");
    let pruned_50 = prune_magnitude(&weights, 0.5);
    weight_histogram(&pruned_50, 8, "Pruned@50%");
    println!();

    // --- Section 4: Quality Impact (RMSE) ---
    println!("--- Quality Impact (RMSE from Original) ---");
    for &target in &sparsities {
        let pruned = prune_magnitude(&weights, target);
        let rmse = compute_rmse(&weights, &pruned);
        let bar_len = (rmse * 50.0).round() as usize;
        let bar: String = "|".repeat(bar_len.min(50));
        println!("  Sparsity {:.0}%: RMSE = {rmse:.6}  {bar}", target * 100.0);

        let metric_name = format!("rmse_at_{}", (target * 100.0) as i64);
        ctx.record_float_metric(&metric_name, rmse);
    }
    println!();

    // --- Section 5: Save Pruned Model to APR v2 ---
    println!("--- Save Pruned Model (APR v2) ---");
    let pruned_final = prune_magnitude(&weights, 0.5);
    let final_sparsity = compute_sparsity(&pruned_final);

    let weight_bytes: Vec<u8> = pruned_final.iter().flat_map(|f| f.to_le_bytes()).collect();

    let bundle = ModelBundleV2::new()
        .with_name("pruned_magnitude_50")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("pruned_weights", vec![1, pruned_final.len()], weight_bytes)
        .build();

    assert_eq!(&bundle[0..4], b"APR2");
    println!("  Bundle size: {} bytes", bundle.len());
    println!("  Format: APR v2 (LZ4 compressed)");
    println!("  Final sparsity: {:.1}%", final_sparsity * 100.0);
    println!("  Compression advantage: sparse tensors compress well with LZ4");

    ctx.record_metric("bundle_size_bytes", bundle.len() as i64);
    ctx.record_float_metric("final_sparsity", final_sparsity);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preserves_length() {
        let weights = det_weights(1, 256);
        let pruned = prune_magnitude(&weights, 0.5);
        assert_eq!(weights.len(), pruned.len());
    }

    #[test]
    fn test_achieves_target_sparsity_50() {
        let weights = det_weights(2, 1000);
        let pruned = prune_magnitude(&weights, 0.5);
        let actual = compute_sparsity(&pruned);
        assert!(
            (actual - 0.5).abs() < 0.01,
            "Expected ~50% sparsity, got {actual}"
        );
    }

    #[test]
    fn test_achieves_target_sparsity_90() {
        let weights = det_weights(3, 1000);
        let pruned = prune_magnitude(&weights, 0.9);
        let actual = compute_sparsity(&pruned);
        assert!(
            (actual - 0.9).abs() < 0.01,
            "Expected ~90% sparsity, got {actual}"
        );
    }

    #[test]
    fn test_zero_sparsity_preserves_all() {
        let weights = det_weights(4, 128);
        let pruned = prune_magnitude(&weights, 0.0);
        assert_eq!(weights, pruned);
    }

    #[test]
    fn test_full_sparsity_zeros_all() {
        let weights = det_weights(5, 128);
        let pruned = prune_magnitude(&weights, 1.0);
        assert!(pruned.iter().all(|&w| w == 0.0));
    }

    #[test]
    fn test_zero_weights_are_smallest() {
        let weights = det_weights(6, 256);
        let pruned = prune_magnitude(&weights, 0.5);
        // All surviving (non-zero) weights should have magnitude >= all pruned weights
        let surviving_min = pruned
            .iter()
            .filter(|&&w| w != 0.0)
            .map(|w| w.abs())
            .fold(f32::INFINITY, f32::min);

        for (i, (&orig, &pr)) in weights.iter().zip(pruned.iter()).enumerate() {
            if pr == 0.0 {
                assert!(
                    orig.abs() <= surviving_min + f32::EPSILON,
                    "Pruned weight at {i} had magnitude {} > surviving min {surviving_min}",
                    orig.abs()
                );
            }
        }
    }

    #[test]
    fn test_rmse_increases_with_sparsity() {
        let weights = det_weights(7, 512);
        let rmses: Vec<f64> = [0.1, 0.3, 0.5, 0.7, 0.9]
            .iter()
            .map(|&s| {
                let pruned = prune_magnitude(&weights, s);
                compute_rmse(&weights, &pruned)
            })
            .collect();

        for window in rmses.windows(2) {
            assert!(
                window[1] >= window[0],
                "RMSE should increase: {} vs {}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn test_rmse_zero_at_no_pruning() {
        let weights = det_weights(8, 256);
        let pruned = prune_magnitude(&weights, 0.0);
        let rmse = compute_rmse(&weights, &pruned);
        assert!(rmse < f64::EPSILON, "RMSE should be 0 with no pruning");
    }

    #[test]
    fn test_deterministic_output() {
        let w1 = det_weights(99, 512);
        let w2 = det_weights(99, 512);
        assert_eq!(w1, w2, "det_weights must be deterministic");

        let p1 = prune_magnitude(&w1, 0.5);
        let p2 = prune_magnitude(&w2, 0.5);
        assert_eq!(p1, p2, "prune_magnitude must be deterministic");
    }

    #[test]
    fn test_empty_weights() {
        let pruned = prune_magnitude(&[], 0.5);
        assert!(pruned.is_empty());
    }

    #[test]
    fn test_sparsity_computation() {
        let weights = vec![0.0, 1.0, 0.0, 2.0, 0.0];
        let sparsity = compute_sparsity(&weights);
        assert!((sparsity - 0.6).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apr_v2_bundle_valid() {
        let weights = det_weights(10, 64);
        let pruned = prune_magnitude(&weights, 0.5);
        let bytes: Vec<u8> = pruned.iter().flat_map(|f| f.to_le_bytes()).collect();
        let bundle = ModelBundleV2::new()
            .with_name("test_pruned")
            .with_compression(Compression::Lz4)
            .with_quantization(Quantization::FP32)
            .add_tensor("w", vec![1, pruned.len()], bytes)
            .build();
        assert_eq!(&bundle[0..4], b"APR2");
    }
}
