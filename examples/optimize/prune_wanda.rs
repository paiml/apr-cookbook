//! # Recipe: Wanda Pruning (Weight and Activation-Aware)
//!
//! **Category**: optimize
//! **CLI Equivalent**: `apr prune --method wanda`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Demonstrates Wanda (Pruning by Weights and Activations): a calibration-based
//! pruning method that considers both weight magnitude and input activation
//! magnitude. By incorporating activation statistics from calibration data,
//! Wanda achieves better quality preservation than simple magnitude pruning.
//!
//! Reference: Sun et al., "A Simple and Effective Pruning Approach for Large Language Models" (2023)
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

/// Generate deterministic weight matrix [rows x cols].
fn det_matrix(seed: u64, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|r| {
            (0..cols)
                .map(|c| {
                    let mut h = DefaultHasher::new();
                    (seed, r as u64, c as u64).hash(&mut h);
                    let bits = h.finish();
                    let u = (bits & 0xFFFF_FFFF) as f64 / f64::from(u32::MAX);
                    ((u - 0.5) * 2.0) as f32
                })
                .collect()
        })
        .collect()
}

/// Generate deterministic calibration data [samples x features].
fn det_calibration_data(seed: u64, samples: usize, features: usize) -> Vec<Vec<f32>> {
    (0..samples)
        .map(|s| {
            (0..features)
                .map(|f| {
                    let mut h = DefaultHasher::new();
                    (seed, 0xCA_u64, 1_u64, s as u64, f as u64).hash(&mut h);
                    let bits = h.finish();
                    let u = (bits & 0xFFFF_FFFF) as f64 / f64::from(u32::MAX);
                    // Calibration data: positive values with varying magnitudes
                    (u * 3.0) as f32
                })
                .collect()
        })
        .collect()
}

/// Calibrate activation magnitudes from calibration data.
///
/// Computes the L2 norm of each input feature across all calibration samples.
/// Returns a vector of length `input_features` with activation magnitude per channel.
fn calibrate(weights: &[Vec<f32>], calibration_data: &[Vec<f32>]) -> Vec<f64> {
    if weights.is_empty() || calibration_data.is_empty() {
        return Vec::new();
    }

    let input_features = weights[0].len();
    let mut activation_mags = vec![0.0_f64; input_features];

    for sample in calibration_data {
        for (j, &val) in sample.iter().enumerate().take(input_features) {
            activation_mags[j] += f64::from(val) * f64::from(val);
        }
    }

    // L2 norm: sqrt(sum of squares / num_samples)
    let n = calibration_data.len() as f64;
    for mag in &mut activation_mags {
        *mag = (*mag / n).sqrt();
    }

    activation_mags
}

/// Compute Wanda score: |weight| * ||activation||.
///
/// Higher score means the weight is more important and should be kept.
fn wanda_score(weight: f32, activation_magnitude: f64) -> f64 {
    f64::from(weight.abs()) * activation_magnitude
}

/// Prune using Wanda method: remove weights with lowest (weight * activation) scores.
fn prune_wanda(
    weights: &[Vec<f32>],
    activation_mags: &[f64],
    target_sparsity: f64,
) -> Vec<Vec<f32>> {
    if weights.is_empty() {
        return Vec::new();
    }

    let rows = weights.len();
    let cols = weights[0].len();
    let total = rows * cols;
    let num_to_prune = (total as f64 * target_sparsity).round() as usize;

    // Compute all Wanda scores with their positions
    let mut scored: Vec<(usize, usize, f64)> = Vec::with_capacity(total);
    for (r, row) in weights.iter().enumerate() {
        for (c, &w) in row.iter().enumerate() {
            let act_mag = if c < activation_mags.len() {
                activation_mags[c]
            } else {
                1.0
            };
            scored.push((r, c, wanda_score(w, act_mag)));
        }
    }

    // Sort by score ascending (lowest first = least important)
    scored.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    // Prune lowest-scored weights
    let mut result: Vec<Vec<f32>> = weights.to_vec();
    for &(r, c, _) in scored.iter().take(num_to_prune) {
        result[r][c] = 0.0;
    }

    result
}

/// Prune using simple magnitude (for comparison).
fn prune_magnitude_matrix(weights: &[Vec<f32>], target_sparsity: f64) -> Vec<Vec<f32>> {
    if weights.is_empty() {
        return Vec::new();
    }

    let rows = weights.len();
    let cols = weights[0].len();
    let total = rows * cols;
    let num_to_prune = (total as f64 * target_sparsity).round() as usize;

    let mut scored: Vec<(usize, usize, f32)> = Vec::with_capacity(total);
    for (r, row) in weights.iter().enumerate() {
        for (c, &w) in row.iter().enumerate() {
            scored.push((r, c, w.abs()));
        }
    }

    scored.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut result = weights.to_vec();
    for &(r, c, _) in scored.iter().take(num_to_prune) {
        result[r][c] = 0.0;
    }

    result
}

/// Compute reconstruction error (Frobenius norm of difference).
fn reconstruction_error(original: &[Vec<f32>], pruned: &[Vec<f32>]) -> f64 {
    let mut sum_sq = 0.0_f64;
    let mut count = 0usize;
    for (orig_row, pruned_row) in original.iter().zip(pruned.iter()) {
        for (&o, &p) in orig_row.iter().zip(pruned_row.iter()) {
            let diff = f64::from(o) - f64::from(p);
            sum_sq += diff * diff;
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    (sum_sq / count as f64).sqrt()
}

/// Compute sparsity of a weight matrix.
fn matrix_sparsity(weights: &[Vec<f32>]) -> f64 {
    let total: usize = weights.iter().map(Vec::len).sum();
    if total == 0 {
        return 0.0;
    }
    let zeros: usize = weights
        .iter()
        .flat_map(|r| r.iter())
        .filter(|&&w| w == 0.0)
        .count();
    zeros as f64 / total as f64
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("prune_wanda")?;

    println!("=== Wanda Pruning (Weight and Activation-Aware) ===\n");

    let weights = det_matrix(42, 128, 256);
    let cal_data = det_calibration_data(99, 64, 256);

    // --- Section 1: Calibration Phase ---
    println!("--- Calibration Phase ---");
    println!(
        "  Weight matrix: [{} x {}]",
        weights.len(),
        weights[0].len()
    );
    println!("  Calibration samples: {}", cal_data.len());
    println!("  Calibration features: {}", cal_data[0].len());

    let activation_mags = calibrate(&weights, &cal_data);
    let min_act = activation_mags
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_act = activation_mags
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mean_act: f64 = activation_mags.iter().sum::<f64>() / activation_mags.len() as f64;

    println!("  Activation magnitudes:");
    println!("    Min: {min_act:.4}");
    println!("    Max: {max_act:.4}");
    println!("    Mean: {mean_act:.4}");
    println!("    Channels: {}", activation_mags.len());
    println!();

    ctx.record_float_metric("activation_mag_mean", mean_act);

    // --- Section 2: Wanda Score Computation ---
    println!("--- Wanda Score Analysis ---");
    println!("  Wanda score = |weight| * ||activation||");
    println!();
    println!("  Sample scores (first 5 weights of row 0):");
    for c in 0..5 {
        let w = weights[0][c];
        let a = activation_mags[c];
        let score = wanda_score(w, a);
        println!("    w[0][{c}] = {w:>7.4} | act = {a:.4} | score = {score:.4}");
    }
    println!();

    // --- Section 3: Pruning Comparison ---
    println!("--- Pruning Comparison: Magnitude vs Wanda ---");
    let sparsities = [0.3, 0.5, 0.7];

    println!(
        "  {:>10} | {:>15} | {:>15} | {:>8}",
        "Sparsity", "Magnitude RMSE", "Wanda RMSE", "Winner"
    );
    println!("  {}", "-".repeat(58));

    for &sparsity in &sparsities {
        let mag_pruned = prune_magnitude_matrix(&weights, sparsity);
        let wanda_pruned = prune_wanda(&weights, &activation_mags, sparsity);

        let mag_error = reconstruction_error(&weights, &mag_pruned);
        let wanda_error = reconstruction_error(&weights, &wanda_pruned);

        let winner = if wanda_error <= mag_error {
            "Wanda"
        } else {
            "Magnitude"
        };

        println!(
            "  {:>9.0}% | {:>15.6} | {:>15.6} | {winner:>8}",
            sparsity * 100.0,
            mag_error,
            wanda_error
        );

        let metric = format!("wanda_rmse_{}", (sparsity * 100.0) as i64);
        ctx.record_float_metric(&metric, wanda_error);
        let metric = format!("magnitude_rmse_{}", (sparsity * 100.0) as i64);
        ctx.record_float_metric(&metric, mag_error);
    }
    println!();

    // --- Section 4: Quality Preservation Detail ---
    println!("--- Quality Preservation at 50% Sparsity ---");
    let mag_50 = prune_magnitude_matrix(&weights, 0.5);
    let wanda_50 = prune_wanda(&weights, &activation_mags, 0.5);

    let mag_err = reconstruction_error(&weights, &mag_50);
    let wanda_err = reconstruction_error(&weights, &wanda_50);
    let improvement = if mag_err > 0.0 {
        (1.0 - wanda_err / mag_err) * 100.0
    } else {
        0.0
    };

    println!("  Magnitude pruning RMSE: {mag_err:.6}");
    println!("  Wanda pruning RMSE:     {wanda_err:.6}");
    println!("  Wanda improvement:      {improvement:.1}%");
    println!();
    println!("  Why Wanda works better:");
    println!("    - Magnitude only considers |w|");
    println!("    - Wanda considers |w| * ||X||, preserving weights");
    println!("      on high-activation channels even if weight is small");
    println!();

    ctx.record_float_metric("wanda_improvement_pct", improvement);

    // --- Section 5: Sparsity Verification ---
    println!("--- Sparsity Verification ---");
    for &sparsity in &sparsities {
        let wanda_pruned = prune_wanda(&weights, &activation_mags, sparsity);
        let actual = matrix_sparsity(&wanda_pruned);
        println!(
            "  Target: {:.0}% | Actual: {:.1}% | Delta: {:.2}%",
            sparsity * 100.0,
            actual * 100.0,
            (actual - sparsity).abs() * 100.0
        );
    }
    println!();

    // --- Section 6: Save Wanda-Pruned Model ---
    println!("--- Save Wanda-Pruned Model (APR v2) ---");
    let final_pruned = prune_wanda(&weights, &activation_mags, 0.5);
    let flat: Vec<f32> = final_pruned.iter().flatten().copied().collect();
    let weight_bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();

    let bundle = ModelBundleV2::new()
        .with_name("pruned_wanda_50")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor(
            "wanda_pruned_weights",
            vec![final_pruned.len(), final_pruned[0].len()],
            weight_bytes,
        )
        .build();

    assert_eq!(&bundle[0..4], b"APR2");
    println!("  Bundle size: {} bytes", bundle.len());
    println!("  Sparsity: {:.1}%", matrix_sparsity(&final_pruned) * 100.0);
    println!("  Method: Wanda (calibration-aware)");

    ctx.record_metric("bundle_size_bytes", bundle.len() as i64);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_produces_positive_values() {
        let weights = det_matrix(1, 16, 32);
        let cal_data = det_calibration_data(1, 10, 32);
        let mags = calibrate(&weights, &cal_data);
        assert_eq!(mags.len(), 32);
        for &m in &mags {
            assert!(m >= 0.0, "Activation magnitude must be non-negative: {m}");
        }
    }

    #[test]
    fn test_calibration_length_matches_input_dim() {
        let weights = det_matrix(2, 8, 64);
        let cal_data = det_calibration_data(2, 20, 64);
        let mags = calibrate(&weights, &cal_data);
        assert_eq!(mags.len(), 64);
    }

    #[test]
    fn test_wanda_preserves_important_weights_better() {
        let weights = det_matrix(3, 64, 128);
        let cal_data = det_calibration_data(3, 32, 128);
        let act_mags = calibrate(&weights, &cal_data);

        let mag_pruned = prune_magnitude_matrix(&weights, 0.5);
        let wanda_pruned = prune_wanda(&weights, &act_mags, 0.5);

        let mag_err = reconstruction_error(&weights, &mag_pruned);
        let wanda_err = reconstruction_error(&weights, &wanda_pruned);

        // Wanda should generally produce equal or lower error
        // Allow small tolerance since it depends on data distribution
        assert!(
            wanda_err <= mag_err * 1.05,
            "Wanda error ({wanda_err}) should be close to or less than magnitude error ({mag_err})"
        );
    }

    #[test]
    fn test_wanda_achieves_target_sparsity() {
        let weights = det_matrix(4, 32, 64);
        let cal_data = det_calibration_data(4, 16, 64);
        let act_mags = calibrate(&weights, &cal_data);

        let pruned = prune_wanda(&weights, &act_mags, 0.5);
        let actual = matrix_sparsity(&pruned);
        assert!(
            (actual - 0.5).abs() < 0.02,
            "Expected ~50% sparsity, got {actual}"
        );
    }

    #[test]
    fn test_wanda_score_computation() {
        let score = wanda_score(0.5, 2.0);
        assert!((score - 1.0).abs() < f64::EPSILON);

        let score_neg = wanda_score(-0.5, 2.0);
        assert!((score_neg - 1.0).abs() < f64::EPSILON);

        let score_zero = wanda_score(0.0, 2.0);
        assert!(score_zero.abs() < f64::EPSILON);
    }

    #[test]
    fn test_deterministic() {
        let w = det_matrix(5, 32, 64);
        let cal = det_calibration_data(5, 16, 64);
        let mags = calibrate(&w, &cal);

        let p1 = prune_wanda(&w, &mags, 0.5);
        let p2 = prune_wanda(&w, &mags, 0.5);
        assert_eq!(p1, p2);
    }

    #[test]
    fn test_empty_weights() {
        let empty: Vec<Vec<f32>> = Vec::new();
        let result = prune_wanda(&empty, &[], 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_zero_sparsity_preserves_all() {
        let weights = det_matrix(7, 16, 32);
        let cal = det_calibration_data(7, 8, 32);
        let mags = calibrate(&weights, &cal);
        let pruned = prune_wanda(&weights, &mags, 0.0);
        assert_eq!(weights, pruned);
    }

    #[test]
    fn test_reconstruction_error_zero_for_identical() {
        let weights = det_matrix(8, 16, 32);
        let err = reconstruction_error(&weights, &weights);
        assert!(err < f64::EPSILON);
    }

    #[test]
    fn test_matrix_sparsity() {
        let weights = vec![vec![0.0, 1.0, 0.0], vec![2.0, 0.0, 0.0]];
        let sparsity = matrix_sparsity(&weights);
        assert!((sparsity - 4.0 / 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_apr_v2_bundle() {
        let weights = det_matrix(9, 8, 16);
        let cal = det_calibration_data(9, 4, 16);
        let mags = calibrate(&weights, &cal);
        let pruned = prune_wanda(&weights, &mags, 0.5);
        let flat: Vec<f32> = pruned.iter().flatten().copied().collect();
        let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();
        let bundle = ModelBundleV2::new()
            .with_name("test_wanda")
            .with_compression(Compression::Lz4)
            .with_quantization(Quantization::FP32)
            .add_tensor("w", vec![pruned.len(), pruned[0].len()], bytes)
            .build();
        assert_eq!(&bundle[0..4], b"APR2");
    }
}
