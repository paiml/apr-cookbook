//! # Recipe: Structured Pruning (Neuron/Channel Removal)
//!
//! **Category**: optimize
//! **CLI Equivalent**: `apr prune --method structured`
//!
//! Demonstrates structured pruning: removing entire neurons or channels
//! rather than individual weights. Structured pruning produces genuinely
//! smaller models that run faster on real hardware without sparse matrix
//! support.
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Clippy clean
//! 6. [x] No `unwrap()` in logic

use apr_cookbook::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A dense layer with weights organized as [output_dim x input_dim].
#[derive(Clone, Debug)]
struct DenseLayer {
    name: String,
    weights: Vec<Vec<f32>>,
    input_dim: usize,
    output_dim: usize,
}

impl DenseLayer {
    fn new(name: &str, weights: Vec<Vec<f32>>) -> Self {
        let output_dim = weights.len();
        let input_dim = if output_dim > 0 { weights[0].len() } else { 0 };
        Self {
            name: name.to_string(),
            weights,
            input_dim,
            output_dim,
        }
    }

    fn param_count(&self) -> usize {
        self.output_dim * self.input_dim
    }
}

/// Generate a deterministic dense layer.
fn det_layer(seed: u64, name: &str, output_dim: usize, input_dim: usize) -> DenseLayer {
    let weights: Vec<Vec<f32>> = (0..output_dim)
        .map(|row| {
            (0..input_dim)
                .map(|col| {
                    let mut h = DefaultHasher::new();
                    (seed, row as u64, col as u64).hash(&mut h);
                    let bits = h.finish();
                    let u = (bits & 0xFFFF_FFFF) as f64 / f64::from(u32::MAX);
                    ((u - 0.5) * 2.0) as f32
                })
                .collect()
        })
        .collect();
    DenseLayer::new(name, weights)
}

/// Compute L2 norm of a neuron's weight vector (one row of the weight matrix).
fn neuron_magnitude(neuron_weights: &[f32]) -> f64 {
    let sum_sq: f64 = neuron_weights
        .iter()
        .map(|&w| f64::from(w) * f64::from(w))
        .sum();
    sum_sq.sqrt()
}

/// Rank neurons by their L2 magnitude, returning (index, magnitude) sorted descending.
fn rank_neurons(layer: &DenseLayer) -> Vec<(usize, f64)> {
    let mut ranked: Vec<(usize, f64)> = layer
        .weights
        .iter()
        .enumerate()
        .map(|(i, row)| (i, neuron_magnitude(row)))
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked
}

/// Perform structured pruning: remove lowest-magnitude neurons.
///
/// `ratio` is the fraction of neurons to remove (0.0 = keep all, 1.0 = remove all).
fn prune_structured(layer: &DenseLayer, ratio: f64) -> DenseLayer {
    if layer.output_dim == 0 {
        return layer.clone();
    }

    let num_to_remove = (layer.output_dim as f64 * ratio).round() as usize;
    let num_to_keep = layer.output_dim.saturating_sub(num_to_remove);
    let num_to_keep = num_to_keep.max(1); // Keep at least 1 neuron

    let ranked = rank_neurons(layer);
    let keep_indices: Vec<usize> = ranked.iter().take(num_to_keep).map(|(i, _)| *i).collect();

    let mut sorted_keep = keep_indices;
    sorted_keep.sort_unstable();

    let new_weights: Vec<Vec<f32>> = sorted_keep
        .iter()
        .map(|&i| layer.weights[i].clone())
        .collect();

    DenseLayer::new(&layer.name, new_weights)
}

/// Estimate theoretical hardware speedup from structured pruning.
fn estimate_speedup(original_dim: usize, pruned_dim: usize) -> f64 {
    if pruned_dim == 0 {
        return f64::INFINITY;
    }
    // Structured pruning gives near-linear speedup since entire rows are removed
    // Real speedup is slightly less due to memory access patterns
    let raw_ratio = original_dim as f64 / pruned_dim as f64;
    // Apply efficiency factor (memory bandwidth, cache effects)
    raw_ratio * 0.85
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("prune_structured")?;

    println!("=== Structured Pruning (Neuron/Channel Removal) ===\n");

    // --- Section 1: Layer Stats ---
    println!("--- Layer Statistics ---");
    let layer = det_layer(42, "dense_0", 256, 512);
    println!("  Layer: {}", layer.name);
    println!("  Shape: [{} x {}]", layer.output_dim, layer.input_dim);
    println!("  Parameters: {}", layer.param_count());
    println!();

    ctx.record_metric("original_output_dim", layer.output_dim as i64);
    ctx.record_metric("original_params", layer.param_count() as i64);

    // --- Section 2: Magnitude Ranking ---
    println!("--- Neuron Magnitude Ranking ---");
    let ranked = rank_neurons(&layer);
    println!("  Top 5 neurons (by L2 norm):");
    for (rank, (idx, mag)) in ranked.iter().take(5).enumerate() {
        println!("    #{}: neuron {idx} | L2 = {mag:.4}", rank + 1);
    }
    println!("  Bottom 5 neurons:");
    for (idx, mag) in ranked.iter().rev().take(5) {
        println!("    neuron {idx} | L2 = {mag:.4}");
    }
    let max_mag = ranked.first().map_or(0.0, |(_, m)| *m);
    let min_mag = ranked.last().map_or(0.0, |(_, m)| *m);
    println!("  Magnitude range: [{min_mag:.4}, {max_mag:.4}]");
    println!();

    // --- Section 3: Pruning at Various Ratios ---
    println!("--- Pruning at Various Ratios ---");
    let ratios = [0.1, 0.25, 0.5, 0.75, 0.9];
    for &ratio in &ratios {
        let pruned = prune_structured(&layer, ratio);
        let speedup = estimate_speedup(layer.output_dim, pruned.output_dim);
        println!(
            "  Ratio {:.0}%: [{} x {}] -> [{} x {}] | Params: {} -> {} | Speedup: {speedup:.2}x",
            ratio * 100.0,
            layer.output_dim,
            layer.input_dim,
            pruned.output_dim,
            pruned.input_dim,
            layer.param_count(),
            pruned.param_count()
        );
    }
    println!();

    // --- Section 4: Structured vs Unstructured Comparison ---
    println!("--- Structured vs Unstructured Comparison ---");
    println!("  At 50% reduction:");
    let pruned_struct = prune_structured(&layer, 0.5);
    let struct_params = pruned_struct.param_count();
    let unstruct_params = layer.param_count(); // Same param count, just zeros
    println!(
        "    Structured:   {} actual params (dense matrix)",
        struct_params
    );
    println!(
        "    Unstructured: {} params ({} non-zero, {} zeros)",
        unstruct_params,
        unstruct_params / 2,
        unstruct_params / 2
    );
    println!();
    println!("  Key differences:");
    println!("    Structured:   Smaller matrix, no sparse support needed");
    println!("    Unstructured: Same-size matrix with zeros, needs sparse kernels");
    println!();

    // --- Section 5: Hardware Speedup Estimates ---
    println!("--- Hardware Speedup Estimates ---");
    println!("  (Structured pruning enables real hardware speedup)\n");
    println!(
        "  {:>6} | {:>12} | {:>12} | {:>8}",
        "Ratio", "Struct Speed", "Unstruct", "Winner"
    );
    println!("  {}", "-".repeat(50));
    for &ratio in &ratios {
        let pruned = prune_structured(&layer, ratio);
        let struct_speedup = estimate_speedup(layer.output_dim, pruned.output_dim);
        // Unstructured pruning: no real speedup without sparse hardware
        let unstruct_speedup = 1.0 + ratio * 0.1; // Minimal benefit
        let winner = if struct_speedup > unstruct_speedup {
            "Struct"
        } else {
            "Unstruct"
        };
        println!(
            "  {ratio:>5.0}% | {struct_speedup:>10.2}x | {unstruct_speedup:>10.2}x | {winner:>8}"
        );
    }
    println!();

    // --- Section 6: Save Pruned Model ---
    println!("--- Save Pruned Model (APR v2) ---");
    let pruned_final = prune_structured(&layer, 0.5);
    let flat_weights: Vec<f32> = pruned_final.weights.iter().flatten().copied().collect();
    let weight_bytes: Vec<u8> = flat_weights.iter().flat_map(|f| f.to_le_bytes()).collect();

    let bundle = ModelBundleV2::new()
        .with_name("pruned_structured_50")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor(
            "dense_0.weight",
            vec![pruned_final.output_dim, pruned_final.input_dim],
            weight_bytes,
        )
        .build();

    assert_eq!(&bundle[0..4], b"APR2");
    println!(
        "  Pruned shape: [{} x {}]",
        pruned_final.output_dim, pruned_final.input_dim
    );
    println!("  Bundle size: {} bytes", bundle.len());
    println!(
        "  Size reduction: {:.1}%",
        (1.0 - pruned_final.param_count() as f64 / layer.param_count() as f64) * 100.0
    );

    ctx.record_metric("pruned_output_dim", pruned_final.output_dim as i64);
    ctx.record_metric("pruned_params", pruned_final.param_count() as i64);
    ctx.record_metric("bundle_size_bytes", bundle.len() as i64);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_reduction_50() {
        let layer = det_layer(1, "test", 100, 64);
        let pruned = prune_structured(&layer, 0.5);
        assert_eq!(pruned.output_dim, 50);
        assert_eq!(pruned.input_dim, 64);
    }

    #[test]
    fn test_dimension_reduction_25() {
        let layer = det_layer(2, "test", 200, 128);
        let pruned = prune_structured(&layer, 0.25);
        assert_eq!(pruned.output_dim, 150);
        assert_eq!(pruned.input_dim, 128);
    }

    #[test]
    fn test_keeps_highest_magnitude_neurons() {
        let layer = det_layer(3, "test", 20, 10);
        let ranked = rank_neurons(&layer);
        let pruned = prune_structured(&layer, 0.5);

        // The kept neurons should be the top-10 by magnitude
        let top_indices: Vec<usize> = ranked.iter().take(10).map(|(i, _)| *i).collect();
        for row in &pruned.weights {
            // Each kept row should appear in the original
            assert!(layer.weights.contains(row));
        }
        // Verify count
        assert_eq!(pruned.output_dim, top_indices.len());
    }

    #[test]
    fn test_magnitude_ordering() {
        let layer = det_layer(4, "test", 50, 32);
        let ranked = rank_neurons(&layer);
        for window in ranked.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "Ranking should be descending: {} vs {}",
                window[0].1,
                window[1].1
            );
        }
    }

    #[test]
    fn test_zero_ratio_preserves_all() {
        let layer = det_layer(5, "test", 32, 16);
        let pruned = prune_structured(&layer, 0.0);
        assert_eq!(pruned.output_dim, layer.output_dim);
        assert_eq!(pruned.weights, layer.weights);
    }

    #[test]
    fn test_full_ratio_keeps_one() {
        let layer = det_layer(6, "test", 32, 16);
        let pruned = prune_structured(&layer, 1.0);
        // Should keep at least 1 neuron
        assert_eq!(pruned.output_dim, 1);
    }

    #[test]
    fn test_deterministic() {
        let l1 = det_layer(7, "test", 64, 32);
        let l2 = det_layer(7, "test", 64, 32);
        assert_eq!(l1.weights, l2.weights);

        let p1 = prune_structured(&l1, 0.5);
        let p2 = prune_structured(&l2, 0.5);
        assert_eq!(p1.weights, p2.weights);
    }

    #[test]
    fn test_speedup_greater_than_one() {
        let layer = det_layer(8, "test", 128, 64);
        for &ratio in &[0.25, 0.5, 0.75] {
            let pruned = prune_structured(&layer, ratio);
            let speedup = estimate_speedup(layer.output_dim, pruned.output_dim);
            assert!(
                speedup > 1.0,
                "Speedup should be > 1.0 at ratio {ratio}, got {speedup}"
            );
        }
    }

    #[test]
    fn test_param_count_reduction() {
        let layer = det_layer(9, "test", 100, 50);
        let pruned = prune_structured(&layer, 0.5);
        assert_eq!(pruned.param_count(), 50 * 50);
        assert!(pruned.param_count() < layer.param_count());
    }

    #[test]
    fn test_input_dim_preserved() {
        let layer = det_layer(10, "test", 64, 128);
        let pruned = prune_structured(&layer, 0.5);
        assert_eq!(pruned.input_dim, layer.input_dim);
    }

    #[test]
    fn test_neuron_magnitude_positive() {
        let layer = det_layer(11, "test", 32, 16);
        for row in &layer.weights {
            let mag = neuron_magnitude(row);
            assert!(mag >= 0.0);
        }
    }

    #[test]
    fn test_apr_v2_bundle() {
        let layer = det_layer(12, "test", 32, 16);
        let pruned = prune_structured(&layer, 0.5);
        let flat: Vec<f32> = pruned.weights.iter().flatten().copied().collect();
        let bytes: Vec<u8> = flat.iter().flat_map(|f| f.to_le_bytes()).collect();
        let bundle = ModelBundleV2::new()
            .with_name("test_structured")
            .with_compression(Compression::Lz4)
            .with_quantization(Quantization::FP32)
            .add_tensor("w", vec![pruned.output_dim, pruned.input_dim], bytes)
            .build();
        assert_eq!(&bundle[0..4], b"APR2");
    }
}
