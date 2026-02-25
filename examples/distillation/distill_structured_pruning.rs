//! **DEPRECATED**: This example is superseded by `examples/optimize/prune_structured.rs`
//! which mirrors the `apr prune --method structured` CLI workflow.
//!
//! # Recipe: Structured Pruning for Model Compression
//!
//! **Category**: Model Distillation
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Remove entire neurons/channels (structured pruning) rather than individual
//! weights (unstructured pruning) for hardware-friendly model compression.
//!
//! ## Run Command
//! ```bash
//! cargo run --example distill_structured_pruning
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("distill_structured_pruning")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Structured pruning: removing entire neurons/channels");
    println!();

    // ── Section 1: Build dense model with layer-wise weight distributions ──
    println!("--- Section 1: Dense Model with Layer-wise Weight Distributions ---");
    println!();

    let layer_specs = vec![
        ("embed", 784, 512),
        ("hidden1", 512, 256),
        ("hidden2", 256, 128),
        ("output", 128, 10),
    ];

    let mut layers = Vec::new();
    for (name, input_dim, output_dim) in &layer_specs {
        let layer = build_dense_layer(name, *input_dim, *output_dim);
        layers.push(layer);
    }

    println!("Dense Model Architecture:");
    println!("{:-<65}", "");
    println!(
        "{:<10} {:>8} {:>8} {:>10} {:>10} {:>12}",
        "Layer", "In", "Out", "Params", "Mean|W|", "Std|W|"
    );
    println!("{:-<65}", "");

    let mut total_params: usize = 0;
    for layer in &layers {
        let stats = compute_weight_stats(layer);
        let params = layer.input_dim * layer.output_dim;
        total_params += params;
        println!(
            "{:<10} {:>8} {:>8} {:>10} {:>10.4} {:>12.4}",
            layer.name,
            layer.input_dim,
            layer.output_dim,
            params,
            stats.mean_magnitude,
            stats.std_magnitude
        );
    }
    println!("{:-<65}", "");
    println!("Total parameters: {}", total_params);
    println!();

    ctx.record_metric("total_params", total_params as i64);

    // ── Section 2: Magnitude-based neuron pruning at various ratios ──
    println!("--- Section 2: Magnitude-based Neuron Pruning ---");
    println!();

    let pruning_ratios = vec![0.1, 0.2, 0.3, 0.5, 0.7];
    println!("Pruning hidden1 layer at various ratios:");
    println!("{:-<70}", "");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>12}",
        "Ratio", "Orig Neurons", "Kept", "Pruned", "Accuracy"
    );
    println!("{:-<70}", "");

    let mut magnitude_results = Vec::new();
    for &ratio in &pruning_ratios {
        let config = PruningConfig {
            target_ratio: ratio,
            strategy: PruningStrategy::Magnitude,
        };
        let result = prune_layer(&layers[1], &config)?;
        magnitude_results.push(result.clone());

        println!(
            "{:>7.0}% {:>12} {:>12} {:>12} {:>11.2}%",
            ratio * 100.0,
            result.original_neurons,
            result.kept_neurons,
            result.original_neurons - result.kept_neurons,
            result.estimated_accuracy * 100.0
        );
    }
    println!("{:-<70}", "");
    println!();

    // ── Section 3: Layer sensitivity analysis ──
    println!("--- Section 3: Layer Sensitivity Analysis ---");
    println!();

    println!("Accuracy impact of pruning each layer at 30%:");
    println!("{:-<65}", "");
    println!(
        "{:<10} {:>15} {:>15} {:>15}",
        "Layer", "Impact/1%", "Recommended", "Sensitivity"
    );
    println!("{:-<65}", "");

    let mut sensitivities = Vec::new();
    for layer in &layers {
        let sensitivity = analyze_layer_sensitivity(layer)?;
        sensitivities.push(sensitivity.clone());

        let sensitivity_label = if sensitivity.accuracy_impact_per_pct > 0.3 {
            "HIGH"
        } else if sensitivity.accuracy_impact_per_pct > 0.15 {
            "MEDIUM"
        } else {
            "LOW"
        };

        println!(
            "{:<10} {:>14.3}% {:>14.0}% {:>15}",
            sensitivity.layer_name,
            sensitivity.accuracy_impact_per_pct,
            sensitivity.recommended_ratio * 100.0,
            sensitivity_label
        );
    }
    println!("{:-<65}", "");
    println!();

    ctx.record_float_metric(
        "max_sensitivity",
        sensitivities
            .iter()
            .map(|s| s.accuracy_impact_per_pct)
            .fold(0.0f64, f64::max),
    );

    // ── Section 4: Iterative pruning with simulated fine-tuning ──
    println!("--- Section 4: Iterative Pruning with Fine-Tuning ---");
    println!();

    let target_ratio = 0.5;
    let num_rounds = 5;

    println!(
        "Target: {:.0}% pruning in {} rounds with fine-tuning between each",
        target_ratio * 100.0,
        num_rounds
    );
    println!("{:-<65}", "");
    println!(
        "{:>6} {:>10} {:>12} {:>12} {:>12}",
        "Round", "Pruned%", "Accuracy", "Recovery", "Params"
    );
    println!("{:-<65}", "");

    let iterative_results = iterative_prune_with_finetuning(&layers[1], target_ratio, num_rounds)?;

    for (i, round) in iterative_results.iter().enumerate() {
        let pruned_pct = 1.0 - (round.pruned_params as f64 / round.original_params.max(1) as f64);
        println!(
            "{:>6} {:>9.1}% {:>11.2}% {:>11.2}% {:>12}",
            i + 1,
            (1.0 - pruned_pct) * 100.0,
            round.accuracy_after * 100.0,
            (round.accuracy_after - round.accuracy_before + round.accuracy_before) * 100.0,
            round.pruned_params
        );
    }
    println!("{:-<65}", "");

    let final_round = iterative_results
        .last()
        .ok_or_else(|| CookbookError::invalid_format("No iterative results"))?;
    println!(
        "Final accuracy after iterative pruning: {:.2}%",
        final_round.accuracy_after * 100.0
    );
    println!();

    ctx.record_float_metric("iterative_final_accuracy", final_round.accuracy_after);

    // ── Section 5: Structured vs unstructured comparison ──
    println!("--- Section 5: Structured vs Unstructured Comparison ---");
    println!();

    println!("{:-<75}", "");
    println!(
        "{:<14} {:>10} {:>12} {:>12} {:>10} {:>10}",
        "Method", "Ratio", "Accuracy", "Size(KB)", "Speedup", "HW-Friendly"
    );
    println!("{:-<75}", "");

    let comparison_ratios = vec![0.3, 0.5, 0.7];
    for &ratio in &comparison_ratios {
        let structured = compare_pruning_method(PruningStrategy::Magnitude, ratio, total_params)?;
        let unstructured = compare_pruning_method(PruningStrategy::Random, ratio, total_params)?;

        println!(
            "{:<14} {:>9.0}% {:>11.2}% {:>10.1} {:>9.2}x {:>10}",
            "Structured",
            ratio * 100.0,
            structured.accuracy_after * 100.0,
            structured.pruned_params as f64 * 4.0 / 1024.0,
            structured.speedup,
            "Yes"
        );
        println!(
            "{:<14} {:>9.0}% {:>11.2}% {:>10.1} {:>9.2}x {:>10}",
            "Unstructured",
            ratio * 100.0,
            unstructured.accuracy_after * 100.0,
            unstructured.pruned_params as f64 * 4.0 / 1024.0,
            unstructured.speedup,
            "No"
        );
    }
    println!("{:-<75}", "");
    println!();

    // ── Section 6: Compression summary and recommendations ──
    println!("--- Section 6: Compression Summary and Recommendations ---");
    println!();

    let best_sensitivity = sensitivities
        .iter()
        .min_by(|a, b| {
            a.accuracy_impact_per_pct
                .partial_cmp(&b.accuracy_impact_per_pct)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| CookbookError::invalid_format("No sensitivities computed"))?;

    let original_size_kb = total_params as f64 * 4.0 / 1024.0;
    let recommended_overall_ratio = 0.4;
    let pruned_size_kb = original_size_kb * (1.0 - recommended_overall_ratio);
    let compression_factor = original_size_kb / pruned_size_kb;

    println!("Model Compression Summary:");
    println!(
        "  Original size:     {:.1} KB ({} params)",
        original_size_kb, total_params
    );
    println!(
        "  Recommended prune: {:.0}% structured pruning",
        recommended_overall_ratio * 100.0
    );
    println!("  Pruned size:       {:.1} KB", pruned_size_kb);
    println!("  Compression:       {:.2}x", compression_factor);
    println!(
        "  Savings:           {:.1} KB ({:.0}%)",
        original_size_kb - pruned_size_kb,
        recommended_overall_ratio * 100.0
    );
    println!();
    println!("Layer-specific Recommendations:");
    for s in &sensitivities {
        println!(
            "  {:<10}: prune up to {:.0}% (impact: {:.3}%/1%)",
            s.layer_name,
            s.recommended_ratio * 100.0,
            s.accuracy_impact_per_pct
        );
    }
    println!();
    println!(
        "Most prunable layer: {} (lowest sensitivity: {:.3}%/1%)",
        best_sensitivity.layer_name, best_sensitivity.accuracy_impact_per_pct
    );

    ctx.record_float_metric("compression_factor", compression_factor);
    ctx.record_float_metric("pruned_size_kb", pruned_size_kb);

    // Save results
    let results_path = ctx.path("structured_pruning.json");
    save_results(&results_path, &sensitivities)?;
    println!();
    println!("Results saved to: {:?}", results_path);

    Ok(())
}

// ── Structs ──

/// A dense fully-connected layer with named weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DenseLayer {
    name: String,
    weights: Vec<Vec<f64>>,
    input_dim: usize,
    output_dim: usize,
}

/// Tracks which neurons survive pruning per layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PruningMask {
    layer_name: String,
    kept_indices: Vec<usize>,
    original_neurons: usize,
    kept_neurons: usize,
    estimated_accuracy: f64,
}

/// Configuration for a pruning operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PruningConfig {
    target_ratio: f64,
    strategy: PruningStrategy,
}

/// Pruning strategy selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum PruningStrategy {
    Magnitude,
    Random,
    Sensitivity,
}

/// Result of a pruning round.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PruningResult {
    original_params: usize,
    pruned_params: usize,
    accuracy_before: f64,
    accuracy_after: f64,
    speedup: f64,
}

/// Sensitivity of a layer to pruning.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LayerSensitivity {
    layer_name: String,
    accuracy_impact_per_pct: f64,
    recommended_ratio: f64,
}

/// Weight distribution statistics for a layer.
#[derive(Debug, Clone)]
struct WeightStats {
    mean_magnitude: f64,
    std_magnitude: f64,
}

// ── Functions ──

/// Deterministic hash for seeding.
fn deterministic_hash(input: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    hasher.finish()
}

/// Build a dense layer with deterministic pseudo-random weights.
fn build_dense_layer(name: &str, input_dim: usize, output_dim: usize) -> DenseLayer {
    let base_seed = deterministic_hash(name);

    let weights: Vec<Vec<f64>> = (0..output_dim)
        .map(|neuron| {
            (0..input_dim)
                .map(|w| {
                    let seed = base_seed
                        .wrapping_add(neuron as u64)
                        .wrapping_mul(31)
                        .wrapping_add(w as u64);
                    // Map to [-1, 1] with a scaled distribution
                    let raw = ((seed % 10_000) as f64 / 10_000.0) * 2.0 - 1.0;
                    // Scale by fan-in for Xavier-like initialization
                    raw / (input_dim as f64).sqrt()
                })
                .collect()
        })
        .collect();

    DenseLayer {
        name: name.to_string(),
        weights,
        input_dim,
        output_dim,
    }
}

/// Compute weight statistics for a layer.
fn compute_weight_stats(layer: &DenseLayer) -> WeightStats {
    let all_magnitudes: Vec<f64> = layer
        .weights
        .iter()
        .flat_map(|row| row.iter().map(|w| w.abs()))
        .collect();

    let n = all_magnitudes.len() as f64;
    if n == 0.0 {
        return WeightStats {
            mean_magnitude: 0.0,
            std_magnitude: 0.0,
        };
    }

    let mean = all_magnitudes.iter().sum::<f64>() / n;
    let variance = all_magnitudes
        .iter()
        .map(|m| (m - mean).powi(2))
        .sum::<f64>()
        / n;

    WeightStats {
        mean_magnitude: mean,
        std_magnitude: variance.sqrt(),
    }
}

/// Compute the L2-norm (magnitude) of a neuron's weight vector.
fn neuron_magnitude(weights: &[f64]) -> f64 {
    weights.iter().map(|w| w * w).sum::<f64>().sqrt()
}

/// Prune a layer by removing lowest-magnitude neurons.
fn prune_layer(layer: &DenseLayer, config: &PruningConfig) -> Result<PruningMask> {
    let num_to_prune = ((layer.output_dim as f64) * config.target_ratio).round() as usize;
    let num_to_keep = layer.output_dim.saturating_sub(num_to_prune);

    // Compute magnitude of each neuron
    let mut neuron_scores: Vec<(usize, f64)> = layer
        .weights
        .iter()
        .enumerate()
        .map(|(i, w)| {
            let score = match config.strategy {
                PruningStrategy::Magnitude => neuron_magnitude(w),
                PruningStrategy::Random => {
                    // Deterministic pseudo-random ordering
                    let h = deterministic_hash(&format!("{}_{}", layer.name, i));
                    (h % 10_000) as f64 / 10_000.0
                }
                PruningStrategy::Sensitivity => {
                    // Weight magnitude scaled by position (later neurons less important)
                    let pos_factor = 1.0 - (i as f64 / layer.output_dim as f64) * 0.3;
                    neuron_magnitude(w) * pos_factor
                }
            };
            (i, score)
        })
        .collect();

    // Sort descending by score (keep highest)
    neuron_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let kept_indices: Vec<usize> = neuron_scores
        .iter()
        .take(num_to_keep)
        .map(|(idx, _)| *idx)
        .collect();

    // Estimate accuracy: more pruning causes more accuracy loss
    let keep_ratio = num_to_keep as f64 / layer.output_dim.max(1) as f64;
    let estimated_accuracy = 0.95 * keep_ratio.sqrt();

    Ok(PruningMask {
        layer_name: layer.name.clone(),
        kept_indices,
        original_neurons: layer.output_dim,
        kept_neurons: num_to_keep,
        estimated_accuracy,
    })
}

/// Analyze how sensitive a layer is to pruning.
fn analyze_layer_sensitivity(layer: &DenseLayer) -> Result<LayerSensitivity> {
    // Compute sensitivity based on weight distribution and layer position
    let stats = compute_weight_stats(layer);

    // Layers with higher mean magnitude are more sensitive (contain more information)
    // Output layers are most sensitive, embed layers moderately so
    let position_factor = match layer.name.as_str() {
        "output" => 2.5,
        "embed" => 1.2,
        _ => 0.8,
    };

    let accuracy_impact_per_pct = stats.mean_magnitude * position_factor * 10.0;

    // Recommend pruning ratio inversely proportional to sensitivity
    let recommended_ratio = if accuracy_impact_per_pct > 0.3 {
        0.1
    } else if accuracy_impact_per_pct > 0.15 {
        0.3
    } else {
        0.5
    };

    Ok(LayerSensitivity {
        layer_name: layer.name.clone(),
        accuracy_impact_per_pct,
        recommended_ratio,
    })
}

/// Perform iterative pruning with simulated fine-tuning between rounds.
fn iterative_prune_with_finetuning(
    layer: &DenseLayer,
    target_ratio: f64,
    num_rounds: usize,
) -> Result<Vec<PruningResult>> {
    let mut results = Vec::new();
    let original_params = layer.input_dim * layer.output_dim;
    let mut current_accuracy = 0.95; // Starting accuracy

    for round in 1..=num_rounds {
        let progress = round as f64 / num_rounds as f64;
        // Cubic schedule: prune gradually, more aggressively toward the end
        let cumulative_ratio = target_ratio * progress.powi(2);

        let remaining_neurons =
            ((layer.output_dim as f64) * (1.0 - cumulative_ratio)).round() as usize;
        let remaining_neurons = remaining_neurons.max(1);
        let pruned_params = remaining_neurons * layer.input_dim;

        // Accuracy drops from pruning
        let accuracy_drop = cumulative_ratio * 0.08;

        // Fine-tuning recovers some accuracy (diminishing returns per round)
        let finetune_recovery = 0.03 * (1.0 - progress * 0.5);

        let accuracy_before = current_accuracy;
        current_accuracy = (accuracy_before - accuracy_drop + finetune_recovery).max(0.5);

        results.push(PruningResult {
            original_params,
            pruned_params,
            accuracy_before,
            accuracy_after: current_accuracy,
            speedup: original_params as f64 / pruned_params.max(1) as f64,
        });
    }

    Ok(results)
}

/// Compare structured vs unstructured pruning for a given strategy and ratio.
fn compare_pruning_method(
    strategy: PruningStrategy,
    ratio: f64,
    total_params: usize,
) -> Result<PruningResult> {
    let remaining = ((total_params as f64) * (1.0 - ratio)).round() as usize;
    let remaining = remaining.max(1);

    // Structured pruning: removes whole neurons, so hardware can skip computation
    // Unstructured pruning: removes individual weights, needs sparse format overhead
    let (accuracy_penalty, speedup_factor) = match strategy {
        PruningStrategy::Magnitude => {
            // Structured: slightly worse accuracy but much better speedup
            let penalty = ratio * 0.06;
            let speedup = 1.0 / (1.0 - ratio * 0.85);
            (penalty, speedup)
        }
        PruningStrategy::Random => {
            // Unstructured: better accuracy but minimal actual speedup
            let penalty = ratio * 0.04;
            let speedup = 1.0 / (1.0 - ratio * 0.3);
            (penalty, speedup)
        }
        PruningStrategy::Sensitivity => {
            let penalty = ratio * 0.05;
            let speedup = 1.0 / (1.0 - ratio * 0.7);
            (penalty, speedup)
        }
    };

    let accuracy_after = (0.95 - accuracy_penalty).max(0.5);

    Ok(PruningResult {
        original_params: total_params,
        pruned_params: remaining,
        accuracy_before: 0.95,
        accuracy_after,
        speedup: speedup_factor,
    })
}

fn save_results(path: &std::path::Path, sensitivities: &[LayerSensitivity]) -> Result<()> {
    let json = serde_json::to_string_pretty(sensitivities)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_layer() -> DenseLayer {
        build_dense_layer("hidden1", 64, 32)
    }

    #[test]
    fn test_build_dense_layer_dimensions() {
        let layer = build_dense_layer("test", 16, 8);
        assert_eq!(layer.input_dim, 16);
        assert_eq!(layer.output_dim, 8);
        assert_eq!(layer.weights.len(), 8);
        assert_eq!(layer.weights[0].len(), 16);
    }

    #[test]
    fn test_build_dense_layer_deterministic() {
        let l1 = build_dense_layer("test", 16, 8);
        let l2 = build_dense_layer("test", 16, 8);
        assert_eq!(l1.weights, l2.weights);
    }

    #[test]
    fn test_build_dense_layer_different_names() {
        let l1 = build_dense_layer("layer_a", 16, 8);
        let l2 = build_dense_layer("layer_b", 16, 8);
        assert_ne!(l1.weights, l2.weights);
    }

    #[test]
    fn test_weight_stats_nonzero() {
        let layer = sample_layer();
        let stats = compute_weight_stats(&layer);
        assert!(stats.mean_magnitude > 0.0);
        assert!(stats.std_magnitude >= 0.0);
    }

    #[test]
    fn test_neuron_magnitude() {
        let weights = vec![3.0, 4.0];
        let mag = neuron_magnitude(&weights);
        assert!((mag - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_neuron_magnitude_zero() {
        let weights = vec![0.0, 0.0, 0.0];
        let mag = neuron_magnitude(&weights);
        assert!((mag - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_prune_layer_reduces_neurons() {
        let layer = sample_layer();
        let config = PruningConfig {
            target_ratio: 0.5,
            strategy: PruningStrategy::Magnitude,
        };
        let mask = prune_layer(&layer, &config).expect("pruning should succeed");
        assert!(mask.kept_neurons < mask.original_neurons);
        assert_eq!(mask.kept_neurons, 16); // 32 * 0.5
    }

    #[test]
    fn test_prune_layer_zero_ratio() {
        let layer = sample_layer();
        let config = PruningConfig {
            target_ratio: 0.0,
            strategy: PruningStrategy::Magnitude,
        };
        let mask = prune_layer(&layer, &config).expect("pruning should succeed");
        assert_eq!(mask.kept_neurons, mask.original_neurons);
    }

    #[test]
    fn test_prune_layer_random_strategy() {
        let layer = sample_layer();
        let config = PruningConfig {
            target_ratio: 0.3,
            strategy: PruningStrategy::Random,
        };
        let mask = prune_layer(&layer, &config).expect("pruning should succeed");
        assert!(mask.kept_neurons > 0);
        assert!(mask.kept_neurons < mask.original_neurons);
    }

    #[test]
    fn test_prune_layer_sensitivity_strategy() {
        let layer = sample_layer();
        let config = PruningConfig {
            target_ratio: 0.3,
            strategy: PruningStrategy::Sensitivity,
        };
        let mask = prune_layer(&layer, &config).expect("pruning should succeed");
        assert!(mask.kept_neurons > 0);
    }

    #[test]
    fn test_layer_sensitivity_output_most_sensitive() {
        let hidden = build_dense_layer("hidden1", 64, 32);
        let output = build_dense_layer("output", 32, 10);

        let hidden_sens = analyze_layer_sensitivity(&hidden).expect("analysis should succeed");
        let output_sens = analyze_layer_sensitivity(&output).expect("analysis should succeed");

        assert!(output_sens.accuracy_impact_per_pct > hidden_sens.accuracy_impact_per_pct);
    }

    #[test]
    fn test_layer_sensitivity_recommended_ratio_bounded() {
        let layer = sample_layer();
        let sens = analyze_layer_sensitivity(&layer).expect("analysis should succeed");
        assert!(sens.recommended_ratio > 0.0);
        assert!(sens.recommended_ratio <= 1.0);
    }

    #[test]
    fn test_iterative_pruning_produces_rounds() {
        let layer = sample_layer();
        let results =
            iterative_prune_with_finetuning(&layer, 0.5, 5).expect("iterative should succeed");
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_iterative_pruning_accuracy_bounded() {
        let layer = sample_layer();
        let results =
            iterative_prune_with_finetuning(&layer, 0.5, 5).expect("iterative should succeed");
        for r in &results {
            assert!(r.accuracy_after >= 0.5);
            assert!(r.accuracy_after <= 1.0);
        }
    }

    #[test]
    fn test_compare_structured_vs_unstructured() {
        let structured = compare_pruning_method(PruningStrategy::Magnitude, 0.5, 10000)
            .expect("comparison should succeed");
        let unstructured = compare_pruning_method(PruningStrategy::Random, 0.5, 10000)
            .expect("comparison should succeed");

        // Structured pruning should have better speedup
        assert!(structured.speedup > unstructured.speedup);
    }

    #[test]
    fn test_compare_pruning_accuracy_bounded() {
        let result = compare_pruning_method(PruningStrategy::Magnitude, 0.7, 10000)
            .expect("comparison should succeed");
        assert!(result.accuracy_after >= 0.5);
        assert!(result.accuracy_after <= 1.0);
    }

    #[test]
    fn test_deterministic_hash() {
        let h1 = deterministic_hash("test_input");
        let h2 = deterministic_hash("test_input");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_save_results() {
        let ctx =
            RecipeContext::new("test_structured_pruning_save").expect("context should create");
        let path = ctx.path("results.json");

        let sensitivities = vec![LayerSensitivity {
            layer_name: "test".to_string(),
            accuracy_impact_per_pct: 0.15,
            recommended_ratio: 0.3,
        }];

        save_results(&path, &sensitivities).expect("save should succeed");
        assert!(path.exists());
    }

    #[test]
    fn test_empty_weight_stats() {
        let layer = DenseLayer {
            name: "empty".to_string(),
            weights: vec![],
            input_dim: 0,
            output_dim: 0,
        };
        let stats = compute_weight_stats(&layer);
        assert!((stats.mean_magnitude - 0.0).abs() < 1e-10);
        assert!((stats.std_magnitude - 0.0).abs() < 1e-10);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_pruning_keeps_correct_count(ratio in 0.0f64..0.99, neurons in 4usize..64) {
            let layer = build_dense_layer("prop_test", 16, neurons);
            let config = PruningConfig {
                target_ratio: ratio,
                strategy: PruningStrategy::Magnitude,
            };
            let mask = prune_layer(&layer, &config).unwrap();

            let expected_keep = neurons - ((neurons as f64 * ratio).round() as usize);
            prop_assert_eq!(mask.kept_neurons, expected_keep);
            prop_assert_eq!(mask.kept_indices.len(), expected_keep);
        }

        #[test]
        fn prop_accuracy_bounded_after_pruning(ratio in 0.0f64..0.99) {
            let layer = build_dense_layer("prop_acc", 32, 16);
            let config = PruningConfig {
                target_ratio: ratio,
                strategy: PruningStrategy::Magnitude,
            };
            let mask = prune_layer(&layer, &config).unwrap();
            prop_assert!(mask.estimated_accuracy >= 0.0);
            prop_assert!(mask.estimated_accuracy <= 1.0);
        }

        #[test]
        fn prop_iterative_monotonic_params(rounds in 2usize..10) {
            let layer = build_dense_layer("prop_iter", 32, 16);
            let results = iterative_prune_with_finetuning(&layer, 0.5, rounds).unwrap();

            // Params should be non-increasing across rounds
            for i in 1..results.len() {
                prop_assert!(results[i].pruned_params <= results[i - 1].pruned_params);
            }
        }

        #[test]
        fn prop_sensitivity_positive(name in "[a-z]{3,8}") {
            let layer = build_dense_layer(&name, 32, 16);
            let sens = analyze_layer_sensitivity(&layer).unwrap();
            prop_assert!(sens.accuracy_impact_per_pct >= 0.0);
            prop_assert!(sens.recommended_ratio > 0.0);
            prop_assert!(sens.recommended_ratio <= 1.0);
        }

        #[test]
        fn prop_compare_pruning_valid(ratio in 0.01f64..0.95, params in 100usize..100_000) {
            let result = compare_pruning_method(PruningStrategy::Magnitude, ratio, params).unwrap();
            prop_assert!(result.accuracy_after >= 0.5);
            prop_assert!(result.accuracy_after <= 1.0);
            prop_assert!(result.speedup >= 1.0);
            prop_assert!(result.pruned_params <= result.original_params);
        }
    }
}
