//! # Recipe: Depth Pruning (Minitron-Style Layer Removal)
//!
//! **Category**: optimize
//! **CLI Equivalent**: `apr prune --method depth`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Demonstrates depth pruning: removing entire transformer layers based on
//! importance scores. Inspired by NVIDIA Minitron, this technique produces
//! smaller models by eliminating redundant layers while preserving the most
//! critical computation paths.
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

/// Transformer model configuration.
#[derive(Clone, Debug)]
struct TransformerConfig {
    num_layers: usize,
    hidden_dim: usize,
    num_heads: usize,
    head_dim: usize,
}

impl TransformerConfig {
    fn params_per_layer(&self) -> usize {
        // Attention: Q, K, V, O projections
        let attn_params = 4 * self.hidden_dim * self.hidden_dim;
        // FFN: up projection + down projection (4x hidden)
        let ffn_params = 2 * self.hidden_dim * (4 * self.hidden_dim);
        // Layer norms (negligible but counted)
        let norm_params = 2 * self.hidden_dim;
        attn_params + ffn_params + norm_params
    }

    fn total_params(&self) -> usize {
        // Embedding + layers + final norm
        let embedding = self.hidden_dim * 32000; // Approximate vocab
        let layers = self.num_layers * self.params_per_layer();
        let final_norm = self.hidden_dim;
        embedding + layers + final_norm
    }
}

/// Importance score for a single layer.
#[derive(Clone, Debug)]
struct LayerImportance {
    layer_idx: usize,
    importance_score: f64,
    #[allow(dead_code)]
    param_count: usize,
}

/// Compute layer importance based on weight magnitude and gradient proxy.
///
/// Uses a deterministic proxy: layers near the beginning and end tend to be
/// more important (U-shaped importance curve observed in practice).
fn compute_layer_importance(config: &TransformerConfig, seed: u64) -> Vec<LayerImportance> {
    (0..config.num_layers)
        .map(|idx| {
            // Deterministic weight magnitude proxy
            let mut h = DefaultHasher::new();
            (seed, idx as u64, "magnitude").hash(&mut h);
            let mag_bits = h.finish();
            let base_magnitude = (mag_bits & 0xFFFF) as f64 / f64::from(0xFFFFu16);

            // Gradient proxy: U-shaped curve (first and last layers more important)
            let depth_ratio = idx as f64 / (config.num_layers - 1).max(1) as f64;
            let u_shape = (depth_ratio - 0.5).powi(2) * 4.0; // 0 at middle, 1 at edges
            let gradient_proxy = 0.3 + u_shape * 0.7;

            // Combined importance: weight magnitude * gradient signal
            let importance_score = base_magnitude * 0.4 + gradient_proxy * 0.6;

            LayerImportance {
                layer_idx: idx,
                importance_score,
                param_count: config.params_per_layer(),
            }
        })
        .collect()
}

/// Remove layers by index, returning a new config with fewer layers.
fn remove_layers(config: &TransformerConfig, indices_to_remove: &[usize]) -> TransformerConfig {
    let remaining = config.num_layers
        - indices_to_remove
            .len()
            .min(config.num_layers.saturating_sub(1));
    TransformerConfig {
        num_layers: remaining.max(1),
        hidden_dim: config.hidden_dim,
        num_heads: config.num_heads,
        head_dim: config.head_dim,
    }
}

/// Select layers to remove: pick the least important ones.
fn select_layers_to_remove(importances: &[LayerImportance], num_to_remove: usize) -> Vec<usize> {
    let mut sorted: Vec<&LayerImportance> = importances.iter().collect();
    sorted.sort_by(|a, b| {
        a.importance_score
            .partial_cmp(&b.importance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    sorted
        .iter()
        .take(num_to_remove)
        .map(|li| li.layer_idx)
        .collect()
}

/// Estimate accuracy retention based on layers removed.
fn estimate_accuracy_retention(original_layers: usize, removed: usize) -> f64 {
    if original_layers == 0 {
        return 0.0;
    }
    let ratio = removed as f64 / original_layers as f64;
    // Non-linear degradation: removing first few layers has less impact
    // than removing many layers
    let retention = 1.0 - ratio.powf(0.7) * 0.5;
    retention.clamp(0.0, 1.0)
}

fn format_params(params: usize) -> String {
    if params >= 1_000_000_000 {
        format!("{:.1}B", params as f64 / 1e9)
    } else if params >= 1_000_000 {
        format!("{:.1}M", params as f64 / 1e6)
    } else {
        format!("{:.1}K", params as f64 / 1e3)
    }
}

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("prune_depth")?;

    println!("=== Depth Pruning (Minitron-Style Layer Removal) ===\n");

    // --- Section 1: Model Architecture ---
    println!("--- Model Architecture ---");
    let config = TransformerConfig {
        num_layers: 32,
        hidden_dim: 4096,
        num_heads: 32,
        head_dim: 128,
    };

    println!("  Layers: {}", config.num_layers);
    println!("  Hidden dim: {}", config.hidden_dim);
    println!("  Attention heads: {}", config.num_heads);
    println!("  Head dim: {}", config.head_dim);
    println!(
        "  Params per layer: {}",
        format_params(config.params_per_layer())
    );
    println!("  Total params: {}", format_params(config.total_params()));
    println!();

    ctx.record_metric("original_layers", config.num_layers as i64);
    ctx.record_metric(
        "original_params_millions",
        (config.total_params() / 1_000_000) as i64,
    );

    // --- Section 2: Layer Importance Ranking ---
    println!("--- Layer Importance Ranking ---");
    let importances = compute_layer_importance(&config, 42);

    let mut ranked = importances.clone();
    ranked.sort_by(|a, b| {
        b.importance_score
            .partial_cmp(&a.importance_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("  Most important layers:");
    for li in ranked.iter().take(5) {
        println!(
            "    Layer {:>2}: importance = {:.4}",
            li.layer_idx, li.importance_score
        );
    }
    println!("  Least important layers:");
    for li in ranked.iter().rev().take(5) {
        println!(
            "    Layer {:>2}: importance = {:.4}",
            li.layer_idx, li.importance_score
        );
    }

    // Show U-shape pattern
    println!("\n  Importance curve (U-shaped, first/last layers most important):");
    for li in &importances {
        let bar_len = (li.importance_score * 40.0).round() as usize;
        let bar: String = "#".repeat(bar_len);
        println!(
            "    L{:>2} |{bar:<40}| {:.3}",
            li.layer_idx, li.importance_score
        );
    }
    println!();

    // --- Section 3: Depth Pruning at Various Levels ---
    println!("--- Depth Pruning Results ---");
    let removal_counts = [1, 4, 8, 12];

    println!(
        "  {:>8} | {:>8} | {:>12} | {:>12} | {:>10}",
        "Removed", "Remaining", "Params", "Reduction", "Accuracy"
    );
    println!("  {}", "-".repeat(60));

    for &num_remove in &removal_counts {
        let to_remove = select_layers_to_remove(&importances, num_remove);
        let pruned_config = remove_layers(&config, &to_remove);
        let accuracy = estimate_accuracy_retention(config.num_layers, num_remove);

        let param_reduction =
            1.0 - pruned_config.total_params() as f64 / config.total_params() as f64;

        println!(
            "  {:>8} | {:>8} | {:>12} | {:>10.1}% | {:>9.1}%",
            num_remove,
            pruned_config.num_layers,
            format_params(pruned_config.total_params()),
            param_reduction * 100.0,
            accuracy * 100.0,
        );

        let metric = format!("layers_after_remove_{num_remove}");
        ctx.record_metric(&metric, pruned_config.num_layers as i64);
    }
    println!();

    // --- Section 4: Removed Layer Details ---
    println!("--- Removed Layers (8-layer pruning) ---");
    let to_remove_8 = select_layers_to_remove(&importances, 8);
    let mut remove_sorted = to_remove_8.clone();
    remove_sorted.sort_unstable();
    println!("  Layers removed: {:?}", remove_sorted);
    println!(
        "  Layers kept: {:?}",
        (0..config.num_layers)
            .filter(|i| !remove_sorted.contains(i))
            .collect::<Vec<_>>()
    );
    println!();

    // --- Section 5: Accuracy vs Depth Tradeoff ---
    println!("--- Accuracy vs Depth Tradeoff ---");
    println!("  Layers | Accuracy | Visual");
    println!("  {}", "-".repeat(50));
    for num_remove in (0..=28).step_by(4) {
        let remaining = config.num_layers - num_remove;
        let accuracy = estimate_accuracy_retention(config.num_layers, num_remove);
        let bar_len = (accuracy * 30.0).round() as usize;
        let bar: String = "#".repeat(bar_len);
        println!("  {:>6} | {:>6.1}%  | {bar}", remaining, accuracy * 100.0);
    }
    println!();

    // --- Section 6: Save Depth-Pruned Config ---
    println!("--- Save Depth-Pruned Model (APR v2) ---");
    let pruned_config = remove_layers(&config, &select_layers_to_remove(&importances, 8));

    // Serialize config as metadata in the bundle
    let config_json = format!(
        r#"{{"num_layers":{},"hidden_dim":{},"num_heads":{},"head_dim":{}}}"#,
        pruned_config.num_layers,
        pruned_config.hidden_dim,
        pruned_config.num_heads,
        pruned_config.head_dim
    );
    let config_bytes = config_json.into_bytes();

    let bundle = ModelBundleV2::new()
        .with_name("depth_pruned_24L")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("config", vec![1, config_bytes.len()], config_bytes)
        .build();

    assert_eq!(&bundle[0..4], b"APR2");
    println!(
        "  Pruned config: {} layers, {} hidden",
        pruned_config.num_layers, pruned_config.hidden_dim
    );
    println!("  Bundle size: {} bytes", bundle.len());
    println!(
        "  Original params: {}",
        format_params(config.total_params())
    );
    println!(
        "  Pruned params: {}",
        format_params(pruned_config.total_params())
    );

    ctx.record_metric("pruned_layers", pruned_config.num_layers as i64);
    ctx.record_metric("bundle_size_bytes", bundle.len() as i64);
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_importance_ordering_u_shape() {
        let config = TransformerConfig {
            num_layers: 16,
            hidden_dim: 256,
            num_heads: 4,
            head_dim: 64,
        };
        let importances = compute_layer_importance(&config, 42);
        // First and last layers should generally have higher importance than middle
        let first = importances[0].importance_score;
        let last = importances[config.num_layers - 1].importance_score;
        let middle = importances[config.num_layers / 2].importance_score;
        assert!(
            first > middle || last > middle,
            "U-shape: edges ({first:.4}, {last:.4}) should be > middle ({middle:.4})"
        );
    }

    #[test]
    fn test_removal_reduces_params() {
        let config = TransformerConfig {
            num_layers: 32,
            hidden_dim: 512,
            num_heads: 8,
            head_dim: 64,
        };
        let importances = compute_layer_importance(&config, 1);
        let to_remove = select_layers_to_remove(&importances, 8);
        let pruned = remove_layers(&config, &to_remove);
        assert!(pruned.total_params() < config.total_params());
    }

    #[test]
    fn test_config_valid_after_pruning() {
        let config = TransformerConfig {
            num_layers: 24,
            hidden_dim: 1024,
            num_heads: 16,
            head_dim: 64,
        };
        let importances = compute_layer_importance(&config, 2);
        let to_remove = select_layers_to_remove(&importances, 12);
        let pruned = remove_layers(&config, &to_remove);

        assert!(pruned.num_layers >= 1);
        assert_eq!(pruned.hidden_dim, config.hidden_dim);
        assert_eq!(pruned.num_heads, config.num_heads);
        assert_eq!(pruned.head_dim, config.head_dim);
    }

    #[test]
    fn test_remove_zero_layers() {
        let config = TransformerConfig {
            num_layers: 16,
            hidden_dim: 256,
            num_heads: 4,
            head_dim: 64,
        };
        let pruned = remove_layers(&config, &[]);
        assert_eq!(pruned.num_layers, config.num_layers);
    }

    #[test]
    fn test_importance_count_matches_layers() {
        let config = TransformerConfig {
            num_layers: 32,
            hidden_dim: 512,
            num_heads: 8,
            head_dim: 64,
        };
        let importances = compute_layer_importance(&config, 3);
        assert_eq!(importances.len(), config.num_layers);
    }

    #[test]
    fn test_select_removes_least_important() {
        let config = TransformerConfig {
            num_layers: 10,
            hidden_dim: 128,
            num_heads: 2,
            head_dim: 64,
        };
        let importances = compute_layer_importance(&config, 4);
        let to_remove = select_layers_to_remove(&importances, 3);

        // Get the importance scores of removed layers
        let removed_scores: Vec<f64> = to_remove
            .iter()
            .map(|&idx| importances[idx].importance_score)
            .collect();
        let kept_indices: Vec<usize> = (0..10).filter(|i| !to_remove.contains(i)).collect();
        let min_kept = kept_indices
            .iter()
            .map(|&i| importances[i].importance_score)
            .fold(f64::INFINITY, f64::min);
        let max_removed = removed_scores
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        assert!(
            max_removed <= min_kept + f64::EPSILON,
            "Removed max ({max_removed}) should be <= kept min ({min_kept})"
        );
    }

    #[test]
    fn test_accuracy_retention_monotonic() {
        let retentions: Vec<f64> = (0..=16)
            .map(|r| estimate_accuracy_retention(32, r))
            .collect();
        for window in retentions.windows(2) {
            assert!(
                window[0] >= window[1],
                "Accuracy should decrease: {} vs {}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn test_accuracy_retention_bounds() {
        let full = estimate_accuracy_retention(32, 0);
        assert!((full - 1.0).abs() < f64::EPSILON);
        let heavy = estimate_accuracy_retention(32, 32);
        assert!(heavy >= 0.0 && heavy <= 1.0);
    }

    #[test]
    fn test_deterministic() {
        let config = TransformerConfig {
            num_layers: 16,
            hidden_dim: 256,
            num_heads: 4,
            head_dim: 64,
        };
        let i1 = compute_layer_importance(&config, 99);
        let i2 = compute_layer_importance(&config, 99);
        for (a, b) in i1.iter().zip(i2.iter()) {
            assert_eq!(a.layer_idx, b.layer_idx);
            assert!((a.importance_score - b.importance_score).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_format_params() {
        assert_eq!(format_params(1_500_000_000), "1.5B");
        assert_eq!(format_params(350_000_000), "350.0M");
        assert_eq!(format_params(500_000), "500.0K");
    }
}
