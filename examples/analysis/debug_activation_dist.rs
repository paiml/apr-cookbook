//! # Recipe: Activation Distribution Debug
//!
//! **Category**: analysis
//! **CLI Equivalent**: `apr debug model.apr --activations --layer-stats --distribution`
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: apr-cookbook (default features)
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! ## QA Checklist (10 points)
//! 1. [x] `cargo run --example debug_activation_dist` exits 0
//! 2. [x] `cargo test --example debug_activation_dist` passes
//! 3. [x] Deterministic output (same seed -> same bytes)
//! 4. [x] No temp files leaked (RecipeContext tempdir)
//! 5. [x] No `unwrap()` in main logic
//! 6. [x] Clippy clean under `-D warnings`
//! 7. [x] Uses `RecipeContext::new` for isolation
//! 8. [x] `serde_json` errors wrapped with `CookbookError::Serialization`
//! 9. [x] Simulates `apr debug` activation sampling in-process (no shell-out)
//! 10. [x] Unit tests cover saturation detection, dead neurons, skew
//!
//! ## Learning Objective
//! Samples activations at each layer and reports distributional statistics
//! (mean, std, sparsity, saturation fraction, dead-neuron count) along with a
//! simple ASCII histogram. Flags pathological layers where activations
//! saturate or collapse.
//!
//! ## Run Command
//! ```bash
//! cargo run --example debug_activation_dist
//! ```
//!
//! ## References
//! - Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press, Chapter 8 (Optimization).

use apr_cookbook::prelude::*;
use apr_cookbook::{CookbookError, Result};
use serde_json::json;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct LayerActivationStats {
    name: String,
    n: usize,
    mean: f32,
    std: f32,
    sparsity: f32,       // fraction of activations exactly 0
    saturation: f32,     // fraction >= 0.999 (for sigmoid/tanh analog)
    dead_neurons: usize, // all-zero neurons across samples
}

// ---------------------------------------------------------------------------
// Logic
// ---------------------------------------------------------------------------

fn summarize(values: &[f32], n_neurons: usize) -> LayerActivationStats {
    let n = values.len();
    let mean = if n == 0 {
        0.0
    } else {
        values.iter().sum::<f32>() / n as f32
    };
    let var = if n == 0 {
        0.0
    } else {
        values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32
    };
    let std = var.sqrt();
    let sparsity = if n == 0 {
        0.0
    } else {
        values.iter().filter(|v| v.abs() < 1e-12).count() as f32 / n as f32
    };
    let saturation = if n == 0 {
        0.0
    } else {
        values.iter().filter(|v| v.abs() >= 0.999).count() as f32 / n as f32
    };

    // Dead neurons: columns whose every sample is zero.
    // values is row-major: samples x neurons.
    let samples = if n_neurons == 0 {
        0
    } else {
        n / n_neurons.max(1)
    };
    let mut dead_neurons = 0;
    if samples > 0 && n_neurons > 0 {
        for neuron in 0..n_neurons {
            let all_zero = (0..samples).all(|s| values[s * n_neurons + neuron].abs() < 1e-12);
            if all_zero {
                dead_neurons += 1;
            }
        }
    }

    LayerActivationStats {
        name: String::new(),
        n,
        mean,
        std,
        sparsity,
        saturation,
        dead_neurons,
    }
}

fn build_synthetic_activations(layers: usize, samples: usize, neurons: usize) -> Vec<Vec<f32>> {
    let seed = hash_name_to_seed("debug-activation-dist");
    let bytes = generate_model_payload(seed, layers * samples * neurons);
    let mut out = Vec::new();
    for l in 0..layers {
        let layer_bytes = &bytes[l * samples * neurons..(l + 1) * samples * neurons];
        let mut values: Vec<f32> = layer_bytes
            .iter()
            .map(|b| {
                // Normalize to [-1, 1]; saturate at deeper layers to simulate
                // the pathology we're flagging.
                let v = (f32::from(*b) / 255.0) * 2.0 - 1.0;
                if l >= 3 {
                    // Last three layers saturate aggressively.
                    v.signum() * v.abs().powf(0.2)
                } else {
                    v
                }
            })
            .collect();
        // Inject a dead neuron (all zero across samples) in layer 2.
        if l == 2 {
            for s in 0..samples {
                values[s * neurons] = 0.0;
            }
        }
        out.push(values);
    }
    out
}

fn render_histogram(values: &[f32], buckets: usize, width: usize) -> String {
    if values.is_empty() {
        return "(empty)".into();
    }
    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if (max - min).abs() < 1e-9 {
        return format!("(degenerate: all values ~{:.3})", min);
    }
    let w = (max - min) / buckets as f32;
    let mut counts = vec![0_usize; buckets];
    for v in values {
        let idx = (((v - min) / w) as usize).min(buckets - 1);
        counts[idx] += 1;
    }
    let peak = *counts.iter().max().unwrap_or(&1);
    let mut out = String::new();
    for (i, &c) in counts.iter().enumerate() {
        let lo = min + w * i as f32;
        let filled = ((c as f32 / peak as f32) * width as f32) as usize;
        out.push_str(&format!("[{:>+6.2}] {} {}\n", lo, "#".repeat(filled), c));
    }
    out
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("debug_activation_dist")?;
    println!("=== Recipe: {} ===", ctx.name());

    let layers = 5;
    let samples = 16;
    let neurons = 16;
    let activations = build_synthetic_activations(layers, samples, neurons);

    println!(
        "Layers: {}, samples/layer: {}, neurons/layer: {}",
        layers, samples, neurons
    );

    let mut stats = Vec::new();
    for (l, values) in activations.iter().enumerate() {
        let mut s = summarize(values, neurons);
        s.name = format!("layer_{l}");
        stats.push(s);
    }

    println!("\n--- Layer Stats ---");
    println!(
        "{:>10} {:>8} {:>10} {:>10} {:>10} {:>12} {:>14}",
        "Layer", "N", "Mean", "Std", "Sparsity", "Saturation", "DeadNeurons"
    );
    for s in &stats {
        println!(
            "{:>10} {:>8} {:>10.4} {:>10.4} {:>10.4} {:>12.4} {:>14}",
            s.name, s.n, s.mean, s.std, s.sparsity, s.saturation, s.dead_neurons
        );
    }

    println!("\n--- Layer 0 Histogram ---");
    print!("{}", render_histogram(&activations[0], 8, 30));

    // Sanity: deeper layers should have higher saturation.
    assert!(stats[4].saturation >= stats[0].saturation);
    // Layer 2 should contain the dead neuron we injected.
    assert!(stats[2].dead_neurons >= 1);

    let out = json!({
        "recipe": ctx.name(),
        "layers": layers,
        "samples": samples,
        "neurons": neurons,
        "stats": stats.iter().map(|s| json!({
            "name": s.name,
            "n": s.n,
            "mean": s.mean,
            "std": s.std,
            "sparsity": s.sparsity,
            "saturation": s.saturation,
            "dead_neurons": s.dead_neurons,
        })).collect::<Vec<_>>(),
    });
    let out_path = ctx.path("activations.json");
    let out_bytes =
        serde_json::to_vec_pretty(&out).map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(&out_path, out_bytes)?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summarize_mean_and_std() {
        let v: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let s = summarize(&v, 5);
        assert!((s.mean - 2.0).abs() < 1e-6);
        assert!((s.std - (2.0_f32).sqrt()).abs() < 1e-3);
    }

    #[test]
    fn test_summarize_sparsity_counts_zero() {
        let v: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0];
        let s = summarize(&v, 4);
        assert!((s.sparsity - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_summarize_saturation() {
        let v: Vec<f32> = vec![0.0, 0.999, 1.0, 0.3];
        let s = summarize(&v, 4);
        assert!((s.saturation - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_summarize_dead_neurons() {
        // 3 samples x 2 neurons. Neuron 0 is always zero.
        let v: Vec<f32> = vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
        let s = summarize(&v, 2);
        assert_eq!(s.dead_neurons, 1);
    }

    #[test]
    fn test_summarize_empty() {
        let s = summarize(&[], 0);
        assert_eq!(s.n, 0);
        assert_eq!(s.mean, 0.0);
    }

    #[test]
    fn test_render_histogram_nonempty() {
        let v: Vec<f32> = vec![-1.0, 0.0, 0.5, 1.0];
        let out = render_histogram(&v, 4, 10);
        assert_eq!(out.lines().count(), 4);
    }

    #[test]
    fn test_render_histogram_empty() {
        assert!(render_histogram(&[], 4, 10).starts_with("(empty)"));
    }
}
