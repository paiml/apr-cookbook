//! Weighted Model Merge
//!
//! CLI equivalent: `apr merge --strategy weighted`
//!
//! Weighted averaging generalizes uniform averaging by assigning a scalar
//! weight to each model. This lets you bias the merge toward models that
//! performed better on a particular benchmark, or toward a specific task
//! capability you want to preserve.
//!
//! ## Algorithm
//!
//! ```text
//! merged[k] = sum(w_i * model_i[k])  where sum(w_i) = 1.0
//! ```
//!
//! ## When to Use
//!
//! - You have benchmark scores and want to bias toward the best model
//! - Combining a general model with a task-specific fine-tune
//! - A/B testing merge ratios for optimal downstream performance
//!
//!
//! ## Format Variants
//! ```bash
//! apr merge model.apr          # APR native format
//! apr merge model.gguf         # GGUF (llama.cpp compatible)
//! apr merge model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Wortsman, M. et al. (2022). *Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy*. ICML. arXiv:2203.05482

use apr_cookbook::prelude::*;
use entrenar::autograd::Tensor;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

type Model = HashMap<String, Tensor>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn create_base_model(layers: &[(&str, usize)], seed: u64) -> Model {
    layers
        .iter()
        .map(|&(name, size)| {
            let data: Vec<f32> = (0..size)
                .map(|i| {
                    let mut h = DefaultHasher::new();
                    (seed, name, i).hash(&mut h);
                    (h.finish() as f32 / u64::MAX as f32 - 0.5) * 0.1
                })
                .collect();
            (name.to_string(), Tensor::from_vec(data, false))
        })
        .collect()
}

fn create_finetuned(base: &Model, seed: u64, scale: f32) -> Model {
    base.iter()
        .map(|(name, tensor)| {
            let delta: Vec<f32> = (0..tensor.len())
                .map(|i| {
                    let mut h = DefaultHasher::new();
                    (seed, name.as_str(), i).hash(&mut h);
                    (h.finish() as f32 / u64::MAX as f32 - 0.5) * scale
                })
                .collect();
            let merged: Vec<f32> = tensor
                .data()
                .iter()
                .zip(delta.iter())
                .map(|(b, d)| b + d)
                .collect();
            (name.clone(), Tensor::from_vec(merged, false))
        })
        .collect()
}

fn model_distance(m1: &Model, m2: &Model) -> f32 {
    m1.iter()
        .map(|(name, t1)| {
            m2.get(name).map_or(0.0, |t2| {
                t1.data()
                    .iter()
                    .zip(t2.data().iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
            })
        })
        .sum::<f32>()
        .sqrt()
}

fn param_count(model: &Model) -> usize {
    model.values().map(Tensor::len).sum()
}

/// Average merge for comparison with uniform weights.
fn average_merge(models: &[Model]) -> Model {
    assert!(!models.is_empty(), "need at least one model");
    let n = models.len() as f32;
    let reference = &models[0];
    reference
        .iter()
        .map(|(name, t0)| {
            let size = t0.len();
            let mut acc = vec![0.0_f32; size];
            for model in models {
                let tensor = model.get(name).unwrap();
                for (a, v) in acc.iter_mut().zip(tensor.data().iter()) {
                    *a += v;
                }
            }
            for a in &mut acc {
                *a /= n;
            }
            (name.clone(), Tensor::from_vec(acc, false))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Weighted merge implementation
// ---------------------------------------------------------------------------

/// Compute element-wise weighted average across N models.
///
/// Weights must sum to 1.0 (within tolerance). Every model must have
/// identical keys and tensor lengths.
fn weighted_merge(models: &[Model], weights: &[f32]) -> Model {
    assert!(!models.is_empty(), "need at least one model");
    assert_eq!(
        models.len(),
        weights.len(),
        "models and weights must have same length"
    );
    let w_sum: f32 = weights.iter().sum();
    assert!(
        (w_sum - 1.0).abs() < 1e-4,
        "weights must sum to 1.0, got {w_sum}"
    );

    let reference = &models[0];
    reference
        .iter()
        .map(|(name, t0)| {
            let size = t0.len();
            let mut acc = vec![0.0_f32; size];
            for (model, &w) in models.iter().zip(weights.iter()) {
                let tensor = model
                    .get(name)
                    .unwrap_or_else(|| panic!("missing key {name}"));
                assert_eq!(tensor.len(), size, "shape mismatch for {name}");
                for (a, v) in acc.iter_mut().zip(tensor.data().iter()) {
                    *a += w * v;
                }
            }
            (name.clone(), Tensor::from_vec(acc, false))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("merge_weighted")?;

    // --- Standard architecture layers ---
    let layers: Vec<(&str, usize)> = vec![
        ("attn.q_proj", 256),
        ("attn.k_proj", 256),
        ("attn.v_proj", 256),
        ("mlp.gate_proj", 512),
        ("mlp.up_proj", 512),
        ("mlp.down_proj", 512),
    ];

    // --- Section 1: Create fine-tuned variants ---
    let base = create_base_model(&layers, 42);
    let ft_math = create_finetuned(&base, 100, 0.05);
    let ft_code = create_finetuned(&base, 200, 0.05);
    let ft_writing = create_finetuned(&base, 300, 0.05);

    let models = vec![ft_math.clone(), ft_code.clone(), ft_writing.clone()];
    println!(
        "Models: 3 fine-tuned variants, {} params each",
        param_count(&base)
    );

    // --- Section 2: Uniform weights (same as average) ---
    let uniform_weights = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    let merged_uniform = weighted_merge(&models, &uniform_weights);
    let avg_merged = average_merge(&models);
    let uniform_vs_avg = model_distance(&merged_uniform, &avg_merged);
    println!("\n--- Uniform Weights [1/3, 1/3, 1/3] ---");
    println!("Distance uniform-weighted vs average: {uniform_vs_avg:.8}");
    println!("(Should be ~0, proving equivalence)");

    // --- Section 3: Task-biased weights ---
    let biased_weights = vec![0.6, 0.2, 0.2];
    let merged_biased = weighted_merge(&models, &biased_weights);
    let d_math = model_distance(&merged_biased, &ft_math);
    let d_code = model_distance(&merged_biased, &ft_code);
    let d_writing = model_distance(&merged_biased, &ft_writing);

    println!("\n--- Task-Biased Weights [0.6, 0.2, 0.2] (math-focused) ---");
    println!("Distance -> math:    {d_math:.6}");
    println!("Distance -> code:    {d_code:.6}");
    println!("Distance -> writing: {d_writing:.6}");
    println!("Merged is closest to math model (highest weight)");

    // --- Section 4: Benchmark-based weights ---
    // Simulate benchmark scores and derive weights
    let scores = [0.85_f32, 0.72, 0.91]; // math, code, writing
    let score_sum: f32 = scores.iter().sum();
    let bench_weights: Vec<f32> = scores.iter().map(|s| s / score_sum).collect();
    let merged_bench = weighted_merge(&models, &bench_weights);

    println!("\n--- Benchmark-Based Weights ---");
    println!(
        "Scores: math={:.2}, code={:.2}, writing={:.2}",
        scores[0], scores[1], scores[2]
    );
    println!(
        "Derived weights: [{:.3}, {:.3}, {:.3}]",
        bench_weights[0], bench_weights[1], bench_weights[2]
    );
    let db_math = model_distance(&merged_bench, &ft_math);
    let db_writing = model_distance(&merged_bench, &ft_writing);
    println!("Distance -> math:    {db_math:.6}");
    println!("Distance -> writing: {db_writing:.6}");

    // --- Section 5: Compare all strategies ---
    println!("\n--- Strategy Comparison ---");
    let d_uniform_biased = model_distance(&merged_uniform, &merged_biased);
    let d_uniform_bench = model_distance(&merged_uniform, &merged_bench);
    let d_biased_bench = model_distance(&merged_biased, &merged_bench);
    println!("uniform <-> biased:    {d_uniform_biased:.6}");
    println!("uniform <-> benchmark: {d_uniform_bench:.6}");
    println!("biased  <-> benchmark: {d_biased_bench:.6}");

    // --- Save to APR v2 ---
    let first_tensor = merged_biased.values().next().unwrap();
    let bytes: Vec<u8> = first_tensor
        .data()
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let bundle = ModelBundleV2::new()
        .with_name("weighted-merged")
        .with_compression(Compression::Lz4)
        .add_tensor("merged_weights", vec![first_tensor.len()], bytes)
        .build();

    assert_eq!(&bundle[0..4], b"APR2");
    println!("\nSaved merged model as APR v2 ({} bytes)", bundle.len());

    ctx.report()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_layers() -> Vec<(&'static str, usize)> {
        vec![
            ("attn.q_proj", 256),
            ("attn.k_proj", 256),
            ("attn.v_proj", 256),
            ("mlp.gate_proj", 512),
            ("mlp.up_proj", 512),
            ("mlp.down_proj", 512),
        ]
    }

    #[test]
    fn test_uniform_weights_equals_average() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let ft3 = create_finetuned(&base, 300, 0.05);
        let models = vec![ft1, ft2, ft3];
        let w_merged = weighted_merge(&models, &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        let a_merged = average_merge(&models);
        let dist = model_distance(&w_merged, &a_merged);
        assert!(dist < 1e-5, "uniform weights should equal average: {dist}");
    }

    #[test]
    #[should_panic(expected = "weights must sum to 1.0")]
    fn test_weights_must_sum_to_one() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        weighted_merge(&[ft1, ft2], &[0.3, 0.3]);
    }

    #[test]
    #[should_panic(expected = "models and weights must have same length")]
    fn test_mismatched_lengths_panics() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        weighted_merge(&[ft1], &[0.5, 0.5]);
    }

    #[test]
    fn test_weighted_closer_to_dominant_model() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let merged = weighted_merge(&[ft1.clone(), ft2.clone()], &[0.9, 0.1]);
        let d1 = model_distance(&merged, &ft1);
        let d2 = model_distance(&merged, &ft2);
        assert!(
            d1 < d2,
            "merged should be closer to model with weight 0.9: d1={d1}, d2={d2}"
        );
    }

    #[test]
    fn test_weight_one_zero_is_identity() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let merged = weighted_merge(&[ft1.clone(), ft2], &[1.0, 0.0]);
        let dist = model_distance(&merged, &ft1);
        assert!(
            dist < 1e-6,
            "weight [1,0] should return first model: {dist}"
        );
    }

    #[test]
    fn test_param_count_preserved() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let merged = weighted_merge(&[ft1, ft2], &[0.6, 0.4]);
        assert_eq!(param_count(&merged), param_count(&base));
    }

    #[test]
    fn test_keys_preserved() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let merged = weighted_merge(&[ft1, ft2], &[0.5, 0.5]);
        let mut mk: Vec<_> = merged.keys().cloned().collect();
        let mut bk: Vec<_> = base.keys().cloned().collect();
        mk.sort();
        bk.sort();
        assert_eq!(mk, bk);
    }

    #[test]
    fn test_weighted_deterministic() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let m1 = weighted_merge(&[ft1.clone(), ft2.clone()], &[0.7, 0.3]);
        let m2 = weighted_merge(&[ft1, ft2], &[0.7, 0.3]);
        let dist = model_distance(&m1, &m2);
        assert!(
            dist < 1e-6,
            "weighted merge should be deterministic: {dist}"
        );
    }

    #[test]
    fn test_symmetric_weights_equidistant() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let merged = weighted_merge(&[ft1.clone(), ft2.clone()], &[0.5, 0.5]);
        // With equal weights, verify element-wise midpoint
        for (name, tm) in &merged {
            let d1 = ft1.get(name).unwrap().data();
            let d2 = ft2.get(name).unwrap().data();
            for (idx, &val) in tm.data().iter().enumerate() {
                let expected = (d1[idx] + d2[idx]) / 2.0;
                assert!((val - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_apr_bundle() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let merged = weighted_merge(&[ft1, ft2], &[0.5, 0.5]);
        let first = merged.values().next().unwrap();
        let bytes: Vec<u8> = first.data().iter().flat_map(|f| f.to_le_bytes()).collect();
        let bundle = ModelBundleV2::new()
            .with_name("test-weighted")
            .with_compression(Compression::Lz4)
            .add_tensor("w", vec![first.len()], bytes)
            .build();
        assert_eq!(&bundle[0..4], b"APR2");
    }
}
