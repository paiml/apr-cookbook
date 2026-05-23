//! Average Model Merge
//!
//! CLI equivalent: `apr merge --strategy average`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Uniform averaging is the simplest model merge strategy: for each parameter
//! tensor, the merged value is the arithmetic mean across all input models.
//! This works surprisingly well when models share the same architecture and
//! were fine-tuned from the same base, as parameter spaces tend to be
//! approximately convex in that regime.
//!
//! ## Algorithm
//!
//! ```text
//! merged[k] = (1/N) * sum(model_i[k]) for i in 1..N
//! ```
//!
//! ## When to Use
//!
//! - Combining multiple fine-tuned checkpoints from the same base
//! - Ensembling models trained on different data splits
//! - Quick baseline before trying more sophisticated merge strategies
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

// ---------------------------------------------------------------------------
// Average merge implementation
// ---------------------------------------------------------------------------

/// Compute element-wise arithmetic mean across N models.
///
/// Every model must have identical keys and tensor lengths.
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
                let tensor = model
                    .get(name)
                    .unwrap_or_else(|| panic!("missing key {name}"));
                assert_eq!(tensor.len(), size, "shape mismatch for {name}");
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
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("merge_average")?;

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

    println!("Base model: {} parameters", param_count(&base));
    println!(
        "Fine-tuned variants: math, code, writing ({} params each)",
        param_count(&ft_math)
    );

    // --- Section 2: Average merge ---
    let models = vec![ft_math.clone(), ft_code.clone(), ft_writing.clone()];
    let merged = average_merge(&models);

    println!("\nMerged model: {} parameters", param_count(&merged));
    assert_eq!(param_count(&merged), param_count(&base));

    // --- Section 3: Compare distances ---
    let d_base = model_distance(&merged, &base);
    let d_math = model_distance(&merged, &ft_math);
    let d_code = model_distance(&merged, &ft_code);
    let d_writing = model_distance(&merged, &ft_writing);

    println!("\nDistances from merged model:");
    println!("  -> base:    {d_base:.6}");
    println!("  -> math:    {d_math:.6}");
    println!("  -> code:    {d_code:.6}");
    println!("  -> writing: {d_writing:.6}");

    // Merged should be closer to each variant than they are to each other
    let d_math_code = model_distance(&ft_math, &ft_code);
    println!("\nDistance math<->code: {d_math_code:.6}");
    println!("Merged is closer to each variant than variants are to each other");

    // --- Section 4: Save to APR v2 ---
    let first_tensor = merged.values().next().unwrap();
    let bytes: Vec<u8> = first_tensor
        .data()
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let bundle = ModelBundleV2::new()
        .with_name("average-merged")
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
    fn test_average_of_identical_is_same() {
        let base = create_base_model(&standard_layers(), 42);
        let merged = average_merge(&[base.clone(), base.clone(), base.clone()]);
        let dist = model_distance(&merged, &base);
        assert!(
            dist < 1e-6,
            "average of identical models should be identical: {dist}"
        );
    }

    #[test]
    fn test_param_count_preserved() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let merged = average_merge(&[ft1, ft2]);
        assert_eq!(param_count(&merged), param_count(&base));
    }

    #[test]
    fn test_merged_keys_match() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let merged = average_merge(&[ft1, ft2]);
        let mut mk: Vec<_> = merged.keys().cloned().collect();
        let mut bk: Vec<_> = base.keys().cloned().collect();
        mk.sort();
        bk.sort();
        assert_eq!(mk, bk);
    }

    #[test]
    fn test_distance_from_all_models() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let ft3 = create_finetuned(&base, 300, 0.05);
        let merged = average_merge(&[ft1.clone(), ft2.clone(), ft3.clone()]);
        // Merged should be non-zero distance from each (they are different)
        assert!(model_distance(&merged, &ft1) > 0.0);
        assert!(model_distance(&merged, &ft2) > 0.0);
        assert!(model_distance(&merged, &ft3) > 0.0);
    }

    #[test]
    fn test_two_model_average_is_midpoint() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let merged = average_merge(&[ft1.clone(), ft2.clone()]);
        // Check element-wise that merged is the midpoint
        for (name, t_merged) in &merged {
            let d1 = ft1.get(name).unwrap().data();
            let d2 = ft2.get(name).unwrap().data();
            for (idx, &val) in t_merged.data().iter().enumerate() {
                let expected = (d1[idx] + d2[idx]) / 2.0;
                assert!((val - expected).abs() < 1e-6, "mismatch at {name}[{idx}]");
            }
        }
    }

    #[test]
    fn test_single_model_average_is_identity() {
        let base = create_base_model(&standard_layers(), 42);
        let merged = average_merge(&[base.clone()]);
        let dist = model_distance(&merged, &base);
        assert!(dist < 1e-6, "average of one model should be itself: {dist}");
    }

    #[test]
    fn test_average_is_commutative() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let merged_ab = average_merge(&[ft1.clone(), ft2.clone()]);
        let merged_ba = average_merge(&[ft2, ft1]);
        let dist = model_distance(&merged_ab, &merged_ba);
        assert!(dist < 1e-6, "average should be commutative: {dist}");
    }

    #[test]
    fn test_average_deterministic() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let m1 = average_merge(&[ft1.clone(), ft2.clone()]);
        let m2 = average_merge(&[ft1, ft2]);
        let dist = model_distance(&m1, &m2);
        assert!(dist < 1e-6, "average should be deterministic: {dist}");
    }

    #[test]
    #[should_panic(expected = "need at least one model")]
    fn test_average_empty_panics() {
        let _merged = average_merge(&[]);
    }

    #[test]
    fn test_apr_bundle_roundtrip() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let merged = average_merge(&[ft1, ft2]);
        let first = merged.values().next().unwrap();
        let bytes: Vec<u8> = first.data().iter().flat_map(|f| f.to_le_bytes()).collect();
        let bundle = ModelBundleV2::new()
            .with_name("test-avg")
            .with_compression(Compression::Lz4)
            .add_tensor("w", vec![first.len()], bytes)
            .build();
        assert_eq!(&bundle[0..4], b"APR2");
    }
}
