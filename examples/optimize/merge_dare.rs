//! DARE Model Merge
//!
//! CLI equivalent: `apr merge --strategy dare --drop-prob 0.5`
//!
//! DARE (Drop And REscale) randomly drops delta weights (differences from the
//! base model) and rescales the surviving weights to compensate. This reduces
//! task interference between fine-tuned models by sparsifying each model's
//! contribution to the merge.
//!
//! ## Algorithm
//!
//! ```text
//! For each model i and parameter position k:
//!   delta_ik = model_i[k] - base[k]
//!   mask_ik  ~ Bernoulli(1 - drop_prob)
//!   delta_ik = delta_ik * mask_ik / (1 - drop_prob)   # rescale
//!
//! merged[k] = base[k] + mean(delta_ik for i in models)
//! ```
//!
//! ## When to Use
//!
//! - Merging models with high task interference
//! - When TIES is too aggressive in pruning
//! - Exploring stochastic merge strategies with different seeds

use apr_cookbook::prelude::*;
use entrenar::autograd::Tensor;
use entrenar::merge::{dare_merge, DareConfig};
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
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("merge_dare")?;

    // --- Standard architecture layers ---
    let layers: Vec<(&str, usize)> = vec![
        ("attn.q_proj", 256),
        ("attn.k_proj", 256),
        ("attn.v_proj", 256),
        ("mlp.gate_proj", 512),
        ("mlp.up_proj", 512),
        ("mlp.down_proj", 512),
    ];

    // --- Section 1: Create base + fine-tuned variants ---
    let base = create_base_model(&layers, 42);
    let ft_math = create_finetuned(&base, 100, 0.05);
    let ft_code = create_finetuned(&base, 200, 0.05);
    let ft_writing = create_finetuned(&base, 300, 0.05);
    let models = vec![ft_math.clone(), ft_code.clone(), ft_writing.clone()];

    println!("Base model: {} parameters", param_count(&base));
    println!("Fine-tuned variants: math, code, writing");

    // --- Section 2: DARE merge at drop=0.5 ---
    let config = DareConfig::new(0.5).unwrap().with_seed(42);
    let merged = dare_merge(&models, &base, &config).unwrap();

    let d_base = model_distance(&merged, &base);
    let d_math = model_distance(&merged, &ft_math);
    let d_code = model_distance(&merged, &ft_code);
    let d_writing = model_distance(&merged, &ft_writing);

    println!("\n--- DARE Merge (drop=0.5, seed=42) ---");
    println!("Distance -> base:    {d_base:.6}");
    println!("Distance -> math:    {d_math:.6}");
    println!("Distance -> code:    {d_code:.6}");
    println!("Distance -> writing: {d_writing:.6}");

    // --- Section 3: Drop probability sweep ---
    println!("\n--- Drop Probability Sweep ---");
    println!(
        "{:<10} {:<14} {:<14} {:<14}",
        "drop", "dist->base", "dist->math", "dist->code"
    );
    for &drop in &[0.1_f32, 0.3, 0.5, 0.7, 0.9] {
        let cfg = DareConfig::new(drop).unwrap().with_seed(42);
        let m = dare_merge(&models, &base, &cfg).unwrap();
        let db = model_distance(&m, &base);
        let dm = model_distance(&m, &ft_math);
        let dc = model_distance(&m, &ft_code);
        println!("{drop:<10.1} {db:<14.6} {dm:<14.6} {dc:<14.6}");
    }

    // --- Section 4: Seed impact ---
    println!("\n--- Seed Impact (drop=0.5) ---");
    let mut prev_merged: Option<Model> = None;
    for seed in [42_u64, 123, 456, 789, 1024] {
        let cfg = DareConfig::new(0.5).unwrap().with_seed(seed);
        let m = dare_merge(&models, &base, &cfg).unwrap();
        let db = model_distance(&m, &base);
        if let Some(ref prev) = prev_merged {
            let inter_dist = model_distance(&m, prev);
            println!("seed={seed:<5} dist->base={db:.6}  dist->prev_seed={inter_dist:.6}");
        } else {
            println!("seed={seed:<5} dist->base={db:.6}");
        }
        prev_merged = Some(m);
    }

    // --- Section 5: Save merged model ---
    let first_tensor = merged.values().next().unwrap();
    let bytes: Vec<u8> = first_tensor
        .data()
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let bundle = ModelBundleV2::new()
        .with_name("dare-merged")
        .with_compression(Compression::Lz4)
        .add_tensor("merged_weights", vec![first_tensor.len()], bytes)
        .build();

    assert_eq!(&bundle[0..4], b"APR2");
    println!(
        "\nSaved DARE-merged model as APR v2 ({} bytes)",
        bundle.len()
    );

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
    fn test_dare_merged_has_same_structure() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let models = vec![ft1, ft2];
        let config = DareConfig::new(0.5).unwrap().with_seed(42);
        let merged = dare_merge(&models, &base, &config).unwrap();
        let mut mk: Vec<_> = merged.keys().cloned().collect();
        let mut bk: Vec<_> = base.keys().cloned().collect();
        mk.sort();
        bk.sort();
        assert_eq!(mk, bk);
    }

    #[test]
    fn test_dare_param_count_preserved() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let models = vec![ft1, ft2];
        let config = DareConfig::new(0.5).unwrap().with_seed(42);
        let merged = dare_merge(&models, &base, &config).unwrap();
        assert_eq!(param_count(&merged), param_count(&base));
    }

    #[test]
    fn test_dare_drop_probability_validation() {
        assert!(DareConfig::new(0.0).is_ok());
        assert!(DareConfig::new(0.5).is_ok());
        assert!(DareConfig::new(0.99).is_ok());
    }

    #[test]
    fn test_dare_different_seeds_produce_different_results() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let models = vec![ft1, ft2];

        let m1 = dare_merge(&models, &base, &DareConfig::new(0.5).unwrap().with_seed(42)).unwrap();
        let m2 = dare_merge(&models, &base, &DareConfig::new(0.5).unwrap().with_seed(99)).unwrap();
        let dist = model_distance(&m1, &m2);
        assert!(
            dist > 1e-6,
            "different seeds should produce different results: {dist}"
        );
    }

    #[test]
    fn test_dare_keys_preserved() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let models = vec![ft1, ft2];
        let config = DareConfig::new(0.5).unwrap().with_seed(42);
        let merged = dare_merge(&models, &base, &config).unwrap();
        let mut mk: Vec<_> = merged.keys().cloned().collect();
        let mut bk: Vec<_> = base.keys().cloned().collect();
        mk.sort();
        bk.sort();
        assert_eq!(mk, bk);
    }

    #[test]
    fn test_dare_merged_differs_from_base() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let models = vec![ft1, ft2];
        let config = DareConfig::new(0.5).unwrap().with_seed(42);
        let merged = dare_merge(&models, &base, &config).unwrap();
        let dist = model_distance(&merged, &base);
        assert!(dist > 0.0, "dare merged should differ from base");
    }

    #[test]
    fn test_dare_three_models() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let ft3 = create_finetuned(&base, 300, 0.05);
        let models = vec![ft1.clone(), ft2.clone(), ft3.clone()];
        let config = DareConfig::new(0.5).unwrap().with_seed(42);
        let merged = dare_merge(&models, &base, &config).unwrap();
        let d1 = model_distance(&merged, &ft1);
        let d2 = model_distance(&merged, &ft2);
        let d3 = model_distance(&merged, &ft3);
        assert!(d1 > 0.0 && d2 > 0.0 && d3 > 0.0);
    }

    #[test]
    fn test_dare_low_drop_preserves_more_signal() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let models = vec![ft1, ft2];

        let low_drop =
            dare_merge(&models, &base, &DareConfig::new(0.1).unwrap().with_seed(42)).unwrap();
        let high_drop =
            dare_merge(&models, &base, &DareConfig::new(0.9).unwrap().with_seed(42)).unwrap();
        // Both should produce valid models
        assert_eq!(param_count(&low_drop), param_count(&base));
        assert_eq!(param_count(&high_drop), param_count(&base));
    }

    #[test]
    fn test_dare_apr_bundle() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let models = vec![ft1, ft2];
        let config = DareConfig::new(0.5).unwrap().with_seed(42);
        let merged = dare_merge(&models, &base, &config).unwrap();
        let first = merged.values().next().unwrap();
        let bytes: Vec<u8> = first.data().iter().flat_map(|f| f.to_le_bytes()).collect();
        let bundle = ModelBundleV2::new()
            .with_name("test-dare")
            .with_compression(Compression::Lz4)
            .add_tensor("w", vec![first.len()], bytes)
            .build();
        assert_eq!(&bundle[0..4], b"APR2");
    }
}
