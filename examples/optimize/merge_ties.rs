//! TIES Model Merge
//!
//! CLI equivalent: `apr merge --strategy ties --density 0.2`
//!
//! TIES (TrIm, Elect Sign & merge) is a task-arithmetic merge strategy that
//! resolves interference between fine-tuned models by:
//!
//! 1. **Trim**: Prune low-magnitude task vectors (deltas from base), keeping
//!    only the top-k% by absolute value (controlled by `density`).
//! 2. **Elect Sign**: For each parameter position, resolve sign conflicts
//!    across models via majority voting.
//! 3. **Merge**: Average only the values that agree on sign.
//!
//! This reduces interference between task-specific fine-tunes, producing
//! cleaner multi-task merges than simple averaging.
//!
//! ## When to Use
//!
//! - Merging 3+ models fine-tuned on different tasks
//! - When simple averaging degrades individual task performance
//! - When models have conflicting gradient directions
//!
//! ## References
//! - Wortsman, M. et al. (2022). *Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy*. ICML. arXiv:2203.05482

use apr_cookbook::prelude::*;
use entrenar::autograd::Tensor;
use entrenar::merge::{ties_merge, TiesConfig};
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
    let ctx = RecipeContext::new("merge_ties")?;

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

    // --- Section 2: TIES merge at density=0.2 ---
    let config = TiesConfig::new(0.2).unwrap();
    let merged = ties_merge(&models, &base, &config).unwrap();

    let d_base = model_distance(&merged, &base);
    let d_math = model_distance(&merged, &ft_math);
    let d_code = model_distance(&merged, &ft_code);
    let d_writing = model_distance(&merged, &ft_writing);

    println!("\n--- TIES Merge (density=0.2) ---");
    println!("Distance -> base:    {d_base:.6}");
    println!("Distance -> math:    {d_math:.6}");
    println!("Distance -> code:    {d_code:.6}");
    println!("Distance -> writing: {d_writing:.6}");

    // --- Section 3: Density sweep ---
    println!("\n--- Density Sweep ---");
    println!(
        "{:<10} {:<14} {:<14} {:<14}",
        "density", "dist->base", "dist->math", "dist->code"
    );
    for &density in &[0.1_f32, 0.2, 0.5, 0.8] {
        let cfg = TiesConfig::new(density).unwrap();
        let m = ties_merge(&models, &base, &cfg).unwrap();
        let db = model_distance(&m, &base);
        let dm = model_distance(&m, &ft_math);
        let dc = model_distance(&m, &ft_code);
        println!("{density:<10.1} {db:<14.6} {dm:<14.6} {dc:<14.6}");
    }

    // --- Section 4: Sign consensus demonstration ---
    println!("\n--- Sign Consensus ---");
    println!("TIES resolves sign conflicts via majority voting.");
    println!("At low density, only the strongest (highest magnitude) deltas survive.");
    println!("At high density, more parameters participate but with more sign conflicts.");

    // Lower density = closer to base (fewer deltas survive)
    let low_density = ties_merge(&models, &base, &TiesConfig::new(0.1).unwrap()).unwrap();
    let high_density = ties_merge(&models, &base, &TiesConfig::new(0.8).unwrap()).unwrap();
    let d_low_base = model_distance(&low_density, &base);
    let d_high_base = model_distance(&high_density, &base);
    println!("Distance from base at density=0.1: {d_low_base:.6}");
    println!("Distance from base at density=0.8: {d_high_base:.6}");

    // --- Section 5: Save merged model ---
    let first_tensor = merged.values().next().unwrap();
    let bytes: Vec<u8> = first_tensor
        .data()
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let bundle = ModelBundleV2::new()
        .with_name("ties-merged")
        .with_compression(Compression::Lz4)
        .add_tensor("merged_weights", vec![first_tensor.len()], bytes)
        .build();

    assert_eq!(&bundle[0..4], b"APR2");
    println!(
        "\nSaved TIES-merged model as APR v2 ({} bytes)",
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
    fn test_ties_merged_has_same_structure() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let models = vec![ft1, ft2];
        let config = TiesConfig::new(0.2).unwrap();
        let merged = ties_merge(&models, &base, &config).unwrap();
        let mut mk: Vec<_> = merged.keys().cloned().collect();
        let mut bk: Vec<_> = base.keys().cloned().collect();
        mk.sort();
        bk.sort();
        assert_eq!(mk, bk);
    }

    #[test]
    fn test_ties_param_count_preserved() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let models = vec![ft1, ft2];
        let config = TiesConfig::new(0.5).unwrap();
        let merged = ties_merge(&models, &base, &config).unwrap();
        assert_eq!(param_count(&merged), param_count(&base));
    }

    #[test]
    fn test_ties_density_validation() {
        assert!(TiesConfig::new(0.0).is_ok());
        assert!(TiesConfig::new(0.5).is_ok());
        assert!(TiesConfig::new(1.0).is_ok());
    }

    #[test]
    fn test_ties_lower_density_closer_to_base() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let ft3 = create_finetuned(&base, 300, 0.05);
        let models = vec![ft1, ft2, ft3];

        let low = ties_merge(&models, &base, &TiesConfig::new(0.1).unwrap()).unwrap();
        let high = ties_merge(&models, &base, &TiesConfig::new(0.8).unwrap()).unwrap();
        let d_low = model_distance(&low, &base);
        let d_high = model_distance(&high, &base);
        assert!(
            d_low <= d_high + 1e-4,
            "lower density should be closer to base: d_low={d_low}, d_high={d_high}"
        );
    }

    #[test]
    fn test_ties_deterministic() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let models = vec![ft1, ft2];
        let config = TiesConfig::new(0.3).unwrap();
        let m1 = ties_merge(&models, &base, &config).unwrap();
        let m2 = ties_merge(&models, &base, &config).unwrap();
        let dist = model_distance(&m1, &m2);
        assert!(dist < 1e-6, "ties merge should be deterministic: {dist}");
    }

    #[test]
    fn test_ties_merged_differs_from_base() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let models = vec![ft1, ft2];
        let config = TiesConfig::new(0.5).unwrap();
        let merged = ties_merge(&models, &base, &config).unwrap();
        let dist = model_distance(&merged, &base);
        assert!(dist > 0.0, "ties merged should differ from base");
    }

    #[test]
    fn test_ties_full_density_includes_all_deltas() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let models = vec![ft1, ft2];
        let config = TiesConfig::new(1.0).unwrap();
        let merged = ties_merge(&models, &base, &config).unwrap();
        let dist = model_distance(&merged, &base);
        // With density=1.0, all deltas participate, should be far from base
        assert!(dist > 0.0, "full density should move away from base");
    }

    #[test]
    fn test_ties_three_models() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let ft3 = create_finetuned(&base, 300, 0.05);
        let models = vec![ft1.clone(), ft2.clone(), ft3.clone()];
        let config = TiesConfig::new(0.3).unwrap();
        let merged = ties_merge(&models, &base, &config).unwrap();
        // Should be within reasonable distance of all sources
        let d1 = model_distance(&merged, &ft1);
        let d2 = model_distance(&merged, &ft2);
        let d3 = model_distance(&merged, &ft3);
        assert!(d1 > 0.0 && d2 > 0.0 && d3 > 0.0);
    }

    #[test]
    fn test_ties_apr_bundle() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let models = vec![ft1, ft2];
        let config = TiesConfig::new(0.2).unwrap();
        let merged = ties_merge(&models, &base, &config).unwrap();
        let first = merged.values().next().unwrap();
        let bytes: Vec<u8> = first.data().iter().flat_map(|f| f.to_le_bytes()).collect();
        let bundle = ModelBundleV2::new()
            .with_name("test-ties")
            .with_compression(Compression::Lz4)
            .add_tensor("w", vec![first.len()], bytes)
            .build();
        assert_eq!(&bundle[0..4], b"APR2");
    }
}
