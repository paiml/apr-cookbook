//! SLERP Model Merge
//!
//! CLI equivalent: `apr merge --strategy slerp`
//! Contract: contracts/recipe-iiur-v1.yaml
//!
//! Spherical Linear Interpolation (SLERP) interpolates between two models
//! along a geodesic on the hypersphere rather than a straight line. This
//! preserves the magnitude (norm) of the parameter vectors, which can be
//! important for maintaining model quality since neural network weights often
//! live on a manifold where norm matters.
//!
//! ## Algorithm
//!
//! ```text
//! slerp(v0, v1, t) = sin((1-t)*omega)/sin(omega) * v0 + sin(t*omega)/sin(omega) * v1
//! where omega = arccos(dot(v0, v1) / (|v0| * |v1|))
//! ```
//!
//! ## When to Use
//!
//! - Merging two models where you want smooth interpolation
//! - Preserving weight magnitude during merge
//! - When linear averaging causes quality degradation
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
use entrenar::merge::{slerp_merge, SlerpConfig};
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
    let ctx = RecipeContext::new("merge_slerp")?;

    // --- Standard architecture layers ---
    let layers: Vec<(&str, usize)> = vec![
        ("attn.q_proj", 256),
        ("attn.k_proj", 256),
        ("attn.v_proj", 256),
        ("mlp.gate_proj", 512),
        ("mlp.up_proj", 512),
        ("mlp.down_proj", 512),
    ];

    // --- Section 1: Create two fine-tuned variants ---
    let base = create_base_model(&layers, 42);
    let ft_math = create_finetuned(&base, 100, 0.05);
    let ft_code = create_finetuned(&base, 200, 0.05);

    println!("Model A (math): {} parameters", param_count(&ft_math));
    println!("Model B (code): {} parameters", param_count(&ft_code));
    let d_ab = model_distance(&ft_math, &ft_code);
    println!("Distance A <-> B: {d_ab:.6}");

    // --- Section 2: Basic SLERP at t=0.5 ---
    let config_half = SlerpConfig::new(0.5).unwrap();
    let merged_half = slerp_merge(&ft_math, &ft_code, &config_half).unwrap();
    let d_half_a = model_distance(&merged_half, &ft_math);
    let d_half_b = model_distance(&merged_half, &ft_code);

    println!("\n--- SLERP t=0.5 ---");
    println!("Distance -> A: {d_half_a:.6}");
    println!("Distance -> B: {d_half_b:.6}");
    println!("Ratio (should be ~1.0): {:.4}", d_half_a / d_half_b);

    // --- Section 3: Interpolation sweep ---
    println!("\n--- Interpolation Sweep ---");
    println!("{:<8} {:<12} {:<12}", "t", "dist->A", "dist->B");
    for &t in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
        let config = SlerpConfig::new(t).unwrap();
        let merged = slerp_merge(&ft_math, &ft_code, &config).unwrap();
        let da = model_distance(&merged, &ft_math);
        let db = model_distance(&merged, &ft_code);
        println!("{t:<8.2} {da:<12.6} {db:<12.6}");
    }

    // --- Section 4: Endpoint verification ---
    let config_zero = SlerpConfig::new(0.0).unwrap();
    let merged_zero = slerp_merge(&ft_math, &ft_code, &config_zero).unwrap();
    let d_zero = model_distance(&merged_zero, &ft_math);

    let config_one = SlerpConfig::new(1.0).unwrap();
    let merged_one = slerp_merge(&ft_math, &ft_code, &config_one).unwrap();
    let d_one = model_distance(&merged_one, &ft_code);

    println!("\n--- Endpoint Verification ---");
    println!("t=0.0: distance from A = {d_zero:.8} (should be ~0)");
    println!("t=1.0: distance from B = {d_one:.8} (should be ~0)");

    // --- Section 5: Save merged model ---
    let first_tensor = merged_half.values().next().unwrap();
    let bytes: Vec<u8> = first_tensor
        .data()
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let bundle = ModelBundleV2::new()
        .with_name("slerp-merged")
        .with_compression(Compression::Lz4)
        .add_tensor("merged_weights", vec![first_tensor.len()], bytes)
        .build();

    assert_eq!(&bundle[0..4], b"APR2");
    println!(
        "\nSaved SLERP-merged model as APR v2 ({} bytes)",
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
    fn test_slerp_t0_near_model_a() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let config = SlerpConfig::new(0.0).unwrap();
        let merged = slerp_merge(&ft1, &ft2, &config).unwrap();
        let dist = model_distance(&merged, &ft1);
        assert!(dist < 1e-4, "t=0 should be near model A: {dist}");
    }

    #[test]
    fn test_slerp_t1_near_model_b() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let config = SlerpConfig::new(1.0).unwrap();
        let merged = slerp_merge(&ft1, &ft2, &config).unwrap();
        let dist = model_distance(&merged, &ft2);
        assert!(dist < 1e-4, "t=1 should be near model B: {dist}");
    }

    #[test]
    fn test_slerp_midpoint_approximately_equidistant() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let config = SlerpConfig::new(0.5).unwrap();
        let merged = slerp_merge(&ft1, &ft2, &config).unwrap();
        let d1 = model_distance(&merged, &ft1);
        let d2 = model_distance(&merged, &ft2);
        let ratio = d1 / d2;
        assert!(
            (0.5..2.0).contains(&ratio),
            "midpoint should be roughly equidistant: d1={d1}, d2={d2}, ratio={ratio}"
        );
    }

    #[test]
    fn test_slerp_config_validation() {
        assert!(SlerpConfig::new(0.0).is_ok());
        assert!(SlerpConfig::new(0.5).is_ok());
        assert!(SlerpConfig::new(1.0).is_ok());
    }

    #[test]
    fn test_slerp_param_count_preserved() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let config = SlerpConfig::new(0.5).unwrap();
        let merged = slerp_merge(&ft1, &ft2, &config).unwrap();
        assert_eq!(param_count(&merged), param_count(&base));
    }

    #[test]
    fn test_slerp_keys_preserved() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let config = SlerpConfig::new(0.5).unwrap();
        let merged = slerp_merge(&ft1, &ft2, &config).unwrap();
        let mut mk: Vec<_> = merged.keys().cloned().collect();
        let mut bk: Vec<_> = base.keys().cloned().collect();
        mk.sort();
        bk.sort();
        assert_eq!(mk, bk);
    }

    #[test]
    fn test_slerp_monotonic_distance() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        // As t increases, distance from ft1 should generally increase
        let mut prev_dist = 0.0_f32;
        for &t in &[0.0_f32, 0.25, 0.5, 0.75, 1.0] {
            let config = SlerpConfig::new(t).unwrap();
            let merged = slerp_merge(&ft1, &ft2, &config).unwrap();
            let dist = model_distance(&merged, &ft1);
            assert!(
                dist >= prev_dist - 1e-4,
                "distance from A should increase with t: t={t}, dist={dist}, prev={prev_dist}"
            );
            prev_dist = dist;
        }
    }

    #[test]
    fn test_slerp_deterministic() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let config = SlerpConfig::new(0.3).unwrap();
        let m1 = slerp_merge(&ft1, &ft2, &config).unwrap();
        let m2 = slerp_merge(&ft1, &ft2, &config).unwrap();
        let dist = model_distance(&m1, &m2);
        assert!(dist < 1e-6, "slerp should be deterministic: {dist}");
    }

    #[test]
    fn test_slerp_apr_bundle() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let config = SlerpConfig::new(0.5).unwrap();
        let merged = slerp_merge(&ft1, &ft2, &config).unwrap();
        let first = merged.values().next().unwrap();
        let bytes: Vec<u8> = first.data().iter().flat_map(|f| f.to_le_bytes()).collect();
        let bundle = ModelBundleV2::new()
            .with_name("test-slerp")
            .with_compression(Compression::Lz4)
            .add_tensor("w", vec![first.len()], bytes)
            .build();
        assert_eq!(&bundle[0..4], b"APR2");
    }
}
