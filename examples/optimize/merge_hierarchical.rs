//! Hierarchical Model Merge
//!
//! CLI equivalent: composed multi-model merge pipeline
//!
//! Hierarchical merging applies different strategies at different stages of
//! a merge pipeline. Instead of merging all models at once, you group related
//! models, merge each group with SLERP (pairwise interpolation), then combine
//! the group results with TIES (multi-model task arithmetic).
//!
//! ## Pipeline
//!
//! ```text
//! Stage 1: SLERP(math, code)      -> stem_analytical   (t=0.5)
//! Stage 2: SLERP(reasoning, writing) -> stem_creative   (t=0.5)
//! Stage 3: TIES(stem_analytical, stem_creative, base)   (density=0.3)
//! ```
//!
//! ## When to Use
//!
//! - Merging 4+ models with natural groupings (e.g., STEM + humanities)
//! - When flat merge of all models causes too much interference
//! - When you want fine-grained control over how capabilities combine
//!
//!
//! ## Format Variants
//! ```bash
//! apr inspect model.apr          # APR native format
//! apr inspect model.gguf         # GGUF (llama.cpp compatible)
//! apr inspect model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Wortsman, M. et al. (2022). *Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy*. ICML. arXiv:2203.05482

use apr_cookbook::prelude::*;
use entrenar::autograd::Tensor;
use entrenar::merge::{slerp_merge, ties_merge, SlerpConfig, TiesConfig};
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

/// Flat average merge for comparison.
fn average_merge(models: &[Model]) -> Model {
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
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let ctx = RecipeContext::new("merge_hierarchical")?;

    // --- Standard architecture layers ---
    let layers: Vec<(&str, usize)> = vec![
        ("attn.q_proj", 256),
        ("attn.k_proj", 256),
        ("attn.v_proj", 256),
        ("mlp.gate_proj", 512),
        ("mlp.up_proj", 512),
        ("mlp.down_proj", 512),
    ];

    // --- Section 1: Create 4 task-specific models ---
    let base = create_base_model(&layers, 42);
    let ft_math = create_finetuned(&base, 100, 0.05);
    let ft_code = create_finetuned(&base, 200, 0.05);
    let ft_reasoning = create_finetuned(&base, 300, 0.05);
    let ft_writing = create_finetuned(&base, 400, 0.05);

    println!("Base model: {} parameters", param_count(&base));
    println!("Task models: math, code, reasoning, writing");

    // --- Section 2: Stage 1 - Pairwise SLERP ---
    let slerp_config = SlerpConfig::new(0.5).unwrap();

    // Group 1: analytical (math + code)
    let stem_analytical = slerp_merge(&ft_math, &ft_code, &slerp_config).unwrap();
    println!("\nStage 1: SLERP(math, code) -> stem_analytical");
    println!(
        "  dist math->analytical: {:.6}",
        model_distance(&ft_math, &stem_analytical)
    );
    println!(
        "  dist code->analytical: {:.6}",
        model_distance(&ft_code, &stem_analytical)
    );

    // Group 2: creative (reasoning + writing)
    let stem_creative = slerp_merge(&ft_reasoning, &ft_writing, &slerp_config).unwrap();
    println!("\nStage 2: SLERP(reasoning, writing) -> stem_creative");
    println!(
        "  dist reasoning->creative: {:.6}",
        model_distance(&ft_reasoning, &stem_creative)
    );
    println!(
        "  dist writing->creative:   {:.6}",
        model_distance(&ft_writing, &stem_creative)
    );

    // --- Section 3: Stage 2 - Final TIES merge ---
    let ties_config = TiesConfig::new(0.3).unwrap();
    let stems = vec![stem_analytical.clone(), stem_creative.clone()];
    let hierarchical = ties_merge(&stems, &base, &ties_config).unwrap();

    println!("\nStage 3: TIES(analytical, creative, base) -> final");
    println!(
        "  dist -> base:       {:.6}",
        model_distance(&hierarchical, &base)
    );
    println!(
        "  dist -> analytical: {:.6}",
        model_distance(&hierarchical, &stem_analytical)
    );
    println!(
        "  dist -> creative:   {:.6}",
        model_distance(&hierarchical, &stem_creative)
    );

    // --- Section 4: Compare vs flat merge ---
    let all_models = vec![
        ft_math.clone(),
        ft_code.clone(),
        ft_reasoning.clone(),
        ft_writing.clone(),
    ];
    let flat_avg = average_merge(&all_models);
    let flat_ties = ties_merge(&all_models, &base, &ties_config).unwrap();

    let d_hier_base = model_distance(&hierarchical, &base);
    let d_flat_avg_base = model_distance(&flat_avg, &base);
    let d_flat_ties_base = model_distance(&flat_ties, &base);

    println!("\n--- Hierarchical vs Flat Merge ---");
    println!("Hierarchical (SLERP+TIES) dist->base: {d_hier_base:.6}");
    println!("Flat average dist->base:               {d_flat_avg_base:.6}");
    println!("Flat TIES dist->base:                  {d_flat_ties_base:.6}");

    let d_hier_flat = model_distance(&hierarchical, &flat_avg);
    println!("Hierarchical <-> flat average: {d_hier_flat:.6}");

    // Distances from each source
    println!("\n--- Per-Source Distances ---");
    println!("{:<14} {:<14} {:<14}", "source", "hierarchical", "flat_avg");
    for (name, model) in [
        ("math", &ft_math),
        ("code", &ft_code),
        ("reasoning", &ft_reasoning),
        ("writing", &ft_writing),
    ] {
        let dh = model_distance(&hierarchical, model);
        let df = model_distance(&flat_avg, model);
        println!("{name:<14} {dh:<14.6} {df:<14.6}");
    }

    // --- Save ---
    let first_tensor = hierarchical.values().next().unwrap();
    let bytes: Vec<u8> = first_tensor
        .data()
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let bundle = ModelBundleV2::new()
        .with_name("hierarchical-merged")
        .with_compression(Compression::Lz4)
        .add_tensor("merged_weights", vec![first_tensor.len()], bytes)
        .build();

    assert_eq!(&bundle[0..4], b"APR2");
    println!(
        "\nSaved hierarchical-merged model as APR v2 ({} bytes)",
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

    fn build_pipeline() -> (Model, Model, Model, Model, Model, Model) {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let ft3 = create_finetuned(&base, 300, 0.05);
        let ft4 = create_finetuned(&base, 400, 0.05);

        let slerp_cfg = SlerpConfig::new(0.5).unwrap();
        let stem_a = slerp_merge(&ft1, &ft2, &slerp_cfg).unwrap();
        let stem_b = slerp_merge(&ft3, &ft4, &slerp_cfg).unwrap();

        let ties_cfg = TiesConfig::new(0.3).unwrap();
        let final_model = ties_merge(&[stem_a, stem_b], &base, &ties_cfg).unwrap();

        (base, ft1, ft2, ft3, ft4, final_model)
    }

    #[test]
    fn test_hierarchical_produces_valid_model() {
        let (base, _, _, _, _, merged) = build_pipeline();
        assert_eq!(merged.len(), base.len());
        for (name, tensor) in &merged {
            assert_eq!(tensor.len(), base.get(name).unwrap().len());
        }
    }

    #[test]
    fn test_hierarchical_param_count_preserved() {
        let (base, _, _, _, _, merged) = build_pipeline();
        assert_eq!(param_count(&merged), param_count(&base));
    }

    #[test]
    fn test_hierarchical_keys_match() {
        let (base, _, _, _, _, merged) = build_pipeline();
        let mut mk: Vec<_> = merged.keys().cloned().collect();
        let mut bk: Vec<_> = base.keys().cloned().collect();
        mk.sort();
        bk.sort();
        assert_eq!(mk, bk);
    }

    #[test]
    fn test_hierarchical_distance_from_all_sources() {
        let (_, ft1, ft2, ft3, ft4, merged) = build_pipeline();
        let d1 = model_distance(&merged, &ft1);
        let d2 = model_distance(&merged, &ft2);
        let d3 = model_distance(&merged, &ft3);
        let d4 = model_distance(&merged, &ft4);
        // Merged should differ from all sources
        assert!(d1 > 0.0 && d2 > 0.0 && d3 > 0.0 && d4 > 0.0);
    }

    #[test]
    fn test_hierarchical_differs_from_flat_average() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let ft3 = create_finetuned(&base, 300, 0.05);
        let ft4 = create_finetuned(&base, 400, 0.05);

        let flat = average_merge(&[ft1.clone(), ft2.clone(), ft3.clone(), ft4.clone()]);

        let slerp_cfg = SlerpConfig::new(0.5).unwrap();
        let stem_a = slerp_merge(&ft1, &ft2, &slerp_cfg).unwrap();
        let stem_b = slerp_merge(&ft3, &ft4, &slerp_cfg).unwrap();
        let hier = ties_merge(&[stem_a, stem_b], &base, &TiesConfig::new(0.3).unwrap()).unwrap();

        let dist = model_distance(&hier, &flat);
        assert!(dist > 1e-6, "hierarchical should differ from flat: {dist}");
    }

    #[test]
    fn test_hierarchical_differs_from_base() {
        let (base, _, _, _, _, merged) = build_pipeline();
        let dist = model_distance(&merged, &base);
        assert!(dist > 0.0, "hierarchical should differ from base");
    }

    #[test]
    fn test_slerp_stage_equidistant_from_inputs() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let slerp_cfg = SlerpConfig::new(0.5).unwrap();
        let stem = slerp_merge(&ft1, &ft2, &slerp_cfg).unwrap();
        let d1 = model_distance(&stem, &ft1);
        let d2 = model_distance(&stem, &ft2);
        let ratio = d1 / d2;
        assert!(
            (0.5..2.0).contains(&ratio),
            "SLERP at t=0.5 should be roughly equidistant: {ratio}"
        );
    }

    #[test]
    fn test_hierarchical_deterministic() {
        let layers = standard_layers();
        let base = create_base_model(&layers, 42);
        let ft1 = create_finetuned(&base, 100, 0.05);
        let ft2 = create_finetuned(&base, 200, 0.05);
        let ft3 = create_finetuned(&base, 300, 0.05);
        let ft4 = create_finetuned(&base, 400, 0.05);

        let run = || {
            let slerp_cfg = SlerpConfig::new(0.5).unwrap();
            let sa = slerp_merge(&ft1, &ft2, &slerp_cfg).unwrap();
            let sb = slerp_merge(&ft3, &ft4, &slerp_cfg).unwrap();
            ties_merge(&[sa, sb], &base, &TiesConfig::new(0.3).unwrap()).unwrap()
        };

        let m1 = run();
        let m2 = run();
        let dist = model_distance(&m1, &m2);
        assert!(dist < 1e-6, "hierarchical should be deterministic: {dist}");
    }

    #[test]
    fn test_hierarchical_apr_bundle() {
        let (_, _, _, _, _, merged) = build_pipeline();
        let first = merged.values().next().unwrap();
        let bytes: Vec<u8> = first.data().iter().flat_map(|f| f.to_le_bytes()).collect();
        let bundle = ModelBundleV2::new()
            .with_name("test-hier")
            .with_compression(Compression::Lz4)
            .add_tensor("w", vec![first.len()], bytes)
            .build();
        assert_eq!(&bundle[0..4], b"APR2");
    }
}
