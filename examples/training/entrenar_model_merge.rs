//! Entrenar Model Merging Example
//!
//! Demonstrates TIES, DARE, and SLERP model merging methods for combining
//! multiple fine-tuned models into a single merged model.
//!
//! # Merging Methods
//!
//! - **TIES**: Task Inference via Elimination and Sign — prunes low-magnitude
//!   deltas, resolves sign conflicts via majority voting
//! - **DARE**: Drop And REscale — randomly drops delta weights and rescales,
//!   reducing interference between task-specific adaptations
//! - **SLERP**: Spherical Linear Interpolation — smoothly blends two models
//!   along the hypersphere for geometrically meaningful interpolation
//!
//! # Running
//!
//! ```bash
//! cargo run --example entrenar_model_merge
//! ```

use apr_cookbook::prelude::*;
use entrenar::autograd::Tensor;
use entrenar::merge::{dare_merge, slerp_merge, ties_merge, DareConfig, SlerpConfig, TiesConfig};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

type Model = HashMap<String, Tensor>;

/// Create a synthetic base model with random weights
fn create_base_model(layers: &[(&str, usize)], seed: u64) -> Model {
    let mut model = HashMap::new();
    for (name, size) in layers {
        let data: Vec<f32> = (0..*size)
            .map(|i| {
                let mut hasher = DefaultHasher::new();
                (seed, *name, i).hash(&mut hasher);
                (hasher.finish() as f32 / u64::MAX as f32 - 0.5) * 0.1
            })
            .collect();
        model.insert(name.to_string(), Tensor::from_vec(data, false));
    }
    model
}

/// Create a fine-tuned variant by adding task-specific deltas to the base
fn create_finetuned_model(base: &Model, task_seed: u64, delta_scale: f32) -> Model {
    let mut model = HashMap::new();
    for (name, tensor) in base {
        let delta: Vec<f32> = (0..tensor.len())
            .map(|i| {
                let mut hasher = DefaultHasher::new();
                (task_seed, name.as_str(), i).hash(&mut hasher);
                (hasher.finish() as f32 / u64::MAX as f32 - 0.5) * delta_scale
            })
            .collect();
        let merged: Vec<f32> = tensor
            .data()
            .iter()
            .zip(delta.iter())
            .map(|(b, d)| b + d)
            .collect();
        model.insert(name.clone(), Tensor::from_vec(merged, false));
    }
    model
}

/// Compute L2 distance between two models
fn model_distance(m1: &Model, m2: &Model) -> f32 {
    let mut total = 0.0f32;
    for (name, t1) in m1 {
        if let Some(t2) = m2.get(name) {
            let dist: f32 = t1
                .data()
                .iter()
                .zip(t2.data().iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            total += dist;
        }
    }
    total.sqrt()
}

/// Count total parameters in a model
fn param_count(model: &Model) -> usize {
    model.values().map(|t| t.len()).sum()
}

fn main() {
    println!("=== Entrenar Model Merging Example ===\n");

    // =========================================================================
    // Section 1: Create Base + Fine-Tuned Models
    // =========================================================================
    println!("1. Model Setup");
    println!("   ─────────────────────────────────────────");

    let layers: Vec<(&str, usize)> = vec![
        ("attn.q_proj", 256),
        ("attn.k_proj", 256),
        ("attn.v_proj", 256),
        ("mlp.gate_proj", 512),
        ("mlp.up_proj", 512),
        ("mlp.down_proj", 512),
    ];

    let base = create_base_model(&layers, 42);
    let model_task_a = create_finetuned_model(&base, 100, 0.05); // Math task
    let model_task_b = create_finetuned_model(&base, 200, 0.03); // Code task
    let model_task_c = create_finetuned_model(&base, 300, 0.04); // Reasoning task

    let n_params = param_count(&base);
    println!("   Base model:    {} params", n_params);
    println!(
        "   Task A (math):     dist={:.4} from base",
        model_distance(&model_task_a, &base)
    );
    println!(
        "   Task B (code):     dist={:.4} from base",
        model_distance(&model_task_b, &base)
    );
    println!(
        "   Task C (reason):   dist={:.4} from base",
        model_distance(&model_task_c, &base)
    );
    println!();

    let models = vec![
        model_task_a.clone(),
        model_task_b.clone(),
        model_task_c.clone(),
    ];

    // =========================================================================
    // Section 2: TIES Merge
    // =========================================================================
    println!("2. TIES Merge (density=0.2)");
    println!("   ─────────────────────────────────────────");

    match TiesConfig::new(0.2) {
        Ok(config) => match ties_merge(&models, &base, &config) {
            Ok(merged) => {
                let dist_from_base = model_distance(&merged, &base);
                let dist_from_a = model_distance(&merged, &model_task_a);
                let dist_from_b = model_distance(&merged, &model_task_b);
                println!("   Merged params: {}", param_count(&merged));
                println!("   Distance from base:   {:.4}", dist_from_base);
                println!("   Distance from task A:  {:.4}", dist_from_a);
                println!("   Distance from task B:  {:.4}", dist_from_b);
                println!("   TIES prunes low-magnitude deltas, keeps sign consensus");
            }
            Err(e) => println!("   Error: {}", e),
        },
        Err(e) => println!("   Config error: {}", e),
    }
    println!();

    // =========================================================================
    // Section 3: DARE Merge
    // =========================================================================
    println!("3. DARE Merge (drop_prob=0.5)");
    println!("   ─────────────────────────────────────────");

    match DareConfig::new(0.5) {
        Ok(config) => {
            let config = config.with_seed(42);
            match dare_merge(&models, &base, &config) {
                Ok(merged) => {
                    let dist_from_base = model_distance(&merged, &base);
                    let dist_from_a = model_distance(&merged, &model_task_a);
                    println!("   Merged params: {}", param_count(&merged));
                    println!("   Distance from base:   {:.4}", dist_from_base);
                    println!("   Distance from task A:  {:.4}", dist_from_a);
                    println!("   DARE randomly drops 50% of deltas, rescales remainder");
                }
                Err(e) => println!("   Error: {}", e),
            }
        }
        Err(e) => println!("   Config error: {}", e),
    }
    println!();

    // =========================================================================
    // Section 4: SLERP Merge (2 models)
    // =========================================================================
    println!("4. SLERP Merge (t=0.5)");
    println!("   ─────────────────────────────────────────");

    match SlerpConfig::new(0.5) {
        Ok(config) => match slerp_merge(&model_task_a, &model_task_b, &config) {
            Ok(merged) => {
                let dist_from_a = model_distance(&merged, &model_task_a);
                let dist_from_b = model_distance(&merged, &model_task_b);
                println!("   Merged params: {}", param_count(&merged));
                println!("   Distance from task A:  {:.4}", dist_from_a);
                println!("   Distance from task B:  {:.4}", dist_from_b);
                println!("   Ratio A/B: {:.2}", dist_from_a / dist_from_b.max(1e-10));
                println!("   SLERP interpolates along the hypersphere");
            }
            Err(e) => println!("   Error: {}", e),
        },
        Err(e) => println!("   Config error: {}", e),
    }
    println!();

    // =========================================================================
    // Section 5: SLERP Sweep
    // =========================================================================
    println!("5. SLERP Interpolation Sweep");
    println!("   ─────────────────────────────────────────");

    for t in [0.0, 0.25, 0.5, 0.75, 1.0] {
        if let Ok(config) = SlerpConfig::new(t) {
            if let Ok(merged) = slerp_merge(&model_task_a, &model_task_b, &config) {
                let da = model_distance(&merged, &model_task_a);
                let db = model_distance(&merged, &model_task_b);
                println!("   t={:.2}: dist_A={:.4}  dist_B={:.4}", t, da, db);
            }
        }
    }
    println!();

    // =========================================================================
    // Section 6: Save Merged Model
    // =========================================================================
    println!("6. Save Merged Model to APR v2");
    println!("   ─────────────────────────────────────────");

    if let Ok(config) = TiesConfig::new(0.2) {
        if let Ok(merged) = ties_merge(&models, &base, &config) {
            let all_bytes: Vec<u8> = merged
                .values()
                .flat_map(|t| t.data().iter().flat_map(|f| f.to_le_bytes()))
                .collect();

            let temp_dir = tempfile::tempdir().expect("temp dir");
            let path = temp_dir.path().join("merged_model.apr");

            let bundle = ModelBundleV2::new()
                .with_name("ties-merged-3task")
                .with_description("TIES merge of math+code+reasoning tasks")
                .with_compression(Compression::Lz4)
                .add_tensor("merged_weights", vec![n_params], all_bytes)
                .build();

            std::fs::write(&path, &bundle).expect("write failed");
            if let Ok(meta) = std::fs::metadata(&path) {
                println!("   Saved: {} ({} bytes)", path.display(), meta.len());
            }
        }
    }
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_layers() -> Vec<(&'static str, usize)> {
        vec![("layer1", 64), ("layer2", 128)]
    }

    #[test]
    fn test_create_base_model() {
        let model = create_base_model(&test_layers(), 42);
        assert_eq!(model.len(), 2);
        assert_eq!(model["layer1"].len(), 64);
        assert_eq!(model["layer2"].len(), 128);
    }

    #[test]
    fn test_create_finetuned_differs_from_base() {
        let base = create_base_model(&test_layers(), 42);
        let ft = create_finetuned_model(&base, 100, 0.1);
        let dist = model_distance(&base, &ft);
        assert!(dist > 0.0, "Fine-tuned model should differ from base");
    }

    #[test]
    fn test_model_distance_self_is_zero() {
        let model = create_base_model(&test_layers(), 42);
        assert!((model_distance(&model, &model)).abs() < 1e-6);
    }

    #[test]
    fn test_ties_merge() {
        let base = create_base_model(&test_layers(), 42);
        let ft1 = create_finetuned_model(&base, 100, 0.05);
        let ft2 = create_finetuned_model(&base, 200, 0.05);
        let models = vec![ft1, ft2];

        let config = TiesConfig::new(0.2).unwrap();
        let merged = ties_merge(&models, &base, &config).unwrap();
        assert_eq!(merged.len(), 2);
        assert_eq!(param_count(&merged), param_count(&base));
    }

    #[test]
    fn test_dare_merge() {
        let base = create_base_model(&test_layers(), 42);
        let ft1 = create_finetuned_model(&base, 100, 0.05);
        let ft2 = create_finetuned_model(&base, 200, 0.05);
        let models = vec![ft1, ft2];

        let config = DareConfig::new(0.5).unwrap().with_seed(42);
        let merged = dare_merge(&models, &base, &config).unwrap();
        assert_eq!(param_count(&merged), param_count(&base));
    }

    #[test]
    fn test_slerp_merge() {
        let base = create_base_model(&test_layers(), 42);
        let ft1 = create_finetuned_model(&base, 100, 0.05);
        let ft2 = create_finetuned_model(&base, 200, 0.05);

        let config = SlerpConfig::new(0.5).unwrap();
        let merged = slerp_merge(&ft1, &ft2, &config).unwrap();
        assert_eq!(param_count(&merged), param_count(&base));
    }

    #[test]
    fn test_slerp_endpoints() {
        let base = create_base_model(&test_layers(), 42);
        let ft1 = create_finetuned_model(&base, 100, 0.05);
        let ft2 = create_finetuned_model(&base, 200, 0.05);

        // t=0 should be close to ft1, t=1 close to ft2
        let config0 = SlerpConfig::new(0.0).unwrap();
        let merged0 = slerp_merge(&ft1, &ft2, &config0).unwrap();
        let d0 = model_distance(&merged0, &ft1);
        assert!(d0 < 0.01, "t=0 should be near model1, got dist={}", d0);
    }

    #[test]
    fn test_ties_invalid_density() {
        assert!(TiesConfig::new(-0.1).is_err());
        assert!(TiesConfig::new(1.5).is_err());
    }

    #[test]
    fn test_dare_produces_valid_model() {
        let base = create_base_model(&test_layers(), 42);
        let ft1 = create_finetuned_model(&base, 100, 0.05);
        let ft2 = create_finetuned_model(&base, 200, 0.05);
        let models = vec![ft1, ft2];

        let config = DareConfig::new(0.5).unwrap().with_seed(42);
        let merged = dare_merge(&models, &base, &config).unwrap();

        // Merged model should have same structure as base
        assert_eq!(merged.len(), base.len());
        for (name, tensor) in &merged {
            assert_eq!(tensor.len(), base[name].len());
        }
    }

    #[test]
    fn test_param_count() {
        let model = create_base_model(&test_layers(), 42);
        assert_eq!(param_count(&model), 64 + 128);
    }
}
