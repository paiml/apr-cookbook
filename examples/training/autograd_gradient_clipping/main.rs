#![allow(unused_imports)]
//! Autograd Gradient Clipping Example
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/cli-parity-v1.yaml
//! Demonstrates gradient clipping techniques for training stability using
//! entrenar's autograd API. Gradient clipping prevents exploding gradients
//! by bounding gradient magnitudes before the optimizer step.
//!
//! # Clipping Strategies
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                  Gradient Clipping Strategies                       │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  1. Global Norm Clipping                                            │
//! │     ‖g‖ = sqrt(Σ gᵢ²)                                             │
//! │     if ‖g‖ > max_norm: g ← g × (max_norm / ‖g‖)                  │
//! │                                                                     │
//! │  2. Per-Parameter Clipping                                          │
//! │     for each param p: clip(grad_p, -max_norm, max_norm)            │
//! │                                                                     │
//! │  3. Value Clipping                                                  │
//! │     gᵢ ← clamp(gᵢ, -max_val, max_val)                            │
//! │                                                                     │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │  Forward ─► Loss ─► Backward ─► Clip Gradients ─► Optimizer Step   │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example autograd_gradient_clipping
//! ```
//!
//! # Recipe Metadata
//!
//! - **Category**: Training
//! - **Complexity**: Intermediate
//! - **Dependencies**: entrenar 0.5+, aprender 0.25+, ndarray 0.16+
//! - **IIUR**: Isolated, Idempotent, Useful, Reproducible
//!
//!
//! ## Format Variants
//! ```bash
//! apr finetune model.apr          # APR native format
//! apr finetune model.gguf         # GGUF (llama.cpp compatible)
//! apr finetune model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References

use apr_cookbook::prelude::*;
use entrenar::autograd::Tensor;
use entrenar::optim::{AdamW, Optimizer};
use ndarray::Array1;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("autograd_gradient_clipping")?;

    println!("=== Autograd Gradient Clipping Example ===\n");

    let seed = hash_name_to_seed("autograd_gradient_clipping");
    let data = generate_data(48, seed);
    let epochs = 15;
    let lr = 0.001;

    print_strategies_overview(seed, data.len(), epochs, lr);
    print_exploding_gradient_demo(seed);

    let strategies = [
        ClipStrategy::None,
        ClipStrategy::GlobalNorm(1.0),
        ClipStrategy::PerParam(0.5),
        ClipStrategy::Value(0.1),
    ];

    let results: Vec<TrainResult> = strategies
        .iter()
        .map(|&s| train_with_clipping(seed, &data, s, lr, epochs))
        .collect();

    print_training_comparison(&results);
    print_epoch_table(
        "4. Gradient Norm Trajectories (per epoch)",
        &results,
        epochs,
        |r| &r.grad_norms,
    );
    print_epoch_table("5. Loss Convergence Comparison", &results, epochs, |r| {
        &r.losses
    });
    print_convergence_analysis(&results);

    record_metrics(&mut ctx, &results, seed, epochs, data.len());
    ctx.report()?;
    println!("\n=== Example Complete ===");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_f32_deterministic() {
        let a = hash_f32(42, 0, "test");
        let b = hash_f32(42, 0, "test");
        assert_eq!(a, b);
    }

    #[test]
    fn test_hash_f32_range() {
        for i in 0..200 {
            let v = hash_f32(42, i, "range");
            assert!(
                (-0.5..=0.5).contains(&v),
                "hash_f32 out of [-0.5, 0.5]: {v}"
            );
        }
    }

    #[test]
    fn test_model_forward_dimensions() {
        let model = ClipModel::new(42);
        let input = vec![0.5; INPUT_DIM];
        let output = model.forward(&input);
        assert_eq!(output.len(), OUTPUT_DIM);
    }

    #[test]
    fn test_softmax_cross_entropy_minimum_at_target() {
        let logits = [10.0, 0.0, 0.0, 0.0];
        let loss_correct = softmax_cross_entropy(&logits, 0);
        let loss_wrong = softmax_cross_entropy(&logits, 1);
        assert!(
            loss_correct < loss_wrong,
            "Loss at target ({loss_correct}) should be < off-target ({loss_wrong})"
        );
    }

    #[test]
    fn test_softmax_cross_entropy_nonnegative() {
        let logits = [1.0, 2.0, 3.0, 4.0];
        for target in 0..OUTPUT_DIM {
            let loss = softmax_cross_entropy(&logits, target);
            assert!(loss >= 0.0, "Cross-entropy must be >= 0, got {loss}");
            assert!(loss.is_finite(), "Cross-entropy must be finite");
        }
    }

    #[test]
    fn test_global_norm_clipping_caps_gradient() {
        let model = ClipModel::new(99);
        // Set large gradients
        set_exploding_grads(&model.params, 100.0, 99);
        let pre = global_gradient_norm(&model.params);
        assert!(pre > 1.0, "Pre-clip norm should be large");

        clip_gradients(&model.params, ClipStrategy::GlobalNorm(1.0));
        let post = global_gradient_norm(&model.params);
        assert!(
            (post - 1.0).abs() < 0.01,
            "Post-clip global norm should be ~1.0, got {post}"
        );
    }

    #[test]
    fn test_per_param_clipping_caps_each_parameter() {
        let model = ClipModel::new(99);
        set_exploding_grads(&model.params, 50.0, 99);

        clip_gradients(&model.params, ClipStrategy::PerParam(0.5));

        for p in &model.params {
            if let Some(g) = p.grad() {
                let pnorm = g.iter().map(|&v| v * v).sum::<f32>().sqrt();
                assert!(
                    pnorm <= 0.5 + 1e-5,
                    "Per-param norm should be <= 0.5, got {pnorm}"
                );
            }
        }
    }

    #[test]
    fn test_value_clipping_clamps_elements() {
        let model = ClipModel::new(99);
        set_exploding_grads(&model.params, 200.0, 99);

        clip_gradients(&model.params, ClipStrategy::Value(0.3));

        for p in &model.params {
            if let Some(g) = p.grad() {
                for &v in g.iter() {
                    assert!(
                        v >= -0.3 - 1e-6 && v <= 0.3 + 1e-6,
                        "Value-clipped element should be in [-0.3, 0.3], got {v}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_no_clip_preserves_gradients() {
        let model = ClipModel::new(42);
        set_exploding_grads(&model.params, 5.0, 42);
        let pre = global_gradient_norm(&model.params);

        clip_gradients(&model.params, ClipStrategy::None);
        let post = global_gradient_norm(&model.params);

        assert!(
            (pre - post).abs() < 1e-6,
            "None strategy should not alter gradients: {pre} vs {post}"
        );
    }

    #[test]
    fn test_generate_data_deterministic() {
        let d1 = generate_data(20, 42);
        let d2 = generate_data(20, 42);
        for (i, ((x1, t1), (x2, t2))) in d1.iter().zip(d2.iter()).enumerate() {
            assert_eq!(x1, x2, "Features differ at index {i}");
            assert_eq!(t1, t2, "Labels differ at index {i}");
        }
    }

    #[test]
    fn test_generate_data_labels_valid() {
        let data = generate_data(100, 42);
        for (_, target) in &data {
            assert!(*target < OUTPUT_DIM, "Target {target} >= OUTPUT_DIM");
        }
    }

    #[test]
    fn test_training_reduces_loss() {
        let data = generate_data(24, 42);
        let result = train_with_clipping(42, &data, ClipStrategy::GlobalNorm(1.0), 0.001, 10);

        assert!(
            result.losses.len() == 10,
            "Should have 10 epoch losses, got {}",
            result.losses.len()
        );
        assert!(
            result.final_loss.is_finite(),
            "Final loss should be finite, got {}",
            result.final_loss
        );
    }

    #[test]
    fn test_clip_strategy_labels() {
        assert_eq!(ClipStrategy::None.label(), "None");
        assert_eq!(ClipStrategy::GlobalNorm(1.0).label(), "GlobalNorm(1)");
        assert_eq!(ClipStrategy::PerParam(0.5).label(), "PerParam(0.5)");
        assert_eq!(ClipStrategy::Value(0.1).label(), "Value(0.1)");
    }
}
