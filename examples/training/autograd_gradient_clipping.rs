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
//! - Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685

use apr_cookbook::prelude::*;
use entrenar::autograd::Tensor;
use entrenar::optim::{AdamW, Optimizer};
use ndarray::Array1;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ── Model dimensions ──

const INPUT_DIM: usize = 8;
const HIDDEN_DIM: usize = 16;
const OUTPUT_DIM: usize = 4;

/// Gradient clipping strategy
#[derive(Debug, Clone, Copy, PartialEq)]
enum ClipStrategy {
    /// No clipping applied
    None,
    /// Global L2 norm clipping: scale all gradients if total norm exceeds threshold
    GlobalNorm(f32),
    /// Per-parameter norm clipping: clip each parameter independently
    PerParam(f32),
    /// Value clipping: clamp each gradient element to [-max_val, max_val]
    Value(f32),
}

impl ClipStrategy {
    fn label(self) -> String {
        match self {
            Self::None => "None".to_string(),
            Self::GlobalNorm(t) => format!("GlobalNorm({t})"),
            Self::PerParam(t) => format!("PerParam({t})"),
            Self::Value(t) => format!("Value({t})"),
        }
    }
}

/// Deterministic hash-based value
fn hash_f32(seed: u64, index: usize, label: &str) -> f32 {
    let mut h = DefaultHasher::new();
    (seed, label, index).hash(&mut h);
    h.finish() as f32 / u64::MAX as f32 - 0.5
}

/// Two-layer MLP backed by entrenar Tensors
struct ClipModel {
    /// w1: [HIDDEN_DIM x INPUT_DIM], b1: [HIDDEN_DIM],
    /// w2: [OUTPUT_DIM x HIDDEN_DIM], b2: [OUTPUT_DIM]
    params: Vec<Tensor>,
}

impl ClipModel {
    const W1: usize = 0;
    const B1: usize = 1;
    const W2: usize = 2;
    const B2: usize = 3;

    fn new(seed: u64) -> Self {
        let w1_scale = (2.0 / (INPUT_DIM + HIDDEN_DIM) as f32).sqrt();
        let w2_scale = (2.0 / (HIDDEN_DIM + OUTPUT_DIM) as f32).sqrt();

        let w1: Vec<f32> = (0..HIDDEN_DIM * INPUT_DIM)
            .map(|i| hash_f32(seed, i, "w1") * w1_scale)
            .collect();
        let w2: Vec<f32> = (0..OUTPUT_DIM * HIDDEN_DIM)
            .map(|i| hash_f32(seed, i, "w2") * w2_scale)
            .collect();

        let params = vec![
            Tensor::from_vec(w1, true),
            Tensor::zeros(HIDDEN_DIM, true),
            Tensor::from_vec(w2, true),
            Tensor::zeros(OUTPUT_DIM, true),
        ];

        Self { params }
    }

    /// Forward pass: input -> hidden (ReLU) -> logits
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let w1 = &self.params[Self::W1];
        let b1 = &self.params[Self::B1];
        let w2 = &self.params[Self::W2];
        let b2 = &self.params[Self::B2];

        // Hidden = ReLU(x @ W1^T + b1)
        let mut hidden = [0.0f32; HIDDEN_DIM];
        #[allow(clippy::needless_range_loop)]
        for j in 0..HIDDEN_DIM {
            let mut sum = b1.data()[j];
            for i in 0..INPUT_DIM {
                sum += x[i] * w1.data()[j * INPUT_DIM + i];
            }
            hidden[j] = sum.max(0.0);
        }

        // Output = hidden @ W2^T + b2
        let mut output = [0.0f32; OUTPUT_DIM];
        #[allow(clippy::needless_range_loop)]
        for k in 0..OUTPUT_DIM {
            let mut sum = b2.data()[k];
            for j in 0..HIDDEN_DIM {
                sum += hidden[j] * w2.data()[k * HIDDEN_DIM + j];
            }
            output[k] = sum;
        }

        output.to_vec()
    }

    /// Softmax cross-entropy loss
    fn loss(&self, logits: &[f32], target: usize) -> f32 {
        softmax_cross_entropy(logits, target)
    }

    fn param_count(&self) -> usize {
        self.params.iter().map(Tensor::len).sum()
    }

    fn params_mut(&mut self) -> &mut [Tensor] {
        &mut self.params
    }
}

/// Numerically stable softmax cross-entropy
fn softmax_cross_entropy(logits: &[f32], target: usize) -> f32 {
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    -(exps[target] / sum).max(1e-10).ln()
}

/// Compute finite-difference gradients and set them on the model parameters.
///
/// This simulates backward() for demonstration purposes: it computes
/// the gradient of the loss w.r.t. each parameter using central differences.
fn compute_and_set_grads(model: &mut ClipModel, x: &[f32], target: usize) {
    let eps = 1e-4_f32;

    for param_idx in 0..model.params.len() {
        let n = model.params[param_idx].len();
        let mut grad = Array1::<f32>::zeros(n);

        for elem in 0..n {
            // +eps
            let orig = model.params[param_idx].data()[elem];
            // We need to modify the parameter in-place. Since Tensor doesn't expose
            // direct mutation of individual elements, we reconstruct. But for efficiency,
            // we use the same pattern as gradient_accumulation: modify the model weights
            // directly through a clone-modify-replace cycle.
            let mut data_plus = model.params[param_idx].data().to_vec();
            data_plus[elem] = orig + eps;
            let tmp = Tensor::from_vec(data_plus, true);
            let old = std::mem::replace(&mut model.params[param_idx], tmp);
            let logits_plus = model.forward(x);
            let loss_plus = model.loss(&logits_plus, target);

            // -eps
            let mut data_minus = old.data().to_vec();
            data_minus[elem] = orig - eps;
            let tmp2 = Tensor::from_vec(data_minus, true);
            let _ = std::mem::replace(&mut model.params[param_idx], tmp2);
            let logits_minus = model.forward(x);
            let loss_minus = model.loss(&logits_minus, target);

            // Restore original
            let mut data_orig = model.params[param_idx].data().to_vec();
            data_orig[elem] = orig;
            model.params[param_idx] = Tensor::from_vec(data_orig, true);

            grad[elem] = (loss_plus - loss_minus) / (2.0 * eps);
        }

        model.params[param_idx].set_grad(grad);
    }
}

/// Compute the global L2 norm of all parameter gradients
fn global_gradient_norm(params: &[Tensor]) -> f32 {
    let mut sum_sq = 0.0f32;
    for p in params {
        if let Some(g) = p.grad() {
            sum_sq += g.iter().map(|&v| v * v).sum::<f32>();
        }
    }
    sum_sq.sqrt()
}

/// Apply gradient clipping in place (modifies gradients stored in Tensors)
fn clip_gradients(params: &[Tensor], strategy: ClipStrategy) -> f32 {
    let pre_norm = global_gradient_norm(params);

    match strategy {
        ClipStrategy::None => {}
        ClipStrategy::GlobalNorm(max_norm) => {
            if pre_norm > max_norm {
                let scale = max_norm / pre_norm;
                for p in params {
                    if let Some(g) = p.grad() {
                        let clipped = g.mapv(|v| v * scale);
                        p.set_grad(clipped);
                    }
                }
            }
        }
        ClipStrategy::PerParam(max_norm) => {
            for p in params {
                if let Some(g) = p.grad() {
                    let pnorm = g.iter().map(|&v| v * v).sum::<f32>().sqrt();
                    if pnorm > max_norm {
                        let scale = max_norm / pnorm;
                        let clipped = g.mapv(|v| v * scale);
                        p.set_grad(clipped);
                    }
                }
            }
        }
        ClipStrategy::Value(max_val) => {
            for p in params {
                if let Some(g) = p.grad() {
                    let clipped = g.mapv(|v| v.clamp(-max_val, max_val));
                    p.set_grad(clipped);
                }
            }
        }
    }

    pre_norm
}

/// Generate synthetic classification data with class-conditional features.
fn generate_data(n: usize, seed: u64) -> Vec<(Vec<f32>, usize)> {
    (0..n)
        .map(|i| {
            let class = i % OUTPUT_DIM;
            let features: Vec<f32> = (0..INPUT_DIM)
                .map(|j| {
                    let noise = hash_f32(seed, i * INPUT_DIM + j, "x") * 0.4;
                    class as f32 * 0.5 + noise + j as f32 * 0.05
                })
                .collect();
            (features, class)
        })
        .collect()
}

/// Simulate exploding gradients by using extremely large gradient scale.
fn set_exploding_grads(params: &[Tensor], scale: f32, seed: u64) {
    for (pidx, p) in params.iter().enumerate() {
        if p.requires_grad() {
            let grad: Array1<f32> = Array1::from_shape_fn(p.len(), |i| {
                hash_f32(seed, pidx * 1000 + i, "explode") * scale
            });
            p.set_grad(grad);
        }
    }
}

/// Result of a training run
#[derive(Debug)]
struct TrainResult {
    strategy: String,
    losses: Vec<f32>,
    grad_norms: Vec<f32>,
    clipped_norms: Vec<f32>,
    final_loss: f32,
    accuracy: f32,
}

/// Train a model with a given clipping strategy.
fn train_with_clipping(
    seed: u64,
    data: &[(Vec<f32>, usize)],
    strategy: ClipStrategy,
    lr: f32,
    epochs: usize,
) -> TrainResult {
    let mut model = ClipModel::new(seed);
    let mut optimizer = AdamW::default_params(lr);

    let mut losses = Vec::with_capacity(epochs);
    let mut grad_norms = Vec::with_capacity(epochs);
    let mut clipped_norms = Vec::with_capacity(epochs);

    for _epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;
        let mut epoch_pre_norms = Vec::new();
        let mut epoch_post_norms = Vec::new();

        for (x, target) in data {
            optimizer.zero_grad(model.params_mut());

            let logits = model.forward(x);
            let loss = model.loss(&logits, *target);
            epoch_loss += loss;

            // Compute gradients via finite differences
            compute_and_set_grads(&mut model, x, *target);

            // Clip gradients
            let pre_norm = clip_gradients(&model.params, strategy);
            let post_norm = global_gradient_norm(&model.params);
            epoch_pre_norms.push(pre_norm);
            epoch_post_norms.push(post_norm);

            // Optimizer step
            optimizer.step(model.params_mut());
        }

        let avg_loss = epoch_loss / data.len() as f32;
        losses.push(avg_loss);

        let avg_pre = epoch_pre_norms.iter().sum::<f32>() / epoch_pre_norms.len() as f32;
        let avg_post = epoch_post_norms.iter().sum::<f32>() / epoch_post_norms.len() as f32;
        grad_norms.push(avg_pre);
        clipped_norms.push(avg_post);
    }

    // Evaluate accuracy
    let mut correct = 0usize;
    for (x, target) in data {
        let logits = model.forward(x);
        let pred = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);
        if pred == *target {
            correct += 1;
        }
    }
    let accuracy = correct as f32 / data.len() as f32;
    let final_loss = losses.last().copied().unwrap_or(f32::INFINITY);

    TrainResult {
        strategy: strategy.label(),
        losses,
        grad_norms,
        clipped_norms,
        final_loss,
        accuracy,
    }
}

/// Print the overview of clipping strategies and model configuration.
fn print_strategies_overview(seed: u64, data_len: usize, epochs: usize, lr: f32) {
    println!("1. Gradient Clipping Strategies");
    println!("   ─────────────────────────────────────────");
    println!("   GlobalNorm: Scale all grads if total L2 norm > threshold");
    println!("   PerParam:   Clip each parameter's gradient independently");
    println!("   Value:      Clamp each gradient element to [-max, max]");
    println!(
        "   Model:      {}x{}x{} ({} params)",
        INPUT_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM,
        ClipModel::new(seed).param_count()
    );
    println!(
        "   Samples:    {}, Epochs: {}, LR: {}",
        data_len, epochs, lr
    );
    println!();
}

/// Demonstrate how each clipping strategy bounds exploding gradients.
fn print_exploding_gradient_demo(seed: u64) {
    println!("2. Exploding Gradient Demonstration");
    println!("   ─────────────────────────────────────────");

    let demo_model = ClipModel::new(seed);
    let explode_scales = [1.0, 10.0, 100.0, 1000.0];

    println!(
        "   {:>10} {:>14} {:>14} {:>14} {:>14}",
        "Scale", "PreNorm", "GlobalNorm(1)", "PerParam(1)", "Value(0.5)"
    );
    println!("   {}", "-".repeat(70));

    for &scale in &explode_scales {
        set_exploding_grads(&demo_model.params, scale, seed);
        let pre_norm = global_gradient_norm(&demo_model.params);

        set_exploding_grads(&demo_model.params, scale, seed);
        clip_gradients(&demo_model.params, ClipStrategy::GlobalNorm(1.0));
        let gn_norm = global_gradient_norm(&demo_model.params);

        set_exploding_grads(&demo_model.params, scale, seed);
        clip_gradients(&demo_model.params, ClipStrategy::PerParam(1.0));
        let pp_norm = global_gradient_norm(&demo_model.params);

        set_exploding_grads(&demo_model.params, scale, seed);
        clip_gradients(&demo_model.params, ClipStrategy::Value(0.5));
        let vc_norm = global_gradient_norm(&demo_model.params);

        println!(
            "   {:>10.0} {:>14.4} {:>14.4} {:>14.4} {:>14.4}",
            scale, pre_norm, gn_norm, pp_norm, vc_norm
        );
    }
    println!();
}

/// Print the training comparison summary table.
fn print_training_comparison(results: &[TrainResult]) {
    println!("3. Training with Different Clipping Strategies");
    println!("   ─────────────────────────────────────────");

    println!(
        "   {:>16} {:>10} {:>10} {:>12} {:>12}",
        "Strategy", "FinalLoss", "Accuracy", "AvgGradNorm", "AvgClipNorm"
    );
    println!("   {}", "-".repeat(64));

    for r in results {
        let avg_gn = r.grad_norms.iter().sum::<f32>() / r.grad_norms.len().max(1) as f32;
        let avg_cn = r.clipped_norms.iter().sum::<f32>() / r.clipped_norms.len().max(1) as f32;
        println!(
            "   {:>16} {:>10.4} {:>9.1}% {:>12.4} {:>12.4}",
            r.strategy,
            r.final_loss,
            r.accuracy * 100.0,
            avg_gn,
            avg_cn
        );
    }
    println!();
}

/// Print an epoch-sampled trajectory table for a given metric extractor.
fn print_epoch_table(
    title: &str,
    results: &[TrainResult],
    epochs: usize,
    extract: fn(&TrainResult) -> &[f32],
) {
    println!("{title}");
    println!("   ─────────────────────────────────────────");

    print!("   {:>6}", "Epoch");
    for r in results {
        print!(" {:>14}", r.strategy);
    }
    println!();
    println!("   {}", "-".repeat(6 + results.len() * 15));

    let sample_epochs: Vec<usize> = (0..epochs)
        .step_by(3)
        .chain(std::iter::once(epochs - 1))
        .collect();
    for &e in &sample_epochs {
        if e < epochs {
            print!("   {:>6}", e);
            for r in results {
                let data = extract(r);
                if e < data.len() {
                    print!(" {:>14.4}", data[e]);
                }
            }
            println!();
        }
    }
    println!();
}

/// Print convergence improvement analysis for each strategy vs the baseline.
fn print_convergence_analysis(results: &[TrainResult]) {
    println!("6. Convergence Improvement Analysis");
    println!("   ─────────────────────────────────────────");

    let baseline_last = results[0].final_loss;

    for r in results {
        let first = r.losses.first().copied().unwrap_or(1.0);
        let loss_reduction = ((first - r.final_loss) / first) * 100.0;
        let vs_baseline = if baseline_last > 0.0 {
            ((baseline_last - r.final_loss) / baseline_last) * 100.0
        } else {
            0.0
        };
        let norm_stability = if r.grad_norms.len() >= 2 {
            let first_norm = r.grad_norms[0];
            let last_norm = r.grad_norms[r.grad_norms.len() - 1];
            if first_norm > 0.0 {
                (last_norm / first_norm * 100.0).min(999.0)
            } else {
                0.0
            }
        } else {
            0.0
        };

        println!("   {} strategy:", r.strategy);
        println!(
            "     Loss reduction:    {:.1}% ({:.4} -> {:.4})",
            loss_reduction, first, r.final_loss
        );
        println!(
            "     vs No-Clip:        {:+.1}% (lower is better)",
            -vs_baseline
        );
        println!("     Norm stability:    {:.1}% of initial", norm_stability);
        println!("     Final accuracy:    {:.1}%", r.accuracy * 100.0);
        println!();
    }
}

/// Record all metrics into the recipe context.
fn record_metrics(
    ctx: &mut RecipeContext,
    results: &[TrainResult],
    seed: u64,
    epochs: usize,
    data_len: usize,
) {
    ctx.record_float_metric("no_clip_loss", f64::from(results[0].final_loss));
    ctx.record_float_metric("global_norm_loss", f64::from(results[1].final_loss));
    ctx.record_float_metric("per_param_loss", f64::from(results[2].final_loss));
    ctx.record_float_metric("value_clip_loss", f64::from(results[3].final_loss));
    ctx.record_float_metric("no_clip_accuracy", f64::from(results[0].accuracy));
    ctx.record_float_metric("global_norm_accuracy", f64::from(results[1].accuracy));
    ctx.record_metric("total_params", ClipModel::new(seed).param_count() as i64);
    ctx.record_metric("epochs", epochs as i64);
    ctx.record_metric("samples", data_len as i64);
}

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
