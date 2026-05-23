//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports, clippy::wildcard_imports)]
use super::types::*;

use apr_cookbook::prelude::*;
use entrenar::autograd::Tensor;
use entrenar::optim::{AdamW, Optimizer};
use ndarray::Array1;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Generate synthetic classification data with class-conditional features.
pub fn generate_data(n: usize, seed: u64) -> Vec<(Vec<f32>, usize)> {
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
pub fn set_exploding_grads(params: &[Tensor], scale: f32, seed: u64) {
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
pub struct TrainResult {
    pub strategy: String,
    pub losses: Vec<f32>,
    pub grad_norms: Vec<f32>,
    pub clipped_norms: Vec<f32>,
    pub final_loss: f32,
    pub accuracy: f32,
}

/// Train a model with a given clipping strategy.
pub fn train_with_clipping(
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
pub fn print_strategies_overview(seed: u64, data_len: usize, epochs: usize, lr: f32) {
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
pub fn print_exploding_gradient_demo(seed: u64) {
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
pub fn print_training_comparison(results: &[TrainResult]) {
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
pub fn print_epoch_table(
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
pub fn print_convergence_analysis(results: &[TrainResult]) {
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
pub fn record_metrics(
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
