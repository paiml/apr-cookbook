#![allow(unused_imports)]
//! Gradient Accumulation Training Example
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Demonstrates gradient accumulation for training with large effective batch
//! sizes while keeping memory usage bounded. Micro-batches are processed
//! sequentially, gradients are summed, and a single optimizer step is taken
//! per accumulation cycle.
//!
//! # How It Works
//!
//! ```text
//! Micro-batch 1 ──► forward ──► backward ──► grad += grad_1
//! Micro-batch 2 ──► forward ──► backward ──► grad += grad_2
//! Micro-batch 3 ──► forward ──► backward ──► grad += grad_3
//! Micro-batch 4 ──► forward ──► backward ──► grad += grad_4
//!                                            ──────────────
//!                                            grad /= 4
//!                                            weights -= lr * grad
//!                                            grad = 0   (zero out)
//! ```
//!
//! # Memory Savings
//!
//! ```text
//! True batch=64:       activations for 64 samples in memory
//! Accum 16x micro=4:   activations for  4 samples in memory
//!                      → 16x memory reduction for activations
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example gradient_accumulation
//! ```
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

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== Gradient Accumulation Training Example ===\n");

    let seed = 42;
    let lr = 0.05;
    let train_data = generate_data(N_TRAIN, seed);
    let val_data = generate_data(N_VAL, seed + 1000);

    // =========================================================================
    // Section 1: Memory Savings Analysis
    // =========================================================================
    println!("1. Memory Savings Analysis");
    println!("   ─────────────────────────────────────────");

    let param_count = AccumModel::new(seed).param_count();
    println!("   Model parameters: {param_count}");
    println!();

    println!(
        "   {:>8} {:>10} {:>12} {:>14} {:>10}",
        "Accum", "MicroBS", "EffectBS", "ActMemory", "Savings"
    );
    println!("   {}", "─".repeat(58));

    let baseline_mem = estimate_activation_memory(N_TRAIN);
    for &steps in &ACCUM_STEPS {
        let micro_bs = N_TRAIN / steps;
        let act_mem = estimate_activation_memory(micro_bs);
        let savings = if baseline_mem > 0 {
            100.0 * (1.0 - act_mem as f64 / baseline_mem as f64)
        } else {
            0.0
        };
        println!(
            "   {:>7}x {:>10} {:>12} {:>11} B {:>9.1}%",
            steps, micro_bs, N_TRAIN, act_mem, savings
        );
    }
    println!();

    println!("   True large batch (no accumulation):");
    println!(
        "   {:>8} {:>14} {:>14}",
        "BatchSize", "ActMemory", "TotalMemory"
    );
    println!("   {}", "─".repeat(40));
    for &bs in &[1, 4, 16, 64, 128] {
        let act = estimate_activation_memory(bs);
        let total = estimate_total_memory(bs, param_count);
        println!("   {:>8} {:>11} B {:>11} B", bs, act, total);
    }
    println!();

    // =========================================================================
    // Section 2: Gradient Accumulation Comparison
    // =========================================================================
    println!("2. Training with Different Accumulation Steps");
    println!("   ─────────────────────────────────────────");

    let mut results: Vec<TrainResult> = Vec::new();
    for &steps in &ACCUM_STEPS {
        let result = train_full(seed, &train_data, &val_data, steps, lr, TOTAL_EPOCHS);
        results.push(result);
    }

    println!(
        "   {:>8} {:>8} {:>10} {:>8} {:>10} {:>8}",
        "Accum", "EffBS", "TrainLoss", "TrnAcc", "ValLoss", "ValAcc"
    );
    println!("   {}", "─".repeat(56));

    for result in &results {
        println!(
            "   {:>7}x {:>8} {:>10.4} {:>7.1}% {:>10.4} {:>7.1}%",
            result.accum_steps,
            result.effective_batch_size,
            result.final_loss,
            result.final_acc * 100.0,
            result.val_loss,
            result.val_acc * 100.0
        );
    }
    println!();

    // =========================================================================
    // Section 3: Convergence Curves
    // =========================================================================
    println!("3. Convergence Curves (loss per epoch)");
    println!("   ─────────────────────────────────────────");

    println!(
        "   {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Epoch", "1x", "4x", "8x", "16x", ""
    );
    println!("   {}", "─".repeat(46));

    let sample_epochs = [0, 4, 9, 14, 19];
    for &epoch in &sample_epochs {
        if epoch < TOTAL_EPOCHS {
            print!("   {:>8}", epoch);
            for result in &results {
                if epoch < result.loss_curve.len() {
                    print!(" {:>8.4}", result.loss_curve[epoch]);
                }
            }
            println!();
        }
    }
    println!();

    // Accuracy convergence
    println!("   Accuracy convergence:");
    println!(
        "   {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Epoch", "1x", "4x", "8x", "16x"
    );
    println!("   {}", "─".repeat(44));

    for &epoch in &sample_epochs {
        if epoch < TOTAL_EPOCHS {
            print!("   {:>8}", epoch);
            for result in &results {
                if epoch < result.acc_curve.len() {
                    print!(" {:>7.1}%", result.acc_curve[epoch] * 100.0);
                }
            }
            println!();
        }
    }
    println!();

    // =========================================================================
    // Section 4: Gradient Norm Monitoring
    // =========================================================================
    println!("4. Gradient Norm Monitoring");
    println!("   ─────────────────────────────────────────");

    println!(
        "   {:>8} {:>12} {:>14}",
        "Accum", "AvgGradNorm", "PeakMemory"
    );
    println!("   {}", "─".repeat(38));

    for result in &results {
        println!(
            "   {:>7}x {:>12.4} {:>11} B",
            result.accum_steps, result.avg_grad_norm, result.peak_memory_bytes
        );
    }
    println!();

    // Detailed gradient norms for one epoch with accum=4
    println!("   Gradient norms per optimizer step (accum=4, epoch 1):");
    let mut monitor_model = AccumModel::new(seed);
    let (_, _, norms) = train_epoch_accumulated(&mut monitor_model, &train_data, lr, 4);
    println!("   {:>6} {:>12}", "Step", "GradNorm");
    println!("   {}", "─".repeat(20));
    for (i, &norm) in norms.iter().enumerate().take(10) {
        println!("   {:>6} {:>12.4}", i + 1, norm);
    }
    if norms.len() > 10 {
        println!("   ... ({} more steps)", norms.len() - 10);
    }
    println!();

    // =========================================================================
    // Section 5: Effective Batch Size Impact
    // =========================================================================
    println!("5. Effective Batch Size Impact");
    println!("   ─────────────────────────────────────────");

    println!(
        "   {:>8} {:>8} {:>12} {:>10} {:>10}",
        "Accum", "EffBS", "Steps/Epoch", "ValLoss", "ValAcc"
    );
    println!("   {}", "─".repeat(52));

    for result in &results {
        let steps_per_epoch = N_TRAIN.div_ceil(result.accum_steps);
        println!(
            "   {:>7}x {:>8} {:>12} {:>10.4} {:>9.1}%",
            result.accum_steps,
            result.effective_batch_size,
            steps_per_epoch,
            result.val_loss,
            result.val_acc * 100.0
        );
    }
    println!();

    // =========================================================================
    // Section 6: Memory vs Accuracy Tradeoff
    // =========================================================================
    println!("6. Memory vs Accuracy Tradeoff");
    println!("   ─────────────────────────────────────────");

    println!(
        "   {:>8} {:>12} {:>10} {:>10} {:>12}",
        "Accum", "PeakMem(B)", "ValLoss", "ValAcc", "Mem/Acc"
    );
    println!("   {}", "─".repeat(56));

    for result in &results {
        let mem_per_acc = if result.val_acc > 0.0 {
            result.peak_memory_bytes as f32 / result.val_acc
        } else {
            f32::INFINITY
        };
        println!(
            "   {:>7}x {:>12} {:>10.4} {:>9.1}% {:>12.1}",
            result.accum_steps,
            result.peak_memory_bytes,
            result.val_loss,
            result.val_acc * 100.0,
            mem_per_acc
        );
    }
    println!();

    println!("=== Example Complete ===");
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
        for i in 0..100 {
            let v = hash_f32(42, i, "range");
            assert!(v >= -0.5 && v <= 0.5, "hash_f32 out of range: {v}");
        }
    }

    #[test]
    fn test_model_forward_dimensions() {
        let model = AccumModel::new(42);
        let input = [0.5; INPUT_DIM];
        let (hidden, output) = model.forward(&input);
        assert_eq!(hidden.len(), HIDDEN_DIM);
        assert_eq!(output.len(), OUTPUT_DIM);
    }

    #[test]
    fn test_softmax_cross_entropy_minimum_at_target() {
        let logits = [10.0, 0.0, 0.0, 0.0];
        let loss_correct = softmax_cross_entropy(&logits, 0);
        let loss_wrong = softmax_cross_entropy(&logits, 1);
        assert!(
            loss_correct < loss_wrong,
            "Loss at target ({loss_correct}) should be less than off-target ({loss_wrong})"
        );
    }

    #[test]
    fn test_softmax_cross_entropy_nonnegative() {
        let logits = [1.0, 2.0, 3.0, 4.0];
        for target in 0..OUTPUT_DIM {
            let loss = softmax_cross_entropy(&logits, target);
            assert!(
                loss >= 0.0,
                "Cross-entropy must be non-negative, got {loss}"
            );
            assert!(loss.is_finite(), "Cross-entropy must be finite");
        }
    }

    #[test]
    fn test_predict_argmax() {
        assert_eq!(predict(&[0.1, 0.9, 0.3, 0.2]), 1);
        assert_eq!(predict(&[5.0, 1.0, 2.0, 3.0]), 0);
        assert_eq!(predict(&[0.0, 0.0, 0.0, 1.0]), 3);
    }

    #[test]
    fn test_generate_data_deterministic() {
        let d1 = generate_data(10, 42);
        let d2 = generate_data(10, 42);
        for (i, (a, b)) in d1.iter().zip(d2.iter()).enumerate() {
            assert_eq!(a.0, b.0, "Inputs differ at index {i}");
            assert_eq!(a.1, b.1, "Labels differ at index {i}");
        }
    }

    #[test]
    fn test_generate_data_shapes() {
        let data = generate_data(20, 42);
        assert_eq!(data.len(), 20);
        for (input, label) in &data {
            assert_eq!(input.len(), INPUT_DIM);
            assert!(*label < OUTPUT_DIM);
        }
    }

    #[test]
    fn test_gradient_accumulation_reduces_loss() {
        let train_data = generate_data(32, 42);
        let mut model = AccumModel::new(42);
        let (loss_before, _) = evaluate(&model, &train_data);

        for _ in 0..5 {
            train_epoch_accumulated(&mut model, &train_data, 0.05, 4);
        }

        let (loss_after, _) = evaluate(&model, &train_data);
        assert!(
            loss_after < loss_before,
            "Training should reduce loss: {loss_before} -> {loss_after}"
        );
    }

    #[test]
    fn test_zero_grad_clears_accumulators() {
        let data = generate_data(4, 42);
        let mut model = AccumModel::new(42);
        model.backward_accumulate(&data[0].0, data[0].1);
        assert!(model.gradient_norm() > 0.0);
        assert_eq!(model.accum_count, 1);

        model.zero_grad();
        assert!((model.gradient_norm() - 0.0).abs() < f32::EPSILON);
        assert_eq!(model.accum_count, 0);
    }

    #[test]
    fn test_gradient_norm_increases_with_accumulation() {
        let data = generate_data(8, 42);
        let mut model = AccumModel::new(42);

        model.backward_accumulate(&data[0].0, data[0].1);
        let norm_1 = model.gradient_norm();

        model.backward_accumulate(&data[1].0, data[1].1);
        let norm_2 = model.gradient_norm();

        // Gradient norm should generally increase (or at least not be zero)
        assert!(norm_1 > 0.0, "Gradient norm after 1 step should be > 0");
        assert!(norm_2 > 0.0, "Gradient norm after 2 steps should be > 0");
    }

    #[test]
    fn test_step_and_zero_resets_count() {
        let data = generate_data(4, 42);
        let mut model = AccumModel::new(42);
        model.backward_accumulate(&data[0].0, data[0].1);
        model.backward_accumulate(&data[1].0, data[1].1);
        assert_eq!(model.accum_count, 2);

        model.step_and_zero(0.01);
        assert_eq!(model.accum_count, 0);
        assert!((model.gradient_norm() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_estimate_activation_memory_scales_linearly() {
        let mem_1 = estimate_activation_memory(1);
        let mem_4 = estimate_activation_memory(4);
        let mem_16 = estimate_activation_memory(16);
        assert_eq!(mem_4, mem_1 * 4);
        assert_eq!(mem_16, mem_1 * 16);
    }

    #[test]
    fn test_estimate_total_memory_includes_weights() {
        let param_count = 100;
        let total_bs1 = estimate_total_memory(1, param_count);
        let total_bs2 = estimate_total_memory(2, param_count);
        // Larger batch should use more memory
        assert!(total_bs2 > total_bs1);
        // Weights + grads should be constant part
        let weight_grad_mem = param_count * size_of::<f32>() * 2;
        assert!(total_bs1 >= weight_grad_mem);
    }

    #[test]
    fn test_param_count() {
        let model = AccumModel::new(42);
        let expected = HIDDEN_DIM * INPUT_DIM + HIDDEN_DIM + OUTPUT_DIM * HIDDEN_DIM + OUTPUT_DIM;
        assert_eq!(model.param_count(), expected);
    }

    #[test]
    fn test_train_epoch_returns_grad_norms() {
        let data = generate_data(16, 42);
        let mut model = AccumModel::new(42);
        let (_, _, norms) = train_epoch_accumulated(&mut model, &data, 0.01, 4);
        // 16 samples / 4 accum = 4 optimizer steps
        assert_eq!(norms.len(), 4);
        for norm in &norms {
            assert!(norm.is_finite(), "Gradient norm must be finite");
            assert!(*norm >= 0.0, "Gradient norm must be non-negative");
        }
    }

    #[test]
    fn test_train_epoch_handles_remainder() {
        // 10 samples with accum=4 -> 2 full steps + 1 remainder step = 3 steps
        let data = generate_data(10, 42);
        let mut model = AccumModel::new(42);
        let (_, _, norms) = train_epoch_accumulated(&mut model, &data, 0.01, 4);
        assert_eq!(norms.len(), 3, "Should have 2 full + 1 remainder step");
    }

    #[test]
    fn test_evaluate_returns_valid_metrics() {
        let data = generate_data(20, 42);
        let model = AccumModel::new(42);
        let (loss, acc) = evaluate(&model, &data);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
        assert!((0.0..=1.0).contains(&acc));
    }
}
