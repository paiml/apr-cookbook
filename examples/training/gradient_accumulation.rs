//! Gradient Accumulation Training Example
//!
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

const INPUT_DIM: usize = 16;
const HIDDEN_DIM: usize = 8;
const OUTPUT_DIM: usize = 4;
const N_TRAIN: usize = 128;
const N_VAL: usize = 32;
const TOTAL_EPOCHS: usize = 20;

/// Accumulation steps to compare
const ACCUM_STEPS: [usize; 4] = [1, 4, 8, 16];

/// Deterministic hash-based value generation
fn hash_f32(seed: u64, index: usize, label: &str) -> f32 {
    let mut h = DefaultHasher::new();
    (seed, label, index).hash(&mut h);
    h.finish() as f32 / u64::MAX as f32 - 0.5
}

/// Two-layer model with gradient accumulation support
struct AccumModel {
    w1: [f32; HIDDEN_DIM * INPUT_DIM],
    b1: [f32; HIDDEN_DIM],
    w2: [f32; OUTPUT_DIM * HIDDEN_DIM],
    b2: [f32; OUTPUT_DIM],
    // Gradient accumulators
    grad_w1: [f32; HIDDEN_DIM * INPUT_DIM],
    grad_b1: [f32; HIDDEN_DIM],
    grad_w2: [f32; OUTPUT_DIM * HIDDEN_DIM],
    grad_b2: [f32; OUTPUT_DIM],
    accum_count: usize,
}

impl AccumModel {
    fn new(seed: u64) -> Self {
        let mut w1 = [0.0f32; HIDDEN_DIM * INPUT_DIM];
        let mut w2 = [0.0f32; OUTPUT_DIM * HIDDEN_DIM];

        for (i, w) in w1.iter_mut().enumerate() {
            *w = hash_f32(seed, i, "w1") * 0.2;
        }
        for (i, w) in w2.iter_mut().enumerate() {
            *w = hash_f32(seed, i, "w2") * 0.2;
        }

        Self {
            w1,
            b1: [0.0; HIDDEN_DIM],
            w2,
            b2: [0.0; OUTPUT_DIM],
            grad_w1: [0.0; HIDDEN_DIM * INPUT_DIM],
            grad_b1: [0.0; HIDDEN_DIM],
            grad_w2: [0.0; OUTPUT_DIM * HIDDEN_DIM],
            grad_b2: [0.0; OUTPUT_DIM],
            accum_count: 0,
        }
    }

    /// Forward pass: input -> hidden (ReLU) -> output
    fn forward(&self, input: &[f32]) -> ([f32; HIDDEN_DIM], [f32; OUTPUT_DIM]) {
        let mut hidden = self.b1;
        for (o, h) in hidden.iter_mut().enumerate() {
            for (i, &x) in input.iter().enumerate() {
                *h += self.w1[o * INPUT_DIM + i] * x;
            }
            *h = h.max(0.0); // ReLU
        }

        let mut output = self.b2;
        for (o, out) in output.iter_mut().enumerate() {
            for (i, &h) in hidden.iter().enumerate() {
                *out += self.w2[o * HIDDEN_DIM + i] * h;
            }
        }

        (hidden, output)
    }

    /// Compute softmax cross-entropy loss
    fn loss(&self, input: &[f32], target: usize) -> f32 {
        let (_, output) = self.forward(input);
        softmax_cross_entropy(&output, target)
    }

    /// Simulate backward pass: accumulate gradients via finite differences
    fn backward_accumulate(&mut self, input: &[f32], target: usize) {
        let eps = 1e-4;
        let base_loss = self.loss(input, target);

        // Gradients for w2
        for idx in 0..self.w2.len() {
            let orig = self.w2[idx];
            self.w2[idx] = orig + eps;
            let loss_plus = self.loss(input, target);
            self.w2[idx] = orig;
            self.grad_w2[idx] += (loss_plus - base_loss) / eps;
        }

        // Gradients for b2
        for idx in 0..self.b2.len() {
            let orig = self.b2[idx];
            self.b2[idx] = orig + eps;
            let loss_plus = self.loss(input, target);
            self.b2[idx] = orig;
            self.grad_b2[idx] += (loss_plus - base_loss) / eps;
        }

        // Gradients for w1
        for idx in 0..self.w1.len() {
            let orig = self.w1[idx];
            self.w1[idx] = orig + eps;
            let loss_plus = self.loss(input, target);
            self.w1[idx] = orig;
            self.grad_w1[idx] += (loss_plus - base_loss) / eps;
        }

        // Gradients for b1
        for idx in 0..self.b1.len() {
            let orig = self.b1[idx];
            self.b1[idx] = orig + eps;
            let loss_plus = self.loss(input, target);
            self.b1[idx] = orig;
            self.grad_b1[idx] += (loss_plus - base_loss) / eps;
        }

        self.accum_count += 1;
    }

    /// Compute L2 norm of accumulated gradients (before averaging)
    fn gradient_norm(&self) -> f32 {
        let sum: f32 = self
            .grad_w1
            .iter()
            .chain(self.grad_b1.iter())
            .chain(self.grad_w2.iter())
            .chain(self.grad_b2.iter())
            .map(|g| g * g)
            .sum();
        sum.sqrt()
    }

    /// Apply accumulated gradients and zero them out
    fn step_and_zero(&mut self, lr: f32) {
        if self.accum_count == 0 {
            return;
        }
        let scale = lr / self.accum_count as f32;

        for (w, g) in self.w1.iter_mut().zip(self.grad_w1.iter_mut()) {
            *w -= scale * *g;
            *g = 0.0;
        }
        for (b, g) in self.b1.iter_mut().zip(self.grad_b1.iter_mut()) {
            *b -= scale * *g;
            *g = 0.0;
        }
        for (w, g) in self.w2.iter_mut().zip(self.grad_w2.iter_mut()) {
            *w -= scale * *g;
            *g = 0.0;
        }
        for (b, g) in self.b2.iter_mut().zip(self.grad_b2.iter_mut()) {
            *b -= scale * *g;
            *g = 0.0;
        }
        self.accum_count = 0;
    }

    /// Total number of trainable parameters
    fn param_count(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }

    /// Zero all gradient accumulators
    fn zero_grad(&mut self) {
        self.grad_w1 = [0.0; HIDDEN_DIM * INPUT_DIM];
        self.grad_b1 = [0.0; HIDDEN_DIM];
        self.grad_w2 = [0.0; OUTPUT_DIM * HIDDEN_DIM];
        self.grad_b2 = [0.0; OUTPUT_DIM];
        self.accum_count = 0;
    }
}

/// Softmax cross-entropy loss
fn softmax_cross_entropy(logits: &[f32], target: usize) -> f32 {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    -(exps[target] / sum).ln()
}

/// Argmax prediction
fn predict(output: &[f32]) -> usize {
    output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map_or(0, |(i, _)| i)
}

/// Generate labeled training data
fn generate_data(n: usize, seed: u64) -> Vec<([f32; INPUT_DIM], usize)> {
    (0..n)
        .map(|i| {
            let mut input = [0.0f32; INPUT_DIM];
            for (j, x) in input.iter_mut().enumerate() {
                let mut h = DefaultHasher::new();
                (seed, "data", i, j).hash(&mut h);
                *x = h.finish() as f32 / u64::MAX as f32 - 0.5;
            }
            let mut h = DefaultHasher::new();
            (seed, "label", i).hash(&mut h);
            let label = h.finish() as usize % OUTPUT_DIM;
            (input, label)
        })
        .collect()
}

/// Evaluate model accuracy on a dataset
fn evaluate(model: &AccumModel, data: &[([f32; INPUT_DIM], usize)]) -> (f32, f32) {
    let mut total_loss = 0.0f32;
    let mut correct = 0usize;

    for (input, target) in data {
        let (_, output) = model.forward(input);
        total_loss += softmax_cross_entropy(&output, *target);
        if predict(&output) == *target {
            correct += 1;
        }
    }

    let avg_loss = total_loss / data.len() as f32;
    let accuracy = correct as f32 / data.len() as f32;
    (avg_loss, accuracy)
}

/// Estimate memory usage for activation storage
/// Returns bytes needed for one forward pass of `batch_size` samples
fn estimate_activation_memory(batch_size: usize) -> usize {
    let per_sample = (INPUT_DIM + HIDDEN_DIM + OUTPUT_DIM) * size_of::<f32>();
    batch_size * per_sample
}

/// Estimate total training memory: weights + gradients + activations
fn estimate_total_memory(batch_size: usize, param_count: usize) -> usize {
    let weight_mem = param_count * size_of::<f32>();
    let grad_mem = param_count * size_of::<f32>();
    let activation_mem = estimate_activation_memory(batch_size);
    weight_mem + grad_mem + activation_mem
}

/// Train one epoch with gradient accumulation
/// Returns (avg_loss, accuracy, gradient_norms)
fn train_epoch_accumulated(
    model: &mut AccumModel,
    data: &[([f32; INPUT_DIM], usize)],
    lr: f32,
    accum_steps: usize,
) -> (f32, f32, Vec<f32>) {
    let mut total_loss = 0.0f32;
    let mut correct = 0usize;
    let mut grad_norms = Vec::new();

    model.zero_grad();

    for (i, (input, target)) in data.iter().enumerate() {
        // Forward pass for loss/accuracy tracking
        let (_, output) = model.forward(input);
        total_loss += softmax_cross_entropy(&output, *target);
        if predict(&output) == *target {
            correct += 1;
        }

        // Accumulate gradients
        model.backward_accumulate(input, *target);

        // Step when accumulation is complete
        if (i + 1) % accum_steps == 0 {
            grad_norms.push(model.gradient_norm());
            model.step_and_zero(lr);
        }
    }

    // Handle remainder if data size not divisible by accum_steps
    if model.accum_count > 0 {
        grad_norms.push(model.gradient_norm());
        model.step_and_zero(lr);
    }

    let avg_loss = total_loss / data.len() as f32;
    let accuracy = correct as f32 / data.len() as f32;
    (avg_loss, accuracy, grad_norms)
}

/// Result of a full training run
struct TrainResult {
    accum_steps: usize,
    effective_batch_size: usize,
    loss_curve: Vec<f32>,
    acc_curve: Vec<f32>,
    final_loss: f32,
    final_acc: f32,
    val_loss: f32,
    val_acc: f32,
    avg_grad_norm: f32,
    peak_memory_bytes: usize,
}

/// Run a full training loop with given accumulation steps
fn train_full(
    seed: u64,
    train_data: &[([f32; INPUT_DIM], usize)],
    val_data: &[([f32; INPUT_DIM], usize)],
    accum_steps: usize,
    lr: f32,
    epochs: usize,
) -> TrainResult {
    let mut model = AccumModel::new(seed);
    let micro_batch_size = 1; // Each sample is one micro-batch
    let effective_batch_size = micro_batch_size * accum_steps;

    let mut loss_curve = Vec::with_capacity(epochs);
    let mut acc_curve = Vec::with_capacity(epochs);
    let mut all_grad_norms = Vec::new();

    for _ in 0..epochs {
        let (loss, acc, norms) = train_epoch_accumulated(&mut model, train_data, lr, accum_steps);
        loss_curve.push(loss);
        acc_curve.push(acc);
        all_grad_norms.extend_from_slice(&norms);
    }

    let (val_loss, val_acc) = evaluate(&model, val_data);
    let final_loss = loss_curve.last().copied().unwrap_or(f32::INFINITY);
    let final_acc = acc_curve.last().copied().unwrap_or(0.0);

    let avg_grad_norm = if all_grad_norms.is_empty() {
        0.0
    } else {
        all_grad_norms.iter().sum::<f32>() / all_grad_norms.len() as f32
    };

    let peak_memory_bytes = estimate_total_memory(micro_batch_size, model.param_count());

    TrainResult {
        accum_steps,
        effective_batch_size,
        loss_curve,
        acc_curve,
        final_loss,
        final_acc,
        val_loss,
        val_acc,
        avg_grad_norm,
        peak_memory_bytes,
    }
}

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
