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
#[allow(unused_imports)]
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub const INPUT_DIM: usize = 16;
pub const HIDDEN_DIM: usize = 8;
pub const OUTPUT_DIM: usize = 4;
pub const N_TRAIN: usize = 128;
pub const N_VAL: usize = 32;
pub const TOTAL_EPOCHS: usize = 20;

/// Accumulation steps to compare
pub const ACCUM_STEPS: [usize; 4] = [1, 4, 8, 16];

/// Deterministic hash-based value generation
pub fn hash_f32(seed: u64, index: usize, label: &str) -> f32 {
    let mut h = DefaultHasher::new();
    (seed, label, index).hash(&mut h);
    h.finish() as f32 / u64::MAX as f32 - 0.5
}

/// Two-layer model with gradient accumulation support
pub struct AccumModel {
    pub w1: [f32; HIDDEN_DIM * INPUT_DIM],
    pub b1: [f32; HIDDEN_DIM],
    pub w2: [f32; OUTPUT_DIM * HIDDEN_DIM],
    pub b2: [f32; OUTPUT_DIM],
    // Gradient accumulators
    pub grad_w1: [f32; HIDDEN_DIM * INPUT_DIM],
    pub grad_b1: [f32; HIDDEN_DIM],
    pub grad_w2: [f32; OUTPUT_DIM * HIDDEN_DIM],
    pub grad_b2: [f32; OUTPUT_DIM],
    pub accum_count: usize,
}

impl AccumModel {
    pub fn new(seed: u64) -> Self {
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
    pub fn forward(&self, input: &[f32]) -> ([f32; HIDDEN_DIM], [f32; OUTPUT_DIM]) {
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
    pub fn loss(&self, input: &[f32], target: usize) -> f32 {
        let (_, output) = self.forward(input);
        softmax_cross_entropy(&output, target)
    }

    /// Simulate backward pass: accumulate gradients via finite differences
    pub fn backward_accumulate(&mut self, input: &[f32], target: usize) {
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
    pub fn gradient_norm(&self) -> f32 {
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
    pub fn step_and_zero(&mut self, lr: f32) {
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
    pub fn param_count(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }

    /// Zero all gradient accumulators
    pub fn zero_grad(&mut self) {
        self.grad_w1 = [0.0; HIDDEN_DIM * INPUT_DIM];
        self.grad_b1 = [0.0; HIDDEN_DIM];
        self.grad_w2 = [0.0; OUTPUT_DIM * HIDDEN_DIM];
        self.grad_b2 = [0.0; OUTPUT_DIM];
        self.accum_count = 0;
    }
}

/// Softmax cross-entropy loss
pub fn softmax_cross_entropy(logits: &[f32], target: usize) -> f32 {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    -(exps[target] / sum).ln()
}

/// Argmax prediction
pub fn predict(output: &[f32]) -> usize {
    output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map_or(0, |(i, _)| i)
}

/// Generate labeled training data
pub fn generate_data(n: usize, seed: u64) -> Vec<([f32; INPUT_DIM], usize)> {
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
pub fn evaluate(model: &AccumModel, data: &[([f32; INPUT_DIM], usize)]) -> (f32, f32) {
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

// Estimate memory usage for activation storage
/// Returns bytes needed for one forward pass of `batch_size` samples
pub fn estimate_activation_memory(batch_size: usize) -> usize {
    let per_sample = (INPUT_DIM + HIDDEN_DIM + OUTPUT_DIM) * size_of::<f32>();
    batch_size * per_sample
}

/// Estimate total training memory: weights + gradients + activations
pub fn estimate_total_memory(batch_size: usize, param_count: usize) -> usize {
    let weight_mem = param_count * size_of::<f32>();
    let grad_mem = param_count * size_of::<f32>();
    let activation_mem = estimate_activation_memory(batch_size);
    weight_mem + grad_mem + activation_mem
}

// Train one epoch with gradient accumulation
/// Returns (avg_loss, accuracy, gradient_norms)
pub fn train_epoch_accumulated(
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
pub struct TrainResult {
    pub accum_steps: usize,
    pub effective_batch_size: usize,
    pub loss_curve: Vec<f32>,
    pub acc_curve: Vec<f32>,
    pub final_loss: f32,
    pub final_acc: f32,
    pub val_loss: f32,
    pub val_acc: f32,
    pub avg_grad_norm: f32,
    pub peak_memory_bytes: usize,
}

/// Run a full training loop with given accumulation steps
pub fn train_full(
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
