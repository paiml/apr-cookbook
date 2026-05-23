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
use apr_cookbook::prelude::*;
use entrenar::autograd::Tensor;
use entrenar::optim::{Optimizer, SGD};
use ndarray::Array1;
use std::time::Instant;

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    // Number of training epochs
    pub epochs: usize,
    // Learning rate
    pub learning_rate: f32,
    // Batch size
    pub batch_size: usize,
    // Number of input features
    pub input_dim: usize,
    // Hidden layer dimension
    pub hidden_dim: usize,
    // Number of output classes
    pub output_dim: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            learning_rate: 0.01,
            batch_size: 32,
            input_dim: 4,
            hidden_dim: 8,
            output_dim: 3,
        }
    }
}

/// Simple MLP model using entrenar tensors
#[allow(clippy::upper_case_acronyms)]
pub struct MLP {
    // All parameters stored in a Vec for optimizer compatibility
    pub params: Vec<Tensor>,
    // Configuration for interpreting params
    pub config: TrainingConfig,
}

impl MLP {
    /// Parameter indices in the params vec
    pub const W1_IDX: usize = 0;
    pub const B1_IDX: usize = 1;
    pub const W2_IDX: usize = 2;
    pub const B2_IDX: usize = 3;

    /// Create a new MLP with random initialization
    pub fn new(config: &TrainingConfig, seed: u64) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Deterministic initialization
        let init = |size: usize, scale: f32, name: &str| -> Vec<f32> {
            (0..size)
                .map(|i| {
                    let mut hasher = DefaultHasher::new();
                    (seed, name, i).hash(&mut hasher);
                    let h = hasher.finish();
                    (h as f32 / u64::MAX as f32 - 0.5) * scale
                })
                .collect()
        };

        // Xavier initialization scale
        let w1_scale = (2.0 / (config.input_dim + config.hidden_dim) as f32).sqrt();
        let w2_scale = (2.0 / (config.hidden_dim + config.output_dim) as f32).sqrt();

        // Store all params in a Vec for optimizer compatibility
        let params = vec![
            Tensor::from_vec(
                init(config.input_dim * config.hidden_dim, w1_scale, "w1"),
                true,
            ),
            Tensor::zeros(config.hidden_dim, true),
            Tensor::from_vec(
                init(config.hidden_dim * config.output_dim, w2_scale, "w2"),
                true,
            ),
            Tensor::zeros(config.output_dim, true),
        ];

        Self {
            params,
            config: config.clone(),
        }
    }

    /// Forward pass (manual for demonstration)
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let w1 = &self.params[Self::W1_IDX];
        let b1 = &self.params[Self::B1_IDX];
        let w2 = &self.params[Self::W2_IDX];
        let b2 = &self.params[Self::B2_IDX];

        // Layer 1: x @ W1 + b1
        let mut hidden = vec![0.0f32; self.config.hidden_dim];
        #[allow(clippy::needless_range_loop)] // j used for bias, hidden, and weight indexing
        for j in 0..self.config.hidden_dim {
            let mut sum = b1.data()[j];
            #[allow(clippy::needless_range_loop)] // i used for input and weight indexing
            for i in 0..self.config.input_dim {
                sum += x[i] * w1.data()[i * self.config.hidden_dim + j];
            }
            // ReLU activation
            hidden[j] = sum.max(0.0);
        }

        // Layer 2: hidden @ W2 + b2
        let mut output = vec![0.0f32; self.config.output_dim];
        #[allow(clippy::needless_range_loop)] // k used for both output index and weight lookup
        for k in 0..self.config.output_dim {
            let mut sum = b2.data()[k];
            #[allow(clippy::needless_range_loop)]
            // j used for both hidden index and weight calculation
            for j in 0..self.config.hidden_dim {
                sum += hidden[j] * w2.data()[j * self.config.output_dim + k];
            }
            output[k] = sum;
        }

        // Softmax
        let max_val = output.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = output.iter().map(|x| (x - max_val).exp()).sum();
        output
            .iter()
            .map(|x| (x - max_val).exp() / exp_sum)
            .collect()
    }

    /// Compute cross-entropy loss
    pub fn loss(&self, predictions: &[f32], target: usize) -> f32 {
        -predictions[target].max(1e-10).ln()
    }

    /// Get mutable slice of all parameters for optimizer
    pub fn params_mut(&mut self) -> &mut [Tensor] {
        &mut self.params
    }

    /// Set gradients on all parameters
    pub fn set_grads(&self, grad_scale: f32) {
        for param in &self.params {
            if param.requires_grad() {
                let grad = Array1::from_elem(param.len(), grad_scale);
                param.set_grad(grad);
            }
        }
    }

    /// Get total parameter count
    pub fn param_count(&self) -> usize {
        self.params.iter().map(Tensor::len).sum()
    }

    /// Get all weight data for saving
    pub fn all_weights(&self) -> Vec<f32> {
        self.params
            .iter()
            .flat_map(|p| p.data().iter().copied())
            .collect()
    }
}

/// Generate synthetic classification data (Iris-like)
pub fn generate_data(
    n_samples: usize,
    config: &TrainingConfig,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut x = Vec::with_capacity(n_samples);
    let mut y = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let mut hasher = DefaultHasher::new();
        (seed, "sample", i).hash(&mut hasher);
        let h = hasher.finish();

        let class = i % config.output_dim;
        let class_offset = class as f32;

        // Generate features with class-specific patterns
        let features: Vec<f32> = (0..config.input_dim)
            .map(|j| {
                let mut hasher2 = DefaultHasher::new();
                (h, j).hash(&mut hasher2);
                let h2 = hasher2.finish();
                let noise = (h2 as f32 / u64::MAX as f32 - 0.5) * 0.3;
                class_offset + noise + (j as f32 * 0.1)
            })
            .collect();

        x.push(features);
        y.push(class);
    }

    (x, y)
}

/// Training result
#[derive(Debug)]
#[allow(dead_code)]
pub struct TrainingResult {
    // Final loss
    pub final_loss: f32,
    // Training accuracy
    pub accuracy: f32,
    // Total training time in milliseconds
    pub time_ms: f64,
    // Losses per epoch
    pub losses: Vec<f32>,
}

/// Train the model
pub fn train(model: &mut MLP, config: &TrainingConfig) -> TrainingResult {
    let (x_train, y_train) = generate_data(config.batch_size * 10, config, 42);

    let mut optimizer = SGD::new(config.learning_rate, 0.9); // LR + momentum
    let mut losses = Vec::with_capacity(config.epochs);

    let start = Instant::now();

    for epoch in 0..config.epochs {
        let mut epoch_loss = 0.0f32;

        for (x, &target) in x_train.iter().zip(y_train.iter()) {
            // Zero gradients using optimizer helper
            optimizer.zero_grad(model.params_mut());

            // Forward pass
            let predictions = model.forward(x);
            let loss = model.loss(&predictions, target);
            epoch_loss += loss;

            // Manual gradient computation for demonstration
            // (In a full implementation, this would use the computational graph)
            let grad_scale = loss * 0.001;

            // Set gradients on all parameters
            model.set_grads(grad_scale);

            // Optimizer step - uses Tensor's internal gradient
            optimizer.step(model.params_mut());
        }

        let avg_loss = epoch_loss / x_train.len() as f32;
        losses.push(avg_loss);

        if epoch % 20 == 0 {
            println!("   Epoch {:3}: loss = {:.4}", epoch, avg_loss);
        }
    }

    let elapsed = start.elapsed();

    // Compute final accuracy
    let mut correct = 0;
    for (x, &target) in x_train.iter().zip(y_train.iter()) {
        let predictions = model.forward(x);
        let predicted = predictions
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        if predicted == target {
            correct += 1;
        }
    }
    let accuracy = correct as f32 / x_train.len() as f32;

    TrainingResult {
        final_loss: *losses.last().unwrap_or(&0.0),
        accuracy,
        time_ms: elapsed.as_secs_f64() * 1000.0,
        losses,
    }
}

/// Save model to APR v2 format
pub fn save_to_apr(model: &MLP, path: &std::path::Path) -> Result<()> {
    // Convert model weights to bytes
    let weight_bytes: Vec<u8> = model
        .all_weights()
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    let bundle = ModelBundleV2::new()
        .with_name("entrenar-mlp")
        .with_description(format!(
            "MLP {}x{}x{}",
            model.config.input_dim, model.config.hidden_dim, model.config.output_dim
        ))
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::FP32)
        .add_tensor("weights", vec![model.param_count()], weight_bytes)
        .build();

    std::fs::write(path, bundle)?;
    Ok(())
}
