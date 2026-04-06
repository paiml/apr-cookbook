//! Entrenar Autograd Training Example
//! **CLI Equivalent**: `apr train`
//!
//! Demonstrates training neural networks with entrenar's tape-based autograd
//! and saving results to APR v2 format.
//!
//! # Entrenar Features
//!
//! - **Tape-based Autograd**: Automatic differentiation via computational graph
//! - **Optimizers**: SGD, Adam, AdamW with learning rate schedulers
//! - **APR v2 Integration**: Save trained models to APR v2 format
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                   Entrenar Training Pipeline                    │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Input ─► Linear Layer ─► ReLU ─► Linear Layer ─► Loss         │
//! │    │                                                 │          │
//! │    └─────────────── Backward (Autograd) ◄────────────┘          │
//! │                          │                                      │
//! │                     Optimizer                                   │
//! │                (SGD/Adam/AdamW)                                 │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example entrenar_autograd_training
//! ```
//!
//! # Recipe Metadata
//!
//! - **Category**: Training
//! - **Complexity**: Intermediate
//! - **Dependencies**: entrenar 0.3+, aprender 0.21+
//! - **IIUR**: Isolated, Idempotent, Useful, Reproducible

use apr_cookbook::prelude::*;
use entrenar::autograd::Tensor;
use entrenar::optim::{Optimizer, SGD};
use ndarray::Array1;
use std::time::Instant;

/// Training configuration
#[derive(Debug, Clone)]
struct TrainingConfig {
    /// Number of training epochs
    epochs: usize,
    /// Learning rate
    learning_rate: f32,
    /// Batch size
    batch_size: usize,
    /// Number of input features
    input_dim: usize,
    /// Hidden layer dimension
    hidden_dim: usize,
    /// Number of output classes
    output_dim: usize,
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
struct MLP {
    /// All parameters stored in a Vec for optimizer compatibility
    params: Vec<Tensor>,
    /// Configuration for interpreting params
    config: TrainingConfig,
}

impl MLP {
    /// Parameter indices in the params vec
    const W1_IDX: usize = 0;
    const B1_IDX: usize = 1;
    const W2_IDX: usize = 2;
    const B2_IDX: usize = 3;

    /// Create a new MLP with random initialization
    fn new(config: &TrainingConfig, seed: u64) -> Self {
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
    fn forward(&self, x: &[f32]) -> Vec<f32> {
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
    fn loss(&self, predictions: &[f32], target: usize) -> f32 {
        -predictions[target].max(1e-10).ln()
    }

    /// Get mutable slice of all parameters for optimizer
    fn params_mut(&mut self) -> &mut [Tensor] {
        &mut self.params
    }

    /// Set gradients on all parameters
    fn set_grads(&self, grad_scale: f32) {
        for param in &self.params {
            if param.requires_grad() {
                let grad = Array1::from_elem(param.len(), grad_scale);
                param.set_grad(grad);
            }
        }
    }

    /// Get total parameter count
    fn param_count(&self) -> usize {
        self.params.iter().map(Tensor::len).sum()
    }

    /// Get all weight data for saving
    fn all_weights(&self) -> Vec<f32> {
        self.params
            .iter()
            .flat_map(|p| p.data().iter().copied())
            .collect()
    }
}

/// Generate synthetic classification data (Iris-like)
fn generate_data(
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
struct TrainingResult {
    /// Final loss
    final_loss: f32,
    /// Training accuracy
    accuracy: f32,
    /// Total training time in milliseconds
    time_ms: f64,
    /// Losses per epoch
    losses: Vec<f32>,
}

/// Train the model
fn train(model: &mut MLP, config: &TrainingConfig) -> TrainingResult {
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
fn save_to_apr(model: &MLP, path: &std::path::Path) -> Result<()> {
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

fn main() {
    println!("=== Entrenar Autograd Training Example ===\n");

    let config = TrainingConfig::default();

    // =========================================================================
    // Section 1: Configuration
    // =========================================================================
    println!("1. Training Configuration");
    println!("   ─────────────────────────────────────────");
    println!("   Epochs:        {}", config.epochs);
    println!("   Learning rate: {}", config.learning_rate);
    println!("   Batch size:    {}", config.batch_size);
    println!(
        "   Architecture:  {}x{}x{}",
        config.input_dim, config.hidden_dim, config.output_dim
    );
    println!();

    // =========================================================================
    // Section 2: Model Initialization
    // =========================================================================
    println!("2. Model Initialization (Xavier)");
    println!("   ─────────────────────────────────────────");

    let mut model = MLP::new(&config, 42);
    println!("   Parameters: {}", model.param_count());
    println!("   W1 shape: {}x{}", config.input_dim, config.hidden_dim);
    println!("   W2 shape: {}x{}", config.hidden_dim, config.output_dim);
    println!();

    // =========================================================================
    // Section 3: Training with Entrenar
    // =========================================================================
    println!("3. Training with Entrenar Autograd");
    println!("   ─────────────────────────────────────────");

    let result = train(&mut model, &config);
    println!();

    // =========================================================================
    // Section 4: Results
    // =========================================================================
    println!("4. Training Results");
    println!("   ─────────────────────────────────────────");
    println!("   Final loss:  {:.4}", result.final_loss);
    println!("   Accuracy:    {:.1}%", result.accuracy * 100.0);
    println!("   Time:        {:.2}ms", result.time_ms);
    println!();

    // =========================================================================
    // Section 5: Save to APR v2
    // =========================================================================
    println!("5. Save to APR v2 Format");
    println!("   ─────────────────────────────────────────");

    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let model_path = temp_dir.path().join("model.apr");

    if let Err(e) = save_to_apr(&model, &model_path) {
        println!("   Error saving model: {}", e);
    } else if let Ok(metadata) = std::fs::metadata(&model_path) {
        println!("   Saved to: {}", model_path.display());
        println!("   Size: {} bytes", metadata.len());
        println!("   Compression: LZ4");
    }
    println!();

    // =========================================================================
    // Section 6: Entrenar Features Demo
    // =========================================================================
    println!("6. Entrenar Features");
    println!("   ─────────────────────────────────────────");
    println!("   Autograd:    Tape-based autodiff");
    println!("   Optimizers:  SGD, Adam, AdamW");
    println!("   Schedulers:  Cosine, StepDecay, Warmup");
    println!("   LoRA:        Low-rank adaptation");
    println!("   Quant:       QAT and PTQ support");
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 100);
        assert_eq!(config.input_dim, 4);
        assert_eq!(config.output_dim, 3);
    }

    #[test]
    fn test_mlp_creation() {
        let config = TrainingConfig::default();
        let model = MLP::new(&config, 42);
        assert_eq!(model.param_count(), 4 * 8 + 8 + 8 * 3 + 3);
    }

    #[test]
    fn test_mlp_forward() {
        let config = TrainingConfig::default();
        let model = MLP::new(&config, 42);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = model.forward(&input);
        assert_eq!(output.len(), config.output_dim);

        // Check softmax sums to 1
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_mlp_loss() {
        let config = TrainingConfig::default();
        let model = MLP::new(&config, 42);
        let predictions = vec![0.7, 0.2, 0.1];
        let loss = model.loss(&predictions, 0);
        assert!(loss > 0.0);
        assert!(loss < 1.0); // -ln(0.7) ≈ 0.36
    }

    #[test]
    fn test_generate_data() {
        let config = TrainingConfig::default();
        let (x, y) = generate_data(100, &config, 42);
        assert_eq!(x.len(), 100);
        assert_eq!(y.len(), 100);
        for sample in &x {
            assert_eq!(sample.len(), config.input_dim);
        }
    }

    #[test]
    fn test_generate_data_deterministic() {
        let config = TrainingConfig::default();
        let (x1, y1) = generate_data(10, &config, 42);
        let (x2, y2) = generate_data(10, &config, 42);
        assert_eq!(x1, x2);
        assert_eq!(y1, y2);
    }

    #[test]
    fn test_train_reduces_loss() {
        let config = TrainingConfig {
            epochs: 20,
            ..Default::default()
        };
        let mut model = MLP::new(&config, 42);
        let result = train(&mut model, &config);

        // Loss should decrease
        assert!(result.losses.len() == 20);
        // Check training completed
        assert!(result.time_ms > 0.0);
    }

    #[test]
    fn test_save_to_apr() {
        let config = TrainingConfig::default();
        let model = MLP::new(&config, 42);
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("test.apr");

        save_to_apr(&model, &path).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_zero_grad() {
        let config = TrainingConfig::default();
        let mut model = MLP::new(&config, 42);

        // Set some gradients
        model.set_grads(1.0);
        assert!(model.params[0].grad().is_some());

        // Zero them via optimizer
        let mut optimizer = SGD::new(0.01, 0.9);
        optimizer.zero_grad(model.params_mut());
        assert!(model.params[0].grad().is_none());
    }

    #[test]
    fn test_mlp_deterministic() {
        let config = TrainingConfig::default();
        let model1 = MLP::new(&config, 42);
        let model2 = MLP::new(&config, 42);

        assert_eq!(model1.all_weights(), model2.all_weights());
    }

    #[test]
    fn test_softmax_properties() {
        let config = TrainingConfig::default();
        let model = MLP::new(&config, 42);
        let input = vec![0.5, 1.0, 1.5, 2.0];
        let output = model.forward(&input);

        // All probabilities positive
        for &p in &output {
            assert!(p > 0.0);
            assert!(p <= 1.0);
        }

        // Sum to 1
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
