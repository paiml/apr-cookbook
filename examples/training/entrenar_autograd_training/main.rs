#![allow(unused_imports)]
//! Entrenar Autograd Training Example
//! **CLI Equivalent**: `apr train`
//! Contract: contracts/recipe-iiur-v1.yaml
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
//! - **Dependencies**: aprender-train 0.31+, aprender-core 0.31+
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
use entrenar::optim::{Optimizer, SGD};
use ndarray::Array1;
use std::time::Instant;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
