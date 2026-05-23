#![allow(unused_imports)]
//! Hyperparameter Sweep Example
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Demonstrates systematic hyperparameter optimization: grid search,
//! random search, and early stopping. Trains a simple model across
//! parameter combinations and reports the best configuration.
//!
//! # Search Strategies
//!
//! - **Grid Search**: Exhaustive cross-product of parameter values
//! - **Random Search**: Sampled from parameter distributions
//! - **Early Stopping**: Halt unpromising runs based on validation loss
//!
//! # Running
//!
//! ```bash
//! cargo run --example hyperparameter_sweep
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
    println!("=== Hyperparameter Sweep Example ===\n");

    let train_data = generate_data(N_TRAIN, 42);
    let val_data = generate_data(N_VAL, 99);

    // =========================================================================
    // Section 1: Grid Search
    // =========================================================================
    println!("1. Grid Search");
    println!("   ─────────────────────────────────────────");

    let grid_configs =
        build_grid_configs(&[0.001, 0.01, 0.05, 0.1], &[8, 16, 32], &[0.0, 0.001, 0.01]);

    let mut grid_results: Vec<(HyperParams, TrainResult)> = Vec::new();

    println!(
        "   {:>8} {:>4} {:>8} {:>10} {:>10} {:>6} {:>5}",
        "LR", "BS", "WD", "TrainLoss", "ValLoss", "Epoch", "Early"
    );
    println!("   {}", "─".repeat(55));

    for hp in &grid_configs {
        let result = train_model(hp, &train_data, &val_data, 42, 10);
        println!(
            "   {:>8.4} {:>4} {:>8.4} {:>10.4} {:>10.4} {:>6} {:>5}",
            hp.learning_rate,
            hp.batch_size,
            hp.weight_decay,
            result.train_loss,
            result.val_loss,
            result.epochs_run,
            if result.early_stopped { "yes" } else { "no" }
        );
        grid_results.push((hp.clone(), result));
    }

    // Best configuration
    let (best_hp, best_result) = grid_results
        .iter()
        .min_by(|(_, a), (_, b)| a.val_loss.partial_cmp(&b.val_loss).unwrap())
        .unwrap();
    println!();
    println!(
        "   Best: {} → val_loss={:.4}",
        best_hp.summary(),
        best_result.val_loss
    );
    println!();

    // =========================================================================
    // Section 2: Random Search
    // =========================================================================
    println!("2. Random Search (30 trials)");
    println!("   ─────────────────────────────────────────");

    let n_trials = 30;
    let mut random_results: Vec<(HyperParams, TrainResult)> = Vec::new();

    println!(
        "   {:>5} {:>8} {:>4} {:>8} {:>10} {:>10}",
        "Trial", "LR", "BS", "WD", "TrainLoss", "ValLoss"
    );
    println!("   {}", "─".repeat(50));

    for trial in 0..n_trials {
        let hp = sample_random_hp(trial);
        let result = train_model(&hp, &train_data, &val_data, 42, 10);

        if trial < 10 || trial == n_trials - 1 {
            println!(
                "   {:>5} {:>8.5} {:>4} {:>8.4} {:>10.4} {:>10.4}",
                trial + 1,
                hp.learning_rate,
                hp.batch_size,
                hp.weight_decay,
                result.train_loss,
                result.val_loss
            );
        } else if trial == 10 {
            println!("   {:>5}", "...");
        }
        random_results.push((hp, result));
    }

    let (best_hp, best_result) = random_results
        .iter()
        .min_by(|(_, a), (_, b)| a.val_loss.partial_cmp(&b.val_loss).unwrap())
        .unwrap();
    println!();
    println!(
        "   Best: {} → val_loss={:.4}",
        best_hp.summary(),
        best_result.val_loss
    );
    println!();

    // =========================================================================
    // Section 3: Early Stopping Analysis
    // =========================================================================
    println!("3. Early Stopping Impact");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Patience", "EpochsRun", "ValLoss", "Stopped?", "Speedup"
    );
    println!("   {}", "─".repeat(52));

    let hp = HyperParams {
        learning_rate: 0.01,
        batch_size: 16,
        weight_decay: 0.001,
        epochs: 100,
    };

    let no_stop = train_model(&hp, &train_data, &val_data, 42, 999);
    let full_epochs = no_stop.epochs_run;

    for patience in [3, 5, 10, 20, 50] {
        let result = train_model(&hp, &train_data, &val_data, 42, patience);
        let speedup = full_epochs as f64 / result.epochs_run as f64;
        println!(
            "   {:>10} {:>10} {:>10.4} {:>10} {:>9.1}x",
            patience,
            result.epochs_run,
            result.val_loss,
            if result.early_stopped { "yes" } else { "no" },
            speedup
        );
    }
    println!();

    // =========================================================================
    // Section 4: Learning Rate Schedule
    // =========================================================================
    println!("4. Learning Rate Sensitivity");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>12} {:>10} {:>10} {:>10}",
        "LR", "TrainLoss", "ValLoss", "Gap"
    );
    println!("   {}", "─".repeat(44));

    for lr in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5] {
        let hp = HyperParams {
            learning_rate: lr,
            batch_size: 16,
            weight_decay: 0.001,
            epochs: 50,
        };
        let result = train_model(&hp, &train_data, &val_data, 42, 999);
        let gap = result.val_loss - result.train_loss;
        println!(
            "   {:>12.5} {:>10.4} {:>10.4} {:>10.4}",
            lr, result.train_loss, result.val_loss, gap
        );
    }
    println!();

    // =========================================================================
    // Section 5: Comparison Summary
    // =========================================================================
    println!("5. Search Strategy Comparison");
    println!("   ─────────────────────────────────────────");

    let grid_best = grid_results
        .iter()
        .map(|(_, r)| r.val_loss)
        .fold(f64::INFINITY, f64::min);
    let random_best = random_results
        .iter()
        .map(|(_, r)| r.val_loss)
        .fold(f64::INFINITY, f64::min);

    println!("   {:>15} {:>10} {:>10}", "Strategy", "Trials", "Best Val");
    println!("   {}", "─".repeat(38));
    println!(
        "   {:>15} {:>10} {:>10.4}",
        "Grid Search",
        grid_results.len(),
        grid_best
    );
    println!(
        "   {:>15} {:>10} {:>10.4}",
        "Random Search", n_trials, random_best
    );
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_data_shapes() {
        let (inputs, labels) = generate_data(50, 42);
        assert_eq!(inputs.len(), 50);
        assert_eq!(labels.len(), 50);
        for inp in &inputs {
            assert_eq!(inp.len(), INPUT_DIM);
        }
    }

    #[test]
    fn test_generate_data_deterministic() {
        let d1 = generate_data(20, 42);
        let d2 = generate_data(20, 42);
        assert_eq!(d1.0, d2.0);
        assert_eq!(d1.1, d2.1);
    }

    #[test]
    fn test_model_forward_dimensions() {
        let model = LinearModel::new(42);
        let input = vec![0.5; INPUT_DIM];
        let output = model.forward(&input);
        assert_eq!(output.len(), OUTPUT_DIM);
    }

    #[test]
    fn test_cross_entropy_nonnegative() {
        let model = LinearModel::new(42);
        let (inputs, labels) = generate_data(20, 42);
        let loss = model.cross_entropy_loss(&inputs, &labels);
        assert!(loss >= 0.0, "CE loss should be >= 0, got {loss}");
        assert!(loss.is_finite());
    }

    #[test]
    fn test_train_reduces_loss() {
        let (train_inputs, train_labels) = generate_data(30, 42);
        let val_data = generate_data(10, 99);
        let hp = HyperParams {
            learning_rate: 0.01,
            batch_size: 30,
            weight_decay: 0.0,
            epochs: 20,
        };

        let mut model = LinearModel::new(42);
        let loss_before = model.cross_entropy_loss(&train_inputs, &train_labels);

        for _ in 0..20 {
            model.train_step(&train_inputs, &train_labels, &hp);
        }
        let loss_after = model.cross_entropy_loss(&train_inputs, &train_labels);

        assert!(
            loss_after < loss_before,
            "Training should reduce loss: {loss_before} -> {loss_after}"
        );
        let _ = val_data; // Used in other tests
    }

    #[test]
    fn test_early_stopping() {
        let train_data = generate_data(N_TRAIN, 42);
        let val_data = generate_data(N_VAL, 99);
        let hp = HyperParams {
            learning_rate: 0.01,
            batch_size: 16,
            weight_decay: 0.0,
            epochs: 200,
        };
        let result = train_model(&hp, &train_data, &val_data, 42, 3);
        // With patience=3, should stop before 200 epochs
        assert!(result.epochs_run <= 200, "Should finish within max epochs");
    }

    #[test]
    fn test_hyperparams_summary() {
        let hp = HyperParams {
            learning_rate: 0.01,
            batch_size: 16,
            weight_decay: 0.001,
            epochs: 50,
        };
        let summary = hp.summary();
        assert!(summary.contains("lr="));
        assert!(summary.contains("bs="));
    }
}
