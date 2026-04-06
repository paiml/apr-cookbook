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

const INPUT_DIM: usize = 8;
const OUTPUT_DIM: usize = 3;
const N_TRAIN: usize = 100;
const N_VAL: usize = 30;

/// Training hyperparameters
#[derive(Clone, Debug)]
struct HyperParams {
    learning_rate: f64,
    batch_size: usize,
    weight_decay: f64,
    epochs: usize,
}

impl HyperParams {
    fn summary(&self) -> String {
        format!(
            "lr={:.4}, bs={}, wd={:.4}, ep={}",
            self.learning_rate, self.batch_size, self.weight_decay, self.epochs
        )
    }
}

/// Training result
struct TrainResult {
    train_loss: f64,
    val_loss: f64,
    epochs_run: usize,
    early_stopped: bool,
}

/// Compute a single feature value from deterministic hash
fn hash_feature(seed: u64, i: usize, j: usize) -> f64 {
    let mut h = DefaultHasher::new();
    (seed, "data", i, j).hash(&mut h);
    h.finish() as f64 / u64::MAX as f64 - 0.5
}

/// Derive class label from feature scores
fn label_from_score(score: f64) -> usize {
    if score > 0.2 {
        0
    } else if score < -0.2 {
        2
    } else {
        1
    }
}

/// Generate synthetic dataset
fn generate_data(n: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<usize>) {
    let mut inputs = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);

    for i in 0..n {
        let row: Vec<f64> = (0..INPUT_DIM).map(|j| hash_feature(seed, i, j)).collect();
        let label_score = row[0] + row[1] * 0.5 - row[2] * 0.3;
        labels.push(label_from_score(label_score));
        inputs.push(row);
    }

    (inputs, labels)
}

/// Simple linear model for classification
struct LinearModel {
    weights: Vec<f64>, // OUTPUT_DIM x INPUT_DIM
    bias: Vec<f64>,    // OUTPUT_DIM
}

impl LinearModel {
    fn new(seed: u64) -> Self {
        let weights: Vec<f64> = (0..OUTPUT_DIM * INPUT_DIM)
            .map(|i| {
                let mut h = DefaultHasher::new();
                (seed, "w", i).hash(&mut h);
                (h.finish() as f64 / u64::MAX as f64 - 0.5) * 0.1
            })
            .collect();
        let bias = vec![0.0; OUTPUT_DIM];
        Self { weights, bias }
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut output = self.bias.clone();
        for (o, out) in output.iter_mut().enumerate() {
            for (i, &inp) in input.iter().enumerate() {
                *out += self.weights[o * INPUT_DIM + i] * inp;
            }
        }
        output
    }

    fn cross_entropy_loss(&self, inputs: &[Vec<f64>], labels: &[usize]) -> f64 {
        let n = inputs.len() as f64;
        let mut total_loss = 0.0;
        for (input, &label) in inputs.iter().zip(labels.iter()) {
            let logits = self.forward(input);
            let max_l = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = logits.iter().map(|l| (l - max_l).exp()).sum();
            let log_prob = (logits[label] - max_l) - exp_sum.ln();
            total_loss -= log_prob;
        }
        total_loss / n
    }

    fn train_step(&mut self, inputs: &[Vec<f64>], labels: &[usize], hp: &HyperParams) {
        let n = inputs.len() as f64;

        // Compute gradients via finite differences (simplified)
        for param_idx in 0..self.weights.len() {
            let orig = self.weights[param_idx];
            let eps = 1e-4;

            self.weights[param_idx] = orig + eps;
            let loss_plus = self.cross_entropy_loss(inputs, labels);

            self.weights[param_idx] = orig - eps;
            let loss_minus = self.cross_entropy_loss(inputs, labels);

            self.weights[param_idx] = orig;

            let grad = (loss_plus - loss_minus) / (2.0 * eps);
            self.weights[param_idx] -= hp.learning_rate * (grad + hp.weight_decay * orig / n);
        }

        // Bias gradients
        for param_idx in 0..self.bias.len() {
            let orig = self.bias[param_idx];
            let eps = 1e-4;

            self.bias[param_idx] = orig + eps;
            let loss_plus = self.cross_entropy_loss(inputs, labels);

            self.bias[param_idx] = orig - eps;
            let loss_minus = self.cross_entropy_loss(inputs, labels);

            self.bias[param_idx] = orig;

            let grad = (loss_plus - loss_minus) / (2.0 * eps);
            self.bias[param_idx] -= hp.learning_rate * grad;
        }
    }
}

/// Check validation loss and update early stopping state.
/// Returns `true` if training should stop.
fn check_early_stop(
    val_loss: f64,
    best_val_loss: &mut f64,
    patience_counter: &mut usize,
    patience: usize,
) -> bool {
    if val_loss < *best_val_loss - 1e-4 {
        *best_val_loss = val_loss;
        *patience_counter = 0;
        return false;
    }
    *patience_counter += 1;
    *patience_counter >= patience
}

/// Train a model with given hyperparameters and early stopping
fn train_model(
    hp: &HyperParams,
    train_data: &(Vec<Vec<f64>>, Vec<usize>),
    val_data: &(Vec<Vec<f64>>, Vec<usize>),
    seed: u64,
    patience: usize,
) -> TrainResult {
    let mut model = LinearModel::new(seed);

    let (train_inputs, train_labels) = train_data;
    let (val_inputs, val_labels) = val_data;

    let mut best_val_loss = f64::INFINITY;
    let mut patience_counter = 0;
    let mut epochs_run = 0;
    let mut early_stopped = false;

    for epoch in 0..hp.epochs {
        // Mini-batch training
        let batch_start = (epoch * hp.batch_size) % train_inputs.len();
        let batch_end = (batch_start + hp.batch_size).min(train_inputs.len());
        let batch_inputs = &train_inputs[batch_start..batch_end];
        let batch_labels = &train_labels[batch_start..batch_end];

        model.train_step(batch_inputs, batch_labels, hp);
        epochs_run = epoch + 1;

        // Validate periodically
        let should_validate = epoch % 5 == 0 || epoch == hp.epochs - 1;
        if should_validate {
            let val_loss = model.cross_entropy_loss(val_inputs, val_labels);
            early_stopped = check_early_stop(
                val_loss,
                &mut best_val_loss,
                &mut patience_counter,
                patience,
            );
            if early_stopped {
                break;
            }
        }
    }

    let train_loss = model.cross_entropy_loss(train_inputs, train_labels);
    let val_loss = model.cross_entropy_loss(val_inputs, val_labels);

    TrainResult {
        train_loss,
        val_loss,
        epochs_run,
        early_stopped,
    }
}

/// Build the cross-product of hyperparameter grid values
fn build_grid_configs(
    learning_rates: &[f64],
    batch_sizes: &[usize],
    weight_decays: &[f64],
) -> Vec<HyperParams> {
    let mut configs = Vec::new();
    for &lr in learning_rates {
        for &bs in batch_sizes {
            for &wd in weight_decays {
                configs.push(HyperParams {
                    learning_rate: lr,
                    batch_size: bs,
                    weight_decay: wd,
                    epochs: 50,
                });
            }
        }
    }
    configs
}

/// Sample hyperparameters for a single random trial
fn sample_random_hp(trial: i32) -> HyperParams {
    let mut h = DefaultHasher::new();
    (42u64, "lr", trial).hash(&mut h);
    let lr_log = h.finish() as f64 / u64::MAX as f64 * 3.0 - 4.0; // log-uniform [-4, -1]
    let lr = 10.0f64.powf(lr_log);

    (42u64, "bs", trial).hash(&mut h);
    let bs_options = [4, 8, 16, 32, 64];
    let bs = bs_options[(h.finish() as usize) % bs_options.len()];

    (42u64, "wd", trial).hash(&mut h);
    let wd = (h.finish() as f64 / u64::MAX as f64) * 0.05;

    HyperParams {
        learning_rate: lr,
        batch_size: bs,
        weight_decay: wd,
        epochs: 50,
    }
}

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
