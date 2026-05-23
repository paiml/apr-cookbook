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

pub const INPUT_DIM: usize = 8;
pub const OUTPUT_DIM: usize = 3;
pub const N_TRAIN: usize = 100;
pub const N_VAL: usize = 30;

/// Training hyperparameters
#[derive(Clone, Debug)]
pub struct HyperParams {
    pub learning_rate: f64,
    pub batch_size: usize,
    pub weight_decay: f64,
    pub epochs: usize,
}

impl HyperParams {
    pub fn summary(&self) -> String {
        format!(
            "lr={:.4}, bs={}, wd={:.4}, ep={}",
            self.learning_rate, self.batch_size, self.weight_decay, self.epochs
        )
    }
}

/// Training result
pub struct TrainResult {
    pub train_loss: f64,
    pub val_loss: f64,
    pub epochs_run: usize,
    pub early_stopped: bool,
}

/// Compute a single feature value from deterministic hash
pub fn hash_feature(seed: u64, i: usize, j: usize) -> f64 {
    let mut h = DefaultHasher::new();
    (seed, "data", i, j).hash(&mut h);
    h.finish() as f64 / u64::MAX as f64 - 0.5
}

/// Derive class label from feature scores
pub fn label_from_score(score: f64) -> usize {
    if score > 0.2 {
        0
    } else if score < -0.2 {
        2
    } else {
        1
    }
}

/// Generate synthetic dataset
pub fn generate_data(n: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<usize>) {
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
pub struct LinearModel {
    pub weights: Vec<f64>, // OUTPUT_DIM x INPUT_DIM
    pub bias: Vec<f64>,    // OUTPUT_DIM
}

impl LinearModel {
    pub fn new(seed: u64) -> Self {
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

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut output = self.bias.clone();
        for (o, out) in output.iter_mut().enumerate() {
            for (i, &inp) in input.iter().enumerate() {
                *out += self.weights[o * INPUT_DIM + i] * inp;
            }
        }
        output
    }

    pub fn cross_entropy_loss(&self, inputs: &[Vec<f64>], labels: &[usize]) -> f64 {
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

    pub fn train_step(&mut self, inputs: &[Vec<f64>], labels: &[usize], hp: &HyperParams) {
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

// Check validation loss and update early stopping state.
/// Returns `true` if training should stop.
pub fn check_early_stop(
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
pub fn train_model(
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
pub fn build_grid_configs(
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
pub fn sample_random_hp(trial: i32) -> HyperParams {
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
