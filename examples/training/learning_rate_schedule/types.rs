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

pub const INPUT_DIM: usize = 12;
pub const OUTPUT_DIM: usize = 4;
pub const N_TRAIN: usize = 80;
pub const N_VAL: usize = 20;
pub const TOTAL_EPOCHS: usize = 100;
pub const SPARKLINE_WIDTH: usize = 50;

/// Learning rate schedule type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Schedule {
    Constant,
    StepDecay,
    Exponential,
    CosineAnnealing,
    WarmupCosine,
    OneCycle,
    Polynomial,
}

pub const ALL_SCHEDULES: [Schedule; 7] = [
    Schedule::Constant,
    Schedule::StepDecay,
    Schedule::Exponential,
    Schedule::CosineAnnealing,
    Schedule::WarmupCosine,
    Schedule::OneCycle,
    Schedule::Polynomial,
];

impl Schedule {
    pub fn name(self) -> &'static str {
        match self {
            Schedule::Constant => "Constant",
            Schedule::StepDecay => "StepDecay",
            Schedule::Exponential => "Exponential",
            Schedule::CosineAnnealing => "CosineAnnealing",
            Schedule::WarmupCosine => "Warmup+Cosine",
            Schedule::OneCycle => "OneCycle",
            Schedule::Polynomial => "Polynomial",
        }
    }

    pub fn short_name(self) -> &'static str {
        match self {
            Schedule::Constant => "const",
            Schedule::StepDecay => "step",
            Schedule::Exponential => "exp",
            Schedule::CosineAnnealing => "cosine",
            Schedule::WarmupCosine => "w+cos",
            Schedule::OneCycle => "1cycle",
            Schedule::Polynomial => "poly",
        }
    }
}

/// Configuration for learning rate scheduling
pub struct ScheduleConfig {
    pub base_lr: f64,
    pub min_lr: f64,
    pub warmup_epochs: usize,
    pub total_epochs: usize,
    pub step_size: usize,
    pub step_gamma: f64,
    pub exp_gamma: f64,
    pub poly_power: f64,
    pub one_cycle_max_lr: f64,
    pub one_cycle_pct_start: f64,
}

impl ScheduleConfig {
    pub fn default_config() -> Self {
        Self {
            base_lr: 0.01,
            min_lr: 1e-6,
            warmup_epochs: 10,
            total_epochs: TOTAL_EPOCHS,
            step_size: 30,
            step_gamma: 0.1,
            exp_gamma: 0.97,
            poly_power: 2.0,
            one_cycle_max_lr: 0.05,
            one_cycle_pct_start: 0.3,
        }
    }
}

/// Compute the learning rate at a given epoch for a schedule
pub fn get_lr(schedule: Schedule, epoch: usize, config: &ScheduleConfig) -> f64 {
    let t = epoch as f64;
    let total = config.total_epochs as f64;

    match schedule {
        Schedule::Constant => config.base_lr,

        Schedule::StepDecay => {
            let num_decays = epoch / config.step_size;
            config.base_lr * config.step_gamma.powi(num_decays as i32)
        }

        Schedule::Exponential => config.base_lr * config.exp_gamma.powf(t),

        Schedule::CosineAnnealing => {
            let cos_val = (std::f64::consts::PI * t / total).cos();
            config.min_lr + 0.5 * (config.base_lr - config.min_lr) * (1.0 + cos_val)
        }

        Schedule::WarmupCosine => {
            let warmup = config.warmup_epochs as f64;
            if t < warmup {
                // Linear warmup from min_lr to base_lr
                config.min_lr + (config.base_lr - config.min_lr) * (t / warmup)
            } else {
                // Cosine decay from base_lr to min_lr
                let progress = (t - warmup) / (total - warmup);
                let cos_val = (std::f64::consts::PI * progress).cos();
                config.min_lr + 0.5 * (config.base_lr - config.min_lr) * (1.0 + cos_val)
            }
        }

        Schedule::OneCycle => {
            let pct_start = config.one_cycle_pct_start;
            let max_lr = config.one_cycle_max_lr;
            let progress = t / total;

            if progress < pct_start {
                // Phase 1: ramp up from base_lr to max_lr
                let phase_progress = progress / pct_start;
                config.base_lr + (max_lr - config.base_lr) * phase_progress
            } else {
                // Phase 2: cosine decay from max_lr to min_lr
                let phase_progress = (progress - pct_start) / (1.0 - pct_start);
                let cos_val = (std::f64::consts::PI * phase_progress).cos();
                config.min_lr + 0.5 * (max_lr - config.min_lr) * (1.0 + cos_val)
            }
        }

        Schedule::Polynomial => {
            let progress = t / total;
            let decay = (1.0 - progress).powf(config.poly_power);
            config.min_lr + (config.base_lr - config.min_lr) * decay
        }
    }
}

/// Render a sparkline string from a slice of values
pub fn sparkline(values: &[f64], width: usize) -> String {
    let blocks = [
        ' ', '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}',
        '\u{2588}',
    ];
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-12);

    // Resample to width
    let n = values.len();
    let mut chars = String::with_capacity(width);
    for col in 0..width {
        let idx = col * n / width;
        let normalized = (values[idx.min(n - 1)] - min) / range;
        let level = (normalized * 8.0).round() as usize;
        chars.push(blocks[level.min(8)]);
    }
    chars
}

/// Simple linear model for training simulation
pub struct LinearModel {
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
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

    pub fn mse_loss(&self, inputs: &[Vec<f64>], targets: &[Vec<f64>]) -> f64 {
        let n = inputs.len() as f64;
        inputs
            .iter()
            .zip(targets.iter())
            .map(|(x, t)| {
                let pred = self.forward(x);
                pred.iter()
                    .zip(t.iter())
                    .map(|(p, y)| (p - y).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>()
            / n
    }

    pub fn train_step(&mut self, inputs: &[Vec<f64>], targets: &[Vec<f64>], lr: f64) {
        let eps = 1e-5;
        // Weight gradients via finite differences
        for idx in 0..self.weights.len() {
            let orig = self.weights[idx];
            self.weights[idx] = orig + eps;
            let loss_plus = self.mse_loss(inputs, targets);
            self.weights[idx] = orig - eps;
            let loss_minus = self.mse_loss(inputs, targets);
            self.weights[idx] = orig;
            let grad = (loss_plus - loss_minus) / (2.0 * eps);
            self.weights[idx] -= lr * grad;
        }
        // Bias gradients
        for idx in 0..self.bias.len() {
            let orig = self.bias[idx];
            self.bias[idx] = orig + eps;
            let loss_plus = self.mse_loss(inputs, targets);
            self.bias[idx] = orig - eps;
            let loss_minus = self.mse_loss(inputs, targets);
            self.bias[idx] = orig;
            let grad = (loss_plus - loss_minus) / (2.0 * eps);
            self.bias[idx] -= lr * grad;
        }
    }
}

/// Generate synthetic regression dataset
pub fn generate_data(n: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut inputs = Vec::with_capacity(n);
    let mut targets = Vec::with_capacity(n);
    for i in 0..n {
        let x: Vec<f64> = (0..INPUT_DIM)
            .map(|j| {
                let mut h = DefaultHasher::new();
                (seed, "data", i, j).hash(&mut h);
                h.finish() as f64 / u64::MAX as f64 - 0.5
            })
            .collect();
        let y: Vec<f64> = (0..OUTPUT_DIM)
            .map(|k| {
                x.iter()
                    .enumerate()
                    .map(|(j, &v)| v * ((j + k) as f64 * 0.1).sin())
                    .sum()
            })
            .collect();
        inputs.push(x);
        targets.push(y);
    }
    (inputs, targets)
}

/// Training result for one schedule run
pub struct TrainResult {
    pub schedule: Schedule,
    pub loss_curve: Vec<f64>,
    pub final_train_loss: f64,
    pub final_val_loss: f64,
    pub best_val_loss: f64,
    pub best_epoch: usize,
}

/// Run a full training loop with a given schedule
pub fn train_with_schedule(
    schedule: Schedule,
    config: &ScheduleConfig,
    train: &(Vec<Vec<f64>>, Vec<Vec<f64>>),
    val: &(Vec<Vec<f64>>, Vec<Vec<f64>>),
    seed: u64,
) -> TrainResult {
    let mut model = LinearModel::new(seed);
    let mut loss_curve = Vec::with_capacity(config.total_epochs);
    let mut best_val_loss = f64::INFINITY;
    let mut best_epoch = 0;

    for epoch in 0..config.total_epochs {
        let lr = get_lr(schedule, epoch, config);
        model.train_step(&train.0, &train.1, lr);
        let val_loss = model.mse_loss(&val.0, &val.1);
        loss_curve.push(val_loss);
        if val_loss < best_val_loss {
            best_val_loss = val_loss;
            best_epoch = epoch;
        }
    }

    let final_train_loss = model.mse_loss(&train.0, &train.1);
    let final_val_loss = *loss_curve.last().unwrap_or(&f64::INFINITY);

    TrainResult {
        schedule,
        loss_curve,
        final_train_loss,
        final_val_loss,
        best_val_loss,
        best_epoch,
    }
}
