//! Checkpoint Resume Training Example
//!
//! Demonstrates saving and restoring training state: model weights,
//! optimizer state, epoch counter, and loss history. Enables resuming
//! interrupted training from the last checkpoint.
//!
//! # Checkpoint Contents
//!
//! ```text
//! checkpoint.bin:
//!   ├── epoch: u32
//!   ├── best_val_loss: f32
//!   ├── weights: [f32; N]
//!   ├── loss_history: Vec<f32>
//!   └── rng_state: u64
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example checkpoint_resume
//! ```

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

const INPUT_DIM: usize = 16;
const OUTPUT_DIM: usize = 4;

/// Training checkpoint containing all resumable state
#[derive(Clone)]
struct Checkpoint {
    epoch: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
    loss_history: Vec<f32>,
    best_val_loss: f32,
    rng_state: u64,
}

impl Checkpoint {
    /// Serialize to bytes (simplified binary format)
    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        // Epoch
        bytes.extend_from_slice(&(self.epoch as u32).to_le_bytes());
        // Best val loss
        bytes.extend_from_slice(&self.best_val_loss.to_le_bytes());
        // RNG state
        bytes.extend_from_slice(&self.rng_state.to_le_bytes());
        // Weights
        bytes.extend_from_slice(&(self.weights.len() as u32).to_le_bytes());
        for &w in &self.weights {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        // Bias
        bytes.extend_from_slice(&(self.bias.len() as u32).to_le_bytes());
        for &b in &self.bias {
            bytes.extend_from_slice(&b.to_le_bytes());
        }
        // Loss history
        bytes.extend_from_slice(&(self.loss_history.len() as u32).to_le_bytes());
        for &l in &self.loss_history {
            bytes.extend_from_slice(&l.to_le_bytes());
        }
        bytes
    }

    /// Deserialize from bytes
    fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 16 {
            return None;
        }
        let mut pos = 0;

        let epoch = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        let best_val_loss = f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?);
        pos += 4;
        let rng_state = u64::from_le_bytes(data[pos..pos + 8].try_into().ok()?);
        pos += 8;

        let w_len = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        let mut weights = Vec::with_capacity(w_len);
        for _ in 0..w_len {
            weights.push(f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?));
            pos += 4;
        }

        let b_len = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        let mut bias = Vec::with_capacity(b_len);
        for _ in 0..b_len {
            bias.push(f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?));
            pos += 4;
        }

        let h_len = u32::from_le_bytes(data[pos..pos + 4].try_into().ok()?) as usize;
        pos += 4;
        let mut loss_history = Vec::with_capacity(h_len);
        for _ in 0..h_len {
            loss_history.push(f32::from_le_bytes(data[pos..pos + 4].try_into().ok()?));
            pos += 4;
        }

        Some(Self {
            epoch,
            weights,
            bias,
            loss_history,
            best_val_loss,
            rng_state,
        })
    }

    fn byte_size(&self) -> usize {
        16 + 4 + self.weights.len() * 4 + 4 + self.bias.len() * 4 + 4 + self.loss_history.len() * 4
    }
}

/// Simple model for training
struct TrainableModel {
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl TrainableModel {
    fn new(seed: u64) -> Self {
        let weights: Vec<f32> = (0..OUTPUT_DIM * INPUT_DIM)
            .map(|i| {
                let mut h = DefaultHasher::new();
                (seed, "w", i).hash(&mut h);
                (h.finish() as f32 / u64::MAX as f32 - 0.5) * 0.1
            })
            .collect();
        let bias = vec![0.0; OUTPUT_DIM];
        Self { weights, bias }
    }

    fn from_checkpoint(ckpt: &Checkpoint) -> Self {
        Self {
            weights: ckpt.weights.clone(),
            bias: ckpt.bias.clone(),
        }
    }

    fn to_checkpoint(&self, epoch: usize, losses: &[f32], best_val: f32, rng: u64) -> Checkpoint {
        Checkpoint {
            epoch,
            weights: self.weights.clone(),
            bias: self.bias.clone(),
            loss_history: losses.to_vec(),
            best_val_loss: best_val,
            rng_state: rng,
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut output = self.bias.clone();
        for (o, out) in output.iter_mut().enumerate() {
            for (i, &inp) in input.iter().enumerate() {
                *out += self.weights[o * INPUT_DIM + i] * inp;
            }
        }
        output
    }

    fn mse_loss(&self, inputs: &[Vec<f32>], targets: &[Vec<f32>]) -> f32 {
        let n = inputs.len() as f32;
        inputs
            .iter()
            .zip(targets.iter())
            .map(|(x, t)| {
                let pred = self.forward(x);
                pred.iter()
                    .zip(t.iter())
                    .map(|(p, y)| (p - y).powi(2))
                    .sum::<f32>()
            })
            .sum::<f32>()
            / n
    }

    fn train_step(&mut self, inputs: &[Vec<f32>], targets: &[Vec<f32>], lr: f32) {
        let eps = 1e-4;
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
    }
}

/// Generate synthetic data
fn generate_data(n: usize, seed: u64) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut inputs = Vec::with_capacity(n);
    let mut targets = Vec::with_capacity(n);
    for i in 0..n {
        let x: Vec<f32> = (0..INPUT_DIM)
            .map(|j| {
                let mut h = DefaultHasher::new();
                (seed, i, j).hash(&mut h);
                h.finish() as f32 / u64::MAX as f32 - 0.5
            })
            .collect();
        let y: Vec<f32> = (0..OUTPUT_DIM)
            .map(|k| {
                x.iter()
                    .enumerate()
                    .map(|(j, &v)| v * ((j + k) as f32 * 0.05).sin())
                    .sum()
            })
            .collect();
        inputs.push(x);
        targets.push(y);
    }
    (inputs, targets)
}

/// Train with checkpointing
fn train_with_checkpoints(
    model: &mut TrainableModel,
    train: &(Vec<Vec<f32>>, Vec<Vec<f32>>),
    val: &(Vec<Vec<f32>>, Vec<Vec<f32>>),
    start_epoch: usize,
    total_epochs: usize,
    lr: f32,
    checkpoint_interval: usize,
    existing_losses: &[f32],
) -> (Vec<Checkpoint>, Vec<f32>) {
    let mut checkpoints = Vec::new();
    let mut losses: Vec<f32> = existing_losses.to_vec();
    let mut best_val = losses.iter().copied().fold(f32::INFINITY, f32::min);

    for epoch in start_epoch..total_epochs {
        model.train_step(&train.0, &train.1, lr);
        let val_loss = model.mse_loss(&val.0, &val.1);
        losses.push(val_loss);

        if val_loss < best_val {
            best_val = val_loss;
        }

        if (epoch + 1) % checkpoint_interval == 0 || epoch == total_epochs - 1 {
            let ckpt = model.to_checkpoint(epoch + 1, &losses, best_val, 42 + epoch as u64);
            checkpoints.push(ckpt);
        }
    }

    (checkpoints, losses)
}

fn main() {
    println!("=== Checkpoint Resume Training Example ===\n");

    let train_data = generate_data(50, 42);
    let val_data = generate_data(15, 99);
    let lr = 0.01;

    // =========================================================================
    // Section 1: Training with Checkpoints
    // =========================================================================
    println!("1. Training with Periodic Checkpoints");
    println!("   ─────────────────────────────────────────");

    let mut model = TrainableModel::new(42);
    let initial_loss = model.mse_loss(&val_data.0, &val_data.1);
    println!("   Initial val loss: {:.6}", initial_loss);

    let (checkpoints, losses) =
        train_with_checkpoints(&mut model, &train_data, &val_data, 0, 30, lr, 5, &[]);

    println!("   Checkpoints saved: {}", checkpoints.len());
    for ckpt in &checkpoints {
        println!(
            "     Epoch {:3}: val_loss={:.6}, size={} bytes",
            ckpt.epoch,
            ckpt.best_val_loss,
            ckpt.byte_size()
        );
    }
    println!(
        "   Final val loss: {:.6}",
        losses.last().copied().unwrap_or(0.0)
    );
    println!();

    // =========================================================================
    // Section 2: Serialize and Deserialize
    // =========================================================================
    println!("2. Checkpoint Serialization");
    println!("   ─────────────────────────────────────────");

    let last_ckpt = checkpoints.last().unwrap();
    let bytes = last_ckpt.to_bytes();
    let restored = Checkpoint::from_bytes(&bytes).expect("valid checkpoint");

    println!("   Serialized:   {} bytes", bytes.len());
    println!("   Epoch:        {} → {}", last_ckpt.epoch, restored.epoch);
    println!(
        "   Best val:     {:.6} → {:.6}",
        last_ckpt.best_val_loss, restored.best_val_loss
    );
    println!(
        "   Weights match: {}",
        last_ckpt.weights == restored.weights
    );
    println!(
        "   History len:  {} → {}",
        last_ckpt.loss_history.len(),
        restored.loss_history.len()
    );
    println!();

    // =========================================================================
    // Section 3: Resume Training from Checkpoint
    // =========================================================================
    println!("3. Resume Training from Checkpoint");
    println!("   ─────────────────────────────────────────");

    // Simulate: train interrupted at epoch 15, resume to epoch 30
    let mut model_a = TrainableModel::new(42);
    let (ckpts_a, _) =
        train_with_checkpoints(&mut model_a, &train_data, &val_data, 0, 15, lr, 5, &[]);
    let mid_ckpt = ckpts_a.last().unwrap();

    println!(
        "   Phase 1: Trained epochs 0-15, val_loss={:.6}",
        mid_ckpt.best_val_loss
    );

    // Resume from checkpoint
    let mut model_b = TrainableModel::from_checkpoint(mid_ckpt);
    let (_, resumed_losses) = train_with_checkpoints(
        &mut model_b,
        &train_data,
        &val_data,
        mid_ckpt.epoch,
        30,
        lr,
        5,
        &mid_ckpt.loss_history,
    );

    println!(
        "   Phase 2: Resumed 15-30, val_loss={:.6}",
        resumed_losses.last().copied().unwrap_or(0.0)
    );

    // Compare with continuous training
    let mut model_c = TrainableModel::new(42);
    let (_, continuous_losses) =
        train_with_checkpoints(&mut model_c, &train_data, &val_data, 0, 30, lr, 5, &[]);

    println!(
        "   Continuous 0-30, val_loss={:.6}",
        continuous_losses.last().copied().unwrap_or(0.0)
    );
    println!();

    // =========================================================================
    // Section 4: Checkpoint Frequency Impact
    // =========================================================================
    println!("4. Checkpoint Frequency Impact");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>10} {:>12} {:>12} {:>14}",
        "Interval", "Checkpoints", "TotalBytes", "FinalLoss"
    );
    println!("   {}", "─".repeat(50));

    for interval in [1, 2, 5, 10, 20] {
        let mut model = TrainableModel::new(42);
        let (ckpts, losses) =
            train_with_checkpoints(&mut model, &train_data, &val_data, 0, 30, lr, interval, &[]);
        let total_bytes: usize = ckpts.iter().map(|c| c.to_bytes().len()).sum();
        println!(
            "   {:>10} {:>12} {:>10} B {:>14.6}",
            interval,
            ckpts.len(),
            total_bytes,
            losses.last().copied().unwrap_or(0.0)
        );
    }
    println!();

    // =========================================================================
    // Section 5: Best Checkpoint Selection
    // =========================================================================
    println!("5. Best Checkpoint Selection");
    println!("   ─────────────────────────────────────────");

    let mut model = TrainableModel::new(42);
    let (all_ckpts, _) =
        train_with_checkpoints(&mut model, &train_data, &val_data, 0, 30, lr, 1, &[]);

    let best_ckpt = all_ckpts
        .iter()
        .min_by(|a, b| a.best_val_loss.partial_cmp(&b.best_val_loss).unwrap())
        .unwrap();

    println!("   Total checkpoints: {}", all_ckpts.len());
    println!(
        "   Best at epoch {}:  val_loss={:.6}",
        best_ckpt.epoch, best_ckpt.best_val_loss
    );
    println!(
        "   Last at epoch {}:  val_loss={:.6}",
        all_ckpts.last().unwrap().epoch,
        all_ckpts.last().unwrap().best_val_loss
    );
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_roundtrip() {
        let model = TrainableModel::new(42);
        let ckpt = model.to_checkpoint(5, &[1.0, 0.5, 0.3], 0.3, 42);
        let bytes = ckpt.to_bytes();
        let restored = Checkpoint::from_bytes(&bytes).unwrap();
        assert_eq!(restored.epoch, 5);
        assert_eq!(restored.weights, ckpt.weights);
        assert_eq!(restored.bias, ckpt.bias);
        assert_eq!(restored.loss_history, vec![1.0, 0.5, 0.3]);
        assert!((restored.best_val_loss - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn test_checkpoint_invalid_bytes() {
        assert!(Checkpoint::from_bytes(&[0, 1, 2]).is_none());
    }

    #[test]
    fn test_model_from_checkpoint() {
        let model = TrainableModel::new(42);
        let ckpt = model.to_checkpoint(0, &[], f32::INFINITY, 42);
        let restored = TrainableModel::from_checkpoint(&ckpt);
        assert_eq!(model.weights, restored.weights);
    }

    #[test]
    fn test_training_reduces_loss() {
        let train = generate_data(30, 42);
        let val = generate_data(10, 99);
        let mut model = TrainableModel::new(42);
        let loss_before = model.mse_loss(&val.0, &val.1);
        let (_, losses) = train_with_checkpoints(&mut model, &train, &val, 0, 10, 0.01, 5, &[]);
        let loss_after = losses.last().copied().unwrap_or(f32::INFINITY);
        assert!(loss_after < loss_before, "{loss_after} < {loss_before}");
    }

    #[test]
    fn test_checkpoint_count() {
        let train = generate_data(20, 42);
        let val = generate_data(5, 99);
        let mut model = TrainableModel::new(42);
        let (ckpts, _) = train_with_checkpoints(&mut model, &train, &val, 0, 20, 0.01, 5, &[]);
        assert_eq!(ckpts.len(), 4); // epochs 5, 10, 15, 20
    }

    #[test]
    fn test_generate_data_shapes() {
        let (x, y) = generate_data(10, 42);
        assert_eq!(x.len(), 10);
        assert_eq!(y.len(), 10);
        assert_eq!(x[0].len(), INPUT_DIM);
        assert_eq!(y[0].len(), OUTPUT_DIM);
    }

    #[test]
    fn test_byte_size() {
        let model = TrainableModel::new(42);
        let ckpt = model.to_checkpoint(0, &[1.0, 2.0], f32::INFINITY, 42);
        let bytes = ckpt.to_bytes();
        assert_eq!(bytes.len(), ckpt.byte_size());
    }
}
