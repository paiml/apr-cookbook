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
use std::time::Instant;

pub const INPUT_DIM: usize = 16;
pub const HIDDEN_DIM: usize = 8;
pub const OUTPUT_DIM: usize = 4;

/// Numerical precision for training
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    FP32,
    FP16,
    BF16,
}

impl Precision {
    pub fn name(self) -> &'static str {
        match self {
            Precision::FP32 => "FP32",
            Precision::FP16 => "FP16",
            Precision::BF16 => "BF16",
        }
    }

    pub fn bits(self) -> usize {
        match self {
            Precision::FP32 => 32,
            Precision::FP16 | Precision::BF16 => 16,
        }
    }

    /// Simulate reduced precision by rounding to fewer mantissa bits
    pub fn cast(self, value: f32) -> f32 {
        match self {
            Precision::FP32 => value,
            Precision::FP16 => {
                // FP16: 10-bit mantissa, range ±65504
                let clamped = value.clamp(-65504.0, 65504.0);
                let bits = clamped.to_bits();
                // Zero out lower 13 bits of mantissa (23 - 10 = 13)
                let rounded = bits & 0xFFFF_E000;
                f32::from_bits(rounded)
            }
            Precision::BF16 => {
                // BF16: 7-bit mantissa, same exponent range as FP32
                let bits = value.to_bits();
                // Zero out lower 16 bits of mantissa (23 - 7 = 16)
                let rounded = bits & 0xFFFF_0000;
                f32::from_bits(rounded)
            }
        }
    }
}

/// Deterministic weight initialization
pub fn init_weights(size: usize, seed: u64) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let mut h = DefaultHasher::new();
            (seed, i).hash(&mut h);
            (h.finish() as f32 / u64::MAX as f32 - 0.5) * 0.2
        })
        .collect()
}

/// Two-layer model with mixed-precision support
pub struct MixedPrecisionModel {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
    pub precision: Precision,
}

impl MixedPrecisionModel {
    pub fn new(precision: Precision, seed: u64) -> Self {
        Self {
            w1: init_weights(HIDDEN_DIM * INPUT_DIM, seed),
            b1: vec![0.0; HIDDEN_DIM],
            w2: init_weights(OUTPUT_DIM * HIDDEN_DIM, seed + 1),
            b2: vec![0.0; OUTPUT_DIM],
            precision,
        }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let p = self.precision;

        // Hidden layer + ReLU
        let mut hidden = self.b1.clone();
        for (o, h) in hidden.iter_mut().enumerate() {
            for (i, &x) in input.iter().enumerate() {
                *h += p.cast(self.w1[o * INPUT_DIM + i]) * p.cast(x);
            }
            *h = p.cast(h.max(0.0));
        }

        // Output layer
        let mut output = self.b2.clone();
        for (o, out) in output.iter_mut().enumerate() {
            for (i, &h) in hidden.iter().enumerate() {
                *out += p.cast(self.w2[o * HIDDEN_DIM + i]) * p.cast(h);
            }
            *out = p.cast(*out);
        }
        output
    }

    pub fn param_count(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }
}

/// Loss scaling configuration for FP16 training
pub struct LossScaler {
    pub scale: f32,
    pub growth_factor: f32,
    pub backoff_factor: f32,
    pub growth_interval: usize,
    pub steps_since_growth: usize,
    pub overflow_count: usize,
}

impl LossScaler {
    pub fn new(initial_scale: f32) -> Self {
        Self {
            scale: initial_scale,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 100,
            steps_since_growth: 0,
            overflow_count: 0,
        }
    }

    pub fn scale_loss(&self, loss: f32) -> f32 {
        loss * self.scale
    }

    pub fn unscale_grad(&self, grad: f32) -> f32 {
        grad / self.scale
    }

    pub fn update(&mut self, has_overflow: bool) {
        if has_overflow {
            self.scale *= self.backoff_factor;
            self.steps_since_growth = 0;
            self.overflow_count += 1;
        } else {
            self.steps_since_growth += 1;
            if self.steps_since_growth >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.steps_since_growth = 0;
            }
        }
    }
}

/// Generate labeled training data
pub fn generate_data(n: usize, seed: u64) -> Vec<(Vec<f32>, usize)> {
    (0..n)
        .map(|i| {
            let input: Vec<f32> = (0..INPUT_DIM)
                .map(|j| {
                    let mut h = DefaultHasher::new();
                    (seed, "data", i, j).hash(&mut h);
                    h.finish() as f32 / u64::MAX as f32 - 0.5
                })
                .collect();
            let mut h = DefaultHasher::new();
            (seed, "label", i).hash(&mut h);
            let label = h.finish() as usize % OUTPUT_DIM;
            (input, label)
        })
        .collect()
}

/// Compute cross-entropy loss
pub fn cross_entropy(output: &[f32], target: usize) -> f32 {
    let max = output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps: Vec<f32> = output.iter().map(|&o| (o - max).exp()).collect();
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

/// Train one epoch and return (avg_loss, accuracy, duration_us)
pub fn train_epoch(
    model: &mut MixedPrecisionModel,
    data: &[(Vec<f32>, usize)],
    lr: f32,
    scaler: &mut Option<LossScaler>,
) -> (f32, f32, u128) {
    let start = Instant::now();
    let mut total_loss = 0.0f32;
    let mut correct = 0usize;

    for (input, target) in data {
        let output = model.forward(input);
        let loss = cross_entropy(&output, *target);
        total_loss += loss;

        if predict(&output) == *target {
            correct += 1;
        }

        // Apply loss scaling if present (FP16 stability)
        let scaled_loss = match scaler.as_ref() {
            Some(s) => s.scale_loss(loss),
            None => loss,
        };

        let grad_scale = match scaler.as_ref() {
            Some(s) => s.unscale_grad(scaled_loss),
            None => scaled_loss,
        };

        // Check for overflow in FP16 mode
        let has_overflow = !grad_scale.is_finite();

        if let Some(s) = scaler.as_mut() {
            s.update(has_overflow);
        }

        if !has_overflow {
            // Update output layer weights
            let effective_lr = lr * grad_scale.min(1.0);
            for o in 0..OUTPUT_DIM {
                let target_grad = if o == *target {
                    -effective_lr
                } else {
                    effective_lr
                };
                for h in 0..HIDDEN_DIM {
                    model.w2[o * HIDDEN_DIM + h] += target_grad * 0.01;
                }
            }
        }
    }

    let elapsed = start.elapsed().as_micros();
    let avg_loss = total_loss / data.len() as f32;
    let accuracy = correct as f32 / data.len() as f32;
    (avg_loss, accuracy, elapsed)
}

/// Section 1: Compare memory footprint across precision levels
pub fn section_precision_memory(seed: u64) {
    println!("1. Precision Memory Comparison");
    println!("   ─────────────────────────────────────────");
    println!(
        "   {:>6} {:>8} {:>12} {:>10}",
        "Prec", "Bits", "Params", "Memory"
    );
    println!("   {}", "─".repeat(40));

    for prec in [Precision::FP32, Precision::FP16, Precision::BF16] {
        let model = MixedPrecisionModel::new(prec, seed);
        let mem = model.param_count() * prec.bits() / 8;
        println!(
            "   {:>6} {:>8} {:>12} {:>8} B",
            prec.name(),
            prec.bits(),
            model.param_count(),
            mem
        );
    }
    println!();
}

/// Section 2: Show how casting affects numeric values at each precision
pub fn section_precision_casting() {
    println!("2. Precision Casting Effects");
    println!("   ─────────────────────────────────────────");

    let test_values = [0.1, 0.001, 0.000_01, 1.5, 100.0, 0.123_456_79];
    println!(
        "   {:>12} {:>12} {:>12} {:>12}",
        "Value", "FP32", "FP16", "BF16"
    );
    println!("   {}", "─".repeat(52));

    for &v in &test_values {
        println!(
            "   {:>12.9} {:>12.9} {:>12.9} {:>12.9}",
            v,
            Precision::FP32.cast(v),
            Precision::FP16.cast(v),
            Precision::BF16.cast(v)
        );
    }
    println!();
}

/// Section 3: Train each precision variant and report loss, accuracy, scaler state
pub fn section_training_loop(
    seed: u64,
    train_data: &[(Vec<f32>, usize)],
    test_data: &[(Vec<f32>, usize)],
) {
    println!("3. Training Comparison (5 epochs)");
    println!("   ─────────────────────────────────────────");

    let n_epochs = 5;
    let lr = 0.01;

    for prec in [Precision::FP32, Precision::FP16, Precision::BF16] {
        println!("   --- {} ---", prec.name());
        let mut model = MixedPrecisionModel::new(prec, seed);
        let mut scaler = if prec == Precision::FP16 {
            Some(LossScaler::new(1024.0))
        } else {
            None
        };

        for epoch in 0..n_epochs {
            let (loss, acc, us) = train_epoch(&mut model, train_data, lr, &mut scaler);
            if epoch == 0 || epoch == n_epochs - 1 {
                println!(
                    "   Epoch {}: loss={:.4}, acc={:.1}%, time={}us",
                    epoch,
                    loss,
                    acc * 100.0,
                    us
                );
            }
        }

        // Test accuracy
        let correct = test_data
            .iter()
            .filter(|(input, target)| predict(&model.forward(input)) == *target)
            .count();
        println!(
            "   Test accuracy: {:.1}%",
            f64::from(correct as u32) / test_data.len() as f64 * 100.0
        );

        if let Some(ref scaler) = scaler {
            println!(
                "   Loss scaler: scale={:.0}, overflows={}",
                scaler.scale, scaler.overflow_count
            );
        }
        println!();
    }
}

/// Section 4: Benchmark forward-pass throughput per precision
pub fn section_throughput_benchmark(seed: u64, train_data: &[(Vec<f32>, usize)]) {
    println!("4. Throughput Benchmark");
    println!("   ─────────────────────────────────────────");

    let n_iters: u32 = 500;
    println!(
        "   {:>6} {:>12} {:>14} {:>10}",
        "Prec", "Total(us)", "Samples/sec", "Speedup"
    );
    println!("   {}", "─".repeat(46));

    let mut fp32_time = 0u128;
    for prec in [Precision::FP32, Precision::FP16, Precision::BF16] {
        let model = MixedPrecisionModel::new(prec, seed);
        let input = &train_data[0].0;

        let start = Instant::now();
        for _ in 0..n_iters {
            let _ = model.forward(input);
        }
        let elapsed = start.elapsed().as_micros();

        if prec == Precision::FP32 {
            fp32_time = elapsed;
        }

        let samples_per_sec = f64::from(n_iters) / (elapsed as f64 / 1_000_000.0);
        let speedup = fp32_time as f64 / elapsed.max(1) as f64;
        println!(
            "   {:>6} {:>12} {:>14.0} {:>9.2}x",
            prec.name(),
            elapsed,
            samples_per_sec,
            speedup
        );
    }
    println!();
}

/// Section 5: Demonstrate loss scaler growth, backoff, and recovery
pub fn section_loss_scaler_dynamics() {
    println!("5. Loss Scaler Dynamics");
    println!("   ─────────────────────────────────────────");

    let mut scaler = LossScaler::new(1024.0);
    println!("   Initial scale: {:.0}", scaler.scale);

    // Simulate normal steps
    for _ in 0..100 {
        scaler.update(false);
    }
    println!("   After 100 normal steps: {:.0}", scaler.scale);

    // Simulate overflow
    scaler.update(true);
    println!("   After overflow: {:.0}", scaler.scale);

    // Recover
    for _ in 0..100 {
        scaler.update(false);
    }
    println!("   After 100 more steps: {:.0}", scaler.scale);
    println!("   Total overflows: {}", scaler.overflow_count);
    println!();
}
