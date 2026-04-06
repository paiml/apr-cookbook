//! Quantized Inference Comparison Example
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/int4-quantization-v1.yaml
//! Compares inference across precision levels: FP32, Q8, and Q4.
//! Measures accuracy degradation, latency improvements, and memory
//! savings for each quantization level.
//!
//! # Quantization Levels
//!
//! ```text
//! FP32: 32 bits/param → baseline accuracy, highest memory
//! Q8:    8 bits/param → ~0.1% accuracy loss, 4x compression
//! Q4:    4 bits/param → ~1-2% accuracy loss, 8x compression
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example quantized_inference_comparison
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr run model.apr          # APR native format
//! apr run model.gguf         # GGUF (llama.cpp compatible)
//! apr run model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Crankshaw, D. et al. (2017). *Clipper: A Low-Latency Online Prediction Serving System*. NSDI. arXiv:1612.03079

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::time::Instant;

const INPUT_DIM: usize = 64;
const HIDDEN_DIM: usize = 32;
const OUTPUT_DIM: usize = 8;

/// Quantization precision
#[derive(Clone, Copy)]
enum Precision {
    FP32,
    Q8,
    Q4,
}

impl Precision {
    fn name(self) -> &'static str {
        match self {
            Precision::FP32 => "FP32",
            Precision::Q8 => "INT8",
            Precision::Q4 => "INT4",
        }
    }

    fn bits(self) -> usize {
        match self {
            Precision::FP32 => 32,
            Precision::Q8 => 8,
            Precision::Q4 => 4,
        }
    }
}

/// Generate deterministic weights
fn generate_weights(size: usize, seed: u64) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let mut h = DefaultHasher::new();
            (seed, i).hash(&mut h);
            (h.finish() as f32 / u64::MAX as f32 - 0.5) * 0.2
        })
        .collect()
}

/// Quantize weights to given precision (simulate quantize → dequantize)
fn quantize(weights: &[f32], precision: Precision) -> Vec<f32> {
    match precision {
        Precision::FP32 => weights.to_vec(),
        Precision::Q8 => {
            let max = weights
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max)
                .max(1e-8);
            let scale = max / 127.0;
            weights
                .iter()
                .map(|&v| (v / scale).round().clamp(-128.0, 127.0) * scale)
                .collect()
        }
        Precision::Q4 => {
            let max = weights
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max)
                .max(1e-8);
            let scale = max / 7.0;
            weights
                .iter()
                .map(|&v| (v / scale).round().clamp(-8.0, 7.0) * scale)
                .collect()
        }
    }
}

/// Two-layer feedforward model
struct Model {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    precision: Precision,
}

impl Model {
    fn new(precision: Precision, seed: u64) -> Self {
        let w1_fp32 = generate_weights(HIDDEN_DIM * INPUT_DIM, seed);
        let w2_fp32 = generate_weights(OUTPUT_DIM * HIDDEN_DIM, seed + 1);
        Self {
            w1: quantize(&w1_fp32, precision),
            b1: vec![0.0; HIDDEN_DIM],
            w2: quantize(&w2_fp32, precision),
            b2: vec![0.0; OUTPUT_DIM],
            precision,
        }
    }

    fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Hidden layer + ReLU
        let mut hidden = self.b1.clone();
        for (o, h) in hidden.iter_mut().enumerate() {
            for (i, &x) in input.iter().enumerate() {
                *h += self.w1[o * INPUT_DIM + i] * x;
            }
            *h = h.max(0.0);
        }

        // Output layer
        let mut output = self.b2.clone();
        for (o, out) in output.iter_mut().enumerate() {
            for (i, &h) in hidden.iter().enumerate() {
                *out += self.w2[o * HIDDEN_DIM + i] * h;
            }
        }
        output
    }

    fn param_count(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()
    }

    fn memory_bytes(&self) -> usize {
        self.param_count() * self.precision.bits() / 8
    }
}

/// Generate test inputs
fn generate_inputs(n: usize, seed: u64) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            (0..INPUT_DIM)
                .map(|j| {
                    let mut h = DefaultHasher::new();
                    (seed, "input", i, j).hash(&mut h);
                    h.finish() as f32 / u64::MAX as f32 - 0.5
                })
                .collect()
        })
        .collect()
}

/// Compute output similarity (cosine similarity)
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-8 || norm_b < 1e-8 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Compute RMSE between two outputs
fn rmse(a: &[f32], b: &[f32]) -> f32 {
    let mse: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        / a.len() as f32;
    mse.sqrt()
}

/// Argmax of output
fn argmax(output: &[f32]) -> usize {
    output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map_or(0, |(i, _)| i)
}

fn main() {
    println!("=== Quantized Inference Comparison ===\n");

    let seed = 42;
    let test_inputs = generate_inputs(200, seed);

    // =========================================================================
    // Section 1: Memory Footprint
    // =========================================================================
    println!("1. Memory Footprint Comparison");
    println!("   ─────────────────────────────────────────");

    let fp32 = Model::new(Precision::FP32, seed);
    let q8 = Model::new(Precision::Q8, seed);
    let q4 = Model::new(Precision::Q4, seed);

    println!(
        "   {:>6} {:>10} {:>12} {:>12}",
        "Prec", "Params", "Memory", "Compression"
    );
    println!("   {}", "─".repeat(42));

    let fp32_mem = fp32.memory_bytes();
    for model in [&fp32, &q8, &q4] {
        let mem = model.memory_bytes();
        let compression = fp32_mem as f64 / mem as f64;
        println!(
            "   {:>6} {:>10} {:>10} B {:>11.1}x",
            model.precision.name(),
            model.param_count(),
            mem,
            compression
        );
    }
    println!();

    // =========================================================================
    // Section 2: Accuracy Comparison
    // =========================================================================
    println!("2. Accuracy vs FP32 Baseline");
    println!("   ─────────────────────────────────────────");

    let fp32_outputs: Vec<Vec<f32>> = test_inputs.iter().map(|x| fp32.forward(x)).collect();
    let q8_outputs: Vec<Vec<f32>> = test_inputs.iter().map(|x| q8.forward(x)).collect();
    let q4_outputs: Vec<Vec<f32>> = test_inputs.iter().map(|x| q4.forward(x)).collect();

    // Compute metrics
    let q8_cosine: f32 = fp32_outputs
        .iter()
        .zip(q8_outputs.iter())
        .map(|(a, b)| cosine_similarity(a, b))
        .sum::<f32>()
        / test_inputs.len() as f32;

    let q4_cosine: f32 = fp32_outputs
        .iter()
        .zip(q4_outputs.iter())
        .map(|(a, b)| cosine_similarity(a, b))
        .sum::<f32>()
        / test_inputs.len() as f32;

    let q8_rmse: f32 = fp32_outputs
        .iter()
        .zip(q8_outputs.iter())
        .map(|(a, b)| rmse(a, b))
        .sum::<f32>()
        / test_inputs.len() as f32;

    let q4_rmse: f32 = fp32_outputs
        .iter()
        .zip(q4_outputs.iter())
        .map(|(a, b)| rmse(a, b))
        .sum::<f32>()
        / test_inputs.len() as f32;

    // Argmax agreement
    let q8_agree: usize = fp32_outputs
        .iter()
        .zip(q8_outputs.iter())
        .filter(|(a, b)| argmax(a) == argmax(b))
        .count();

    let q4_agree: usize = fp32_outputs
        .iter()
        .zip(q4_outputs.iter())
        .filter(|(a, b)| argmax(a) == argmax(b))
        .count();

    println!(
        "   {:>6} {:>10} {:>10} {:>12}",
        "Prec", "Cosine", "RMSE", "ArgmaxMatch"
    );
    println!("   {}", "─".repeat(42));
    println!(
        "   {:>6} {:>10} {:>10} {:>12}",
        "FP32", "1.0000", "0.0000", "100.0%"
    );
    println!(
        "   {:>6} {:>10.4} {:>10.6} {:>11.1}%",
        "INT8",
        q8_cosine,
        q8_rmse,
        q8_agree as f64 / test_inputs.len() as f64 * 100.0
    );
    println!(
        "   {:>6} {:>10.4} {:>10.6} {:>11.1}%",
        "INT4",
        q4_cosine,
        q4_rmse,
        q4_agree as f64 / test_inputs.len() as f64 * 100.0
    );
    println!();

    // =========================================================================
    // Section 3: Latency Benchmark
    // =========================================================================
    println!("3. Latency Benchmark");
    println!("   ─────────────────────────────────────────");

    let n_iters = 2000;
    let bench_input = &test_inputs[0];

    println!(
        "   {:>6} {:>12} {:>12} {:>10}",
        "Prec", "Total(ms)", "Avg(us)", "Speedup"
    );
    println!("   {}", "─".repeat(42));

    let mut fp32_time = std::time::Duration::ZERO;
    for model in [&fp32, &q8, &q4] {
        let start = Instant::now();
        for _ in 0..n_iters {
            let _ = model.forward(bench_input);
        }
        let elapsed = start.elapsed();

        if matches!(model.precision, Precision::FP32) {
            fp32_time = elapsed;
        }

        let speedup = fp32_time.as_nanos() as f64 / elapsed.as_nanos().max(1) as f64;
        println!(
            "   {:>6} {:>12} {:>12.1} {:>9.2}x",
            model.precision.name(),
            elapsed.as_millis(),
            elapsed.as_micros() as f64 / f64::from(n_iters),
            speedup
        );
    }
    println!();

    // =========================================================================
    // Section 4: Weight Distribution Analysis
    // =========================================================================
    println!("4. Weight Distribution Analysis");
    println!("   ─────────────────────────────────────────");

    let fp32_w = &fp32.w1;
    let q8_w = &q8.w1;
    let q4_w = &q4.w1;

    for (name, weights) in [("FP32", fp32_w), ("INT8", q8_w), ("INT4", q4_w)] {
        let min = weights.iter().copied().fold(f32::INFINITY, f32::min);
        let max = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mean: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
        let var: f32 =
            weights.iter().map(|w| (w - mean).powi(2)).sum::<f32>() / weights.len() as f32;
        let unique: std::collections::HashSet<u32> = weights.iter().map(|w| w.to_bits()).collect();

        println!(
            "   {}: min={:.4}, max={:.4}, std={:.4}, unique={}",
            name,
            min,
            max,
            var.sqrt(),
            unique.len()
        );
    }
    println!();

    // =========================================================================
    // Section 5: Per-Output Accuracy
    // =========================================================================
    println!("5. Per-Output Dimension Accuracy");
    println!("   ─────────────────────────────────────────");
    println!("   {:>6} {:>8} {:>8}", "Dim", "Q8 RMSE", "Q4 RMSE");
    println!("   {}", "─".repeat(24));

    for dim in 0..OUTPUT_DIM {
        let q8_dim_rmse: f32 = fp32_outputs
            .iter()
            .zip(q8_outputs.iter())
            .map(|(a, b)| (a[dim] - b[dim]).powi(2))
            .sum::<f32>()
            / test_inputs.len() as f32;

        let q4_dim_rmse: f32 = fp32_outputs
            .iter()
            .zip(q4_outputs.iter())
            .map(|(a, b)| (a[dim] - b[dim]).powi(2))
            .sum::<f32>()
            / test_inputs.len() as f32;

        println!(
            "   {:>6} {:>8.5} {:>8.5}",
            dim,
            q8_dim_rmse.sqrt(),
            q4_dim_rmse.sqrt()
        );
    }
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_fp32_identity() {
        let w = generate_weights(100, 42);
        let q = quantize(&w, Precision::FP32);
        assert_eq!(w, q);
    }

    #[test]
    fn test_quantize_q8_bounded() {
        let w = generate_weights(100, 42);
        let q = quantize(&w, Precision::Q8);
        let err = rmse(&w, &q);
        assert!(err < 0.01, "Q8 RMSE {} too large", err);
    }

    #[test]
    fn test_quantize_q4_more_error() {
        let w = generate_weights(100, 42);
        let q8 = quantize(&w, Precision::Q8);
        let q4 = quantize(&w, Precision::Q4);
        assert!(rmse(&w, &q4) >= rmse(&w, &q8));
    }

    #[test]
    fn test_model_output_dimensions() {
        let model = Model::new(Precision::FP32, 42);
        let input = vec![0.5; INPUT_DIM];
        let output = model.forward(&input);
        assert_eq!(output.len(), OUTPUT_DIM);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_argmax() {
        assert_eq!(argmax(&[0.1, 0.5, 0.3]), 1);
        assert_eq!(argmax(&[0.9, 0.1, 0.0]), 0);
    }

    #[test]
    fn test_memory_compression() {
        let fp32 = Model::new(Precision::FP32, 42);
        let q4 = Model::new(Precision::Q4, 42);
        assert!(q4.memory_bytes() < fp32.memory_bytes());
    }
}
