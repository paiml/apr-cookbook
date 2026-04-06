#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use rand::Rng;
use std::fmt;

// ============================================================================
// Constants
// ============================================================================

pub const DIM: usize = 1024;

// ============================================================================
// Data Structures
// ============================================================================

/// Quantization method for weight representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMethod {
    // Full 32-bit floating point (baseline).
    FP32,
    // Simulated 16-bit floating point (truncated mantissa).
    FP16,
    // 8-bit integer with affine scale + zero-point.
    INT8,
    // 4-bit integer packed 2 values per byte.
    INT4,
}

impl fmt::Display for QuantMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FP32 => write!(f, "FP32"),
            Self::FP16 => write!(f, "FP16"),
            Self::INT8 => write!(f, "INT8"),
            Self::INT4 => write!(f, "INT4"),
        }
    }
}

impl QuantMethod {
    /// Bytes per weight element for this method.
    pub fn bytes_per_weight(self) -> f64 {
        match self {
            Self::FP32 => 4.0,
            Self::FP16 => 2.0,
            Self::INT8 => 1.0,
            Self::INT4 => 0.5,
        }
    }
}

/// Quantized weight storage with dequantization parameters.
pub struct QuantizedWeights {
    // Quantization method used.
    pub method: QuantMethod,
    // Raw quantized bytes.
    pub data: Vec<u8>,
    // Scale factor for dequantization.
    pub scale: f64,
    // Zero-point offset for affine quantization.
    pub zero_point: i32,
    // Number of rows in the weight matrix.
    pub rows: usize,
    // Number of columns in the weight matrix.
    pub cols: usize,
}

/// Result of a quantized matmul operation.
pub struct MatmulResult {
    // Quantization method used.
    pub method: QuantMethod,
    // Output vector from the matmul.
    pub output: Vec<f64>,
    // Maximum absolute error vs FP32 baseline.
    pub max_error: f64,
    // Mean absolute error vs FP32 baseline.
    pub mean_error: f64,
    // Cosine similarity to FP32 baseline output.
    pub cosine_sim: f64,
    // Memory footprint of the weight matrix in bytes.
    pub memory_bytes: usize,
    // Estimated throughput in GB/s.
    pub throughput_est: f64,
}

// ============================================================================
// Quantization Functions
// ============================================================================

/// Quantize an f64 weight matrix into the specified format.
pub fn quantize_weights(
    weights: &[f64],
    rows: usize,
    cols: usize,
    method: QuantMethod,
) -> QuantizedWeights {
    match method {
        QuantMethod::FP32 => quantize_fp32(weights, rows, cols),
        QuantMethod::FP16 => quantize_fp16(weights, rows, cols),
        QuantMethod::INT8 => quantize_int8(weights, rows, cols),
        QuantMethod::INT4 => quantize_int4(weights, rows, cols),
    }
}

pub fn quantize_fp32(weights: &[f64], rows: usize, cols: usize) -> QuantizedWeights {
    let data: Vec<u8> = weights
        .iter()
        .flat_map(|&w| (w as f32).to_le_bytes())
        .collect();
    QuantizedWeights {
        method: QuantMethod::FP32,
        data,
        scale: 1.0,
        zero_point: 0,
        rows,
        cols,
    }
}

pub fn quantize_fp16(weights: &[f64], rows: usize, cols: usize) -> QuantizedWeights {
    // Simulate FP16 by storing upper 16 bits of truncated f32
    let data: Vec<u8> = weights
        .iter()
        .flat_map(|&w| {
            let bits = (w as f32).to_bits();
            let truncated = bits & 0xFFFF_E000;
            // Store as 2 bytes (upper half of f32 bits)
            let upper = (truncated >> 16) as u16;
            upper.to_le_bytes()
        })
        .collect();
    QuantizedWeights {
        method: QuantMethod::FP16,
        data,
        scale: 1.0,
        zero_point: 0,
        rows,
        cols,
    }
}

pub fn quantize_int8(weights: &[f64], rows: usize, cols: usize) -> QuantizedWeights {
    let min_val = weights.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (max_val - min_val).max(1e-10);
    let scale = range / 255.0;
    let zero_point = (-min_val / scale).round() as i32;

    let data: Vec<u8> = weights
        .iter()
        .map(|&w| {
            let quantized = (w / scale + f64::from(zero_point)).round();
            quantized.clamp(0.0, 255.0) as u8
        })
        .collect();

    QuantizedWeights {
        method: QuantMethod::INT8,
        data,
        scale,
        zero_point,
        rows,
        cols,
    }
}

pub fn quantize_int4(weights: &[f64], rows: usize, cols: usize) -> QuantizedWeights {
    let min_val = weights.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (max_val - min_val).max(1e-10);
    let scale = range / 15.0;
    let zero_point = (-min_val / scale).round() as i32;

    // Pack 2 INT4 values per byte (low nibble first, high nibble second)
    let n = weights.len();
    let packed_len = n.div_ceil(2);
    let mut data = vec![0u8; packed_len];

    for (i, &w) in weights.iter().enumerate() {
        let quantized = (w / scale + f64::from(zero_point)).round();
        let nibble = quantized.clamp(0.0, 15.0) as u8;
        let byte_idx = i / 2;
        if i % 2 == 0 {
            data[byte_idx] |= nibble;
        } else {
            data[byte_idx] |= nibble << 4;
        }
    }

    QuantizedWeights {
        method: QuantMethod::INT4,
        data,
        scale,
        zero_point,
        rows,
        cols,
    }
}

// ============================================================================
// Dequantization + Matmul
// ============================================================================

/// Dequantize a single weight from the quantized storage.
pub fn dequantize_weight(qw: &QuantizedWeights, index: usize) -> f64 {
    match qw.method {
        QuantMethod::FP32 => {
            let offset = index * 4;
            let bytes = [
                qw.data[offset],
                qw.data[offset + 1],
                qw.data[offset + 2],
                qw.data[offset + 3],
            ];
            f64::from(f32::from_le_bytes(bytes))
        }
        QuantMethod::FP16 => {
            let offset = index * 2;
            let upper = u16::from_le_bytes([qw.data[offset], qw.data[offset + 1]]);
            let bits = u32::from(upper) << 16;
            f64::from(f32::from_bits(bits))
        }
        QuantMethod::INT8 => {
            let val = i32::from(qw.data[index]);
            f64::from(val - qw.zero_point) * qw.scale
        }
        QuantMethod::INT4 => {
            let byte_idx = index / 2;
            let nibble = if index % 2 == 0 {
                qw.data[byte_idx] & 0x0F
            } else {
                (qw.data[byte_idx] >> 4) & 0x0F
            };
            f64::from(i32::from(nibble) - qw.zero_point) * qw.scale
        }
    }
}

// Perform matrix-vector multiplication: output = W * input.
//
/// W is `rows x cols`, input is `cols`, output is `rows`.
pub fn quantized_matmul(qw: &QuantizedWeights, input: &[f64]) -> Vec<f64> {
    let mut output = vec![0.0; qw.rows];
    for (r, out) in output.iter_mut().enumerate() {
        let mut acc: f64 = 0.0;
        for (c, &inp) in input.iter().enumerate().take(qw.cols) {
            let w = dequantize_weight(qw, r * qw.cols + c);
            acc += w * inp;
        }
        *out = acc;
    }
    output
}

// ============================================================================
// Metrics
// ============================================================================

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    let denom = norm_a * norm_b;
    if denom < 1e-15 {
        return 0.0;
    }
    dot / denom
}

/// Compute max absolute error between two vectors.
pub fn max_abs_error(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Compute mean absolute error between two vectors.
pub fn mean_abs_error(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let sum: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
    sum / a.len() as f64
}

// ============================================================================
// Pipeline
// ============================================================================

/// Run quantized matmul for a given method and compare against baseline.
pub fn run_matmul_variant(
    weights: &[f64],
    input: &[f64],
    baseline: &[f64],
    method: QuantMethod,
) -> MatmulResult {
    let qw = quantize_weights(weights, DIM, DIM, method);
    let memory_bytes = qw.data.len();
    let output = quantized_matmul(&qw, input);

    let max_error = max_abs_error(&output, baseline);
    let mean_error = mean_abs_error(&output, baseline);
    let cosine_sim = cosine_similarity(&output, baseline);

    // Estimate throughput: assume weight-bandwidth-bound at 100 GB/s memory BW
    // Effective bytes = weight_elements * bytes_per_weight + input bytes
    let weight_bytes = (DIM * DIM) as f64 * method.bytes_per_weight();
    let input_bytes = DIM as f64 * 8.0;
    let bytes_transferred = weight_bytes + input_bytes;
    let assumed_bandwidth_gbs = 100.0;
    let time_seconds = bytes_transferred / (assumed_bandwidth_gbs * 1e9);
    let throughput_est = if time_seconds > 0.0 {
        bytes_transferred / time_seconds / 1e9
    } else {
        0.0
    };

    MatmulResult {
        method,
        output,
        max_error,
        mean_error,
        cosine_sim,
        memory_bytes,
        throughput_est,
    }
}

/// Generate the full comparison across all methods.
pub fn run_comparison(weights: &[f64], input: &[f64]) -> Vec<MatmulResult> {
    let methods = [
        QuantMethod::FP32,
        QuantMethod::FP16,
        QuantMethod::INT8,
        QuantMethod::INT4,
    ];

    // Compute FP32 baseline first
    let baseline_qw = quantize_weights(weights, DIM, DIM, QuantMethod::FP32);
    let baseline_output = quantized_matmul(&baseline_qw, input);

    methods
        .iter()
        .map(|&method| run_matmul_variant(weights, input, &baseline_output, method))
        .collect()
}

// ============================================================================
// Display
// ============================================================================

pub fn print_comparison_table(results: &[MatmulResult]) {
    println!(
        "  {:>6}  {:>10}  {:>12}  {:>12}  {:>10}",
        "Method", "Memory_MB", "Max_Error", "Cosine_Sim", "TP_GB/s"
    );
    println!("  {}", "-".repeat(58));
    for r in results {
        println!(
            "  {:>6}  {:>10.4}  {:>12.6e}  {:>12.10}  {:>10.1}",
            r.method,
            r.memory_bytes as f64 / (1024.0 * 1024.0),
            r.max_error,
            r.cosine_sim,
            r.throughput_est,
        );
    }
}

pub fn print_tradeoff_curve(results: &[MatmulResult]) {
    println!("\n  Accuracy vs Compression Tradeoff");
    println!("  {}", "-".repeat(50));
    let fp32_mem = results
        .iter()
        .find(|r| r.method == QuantMethod::FP32)
        .map_or(1, |r| r.memory_bytes);

    for r in results {
        let compression = fp32_mem as f64 / r.memory_bytes.max(1) as f64;
        let bar_len = (r.cosine_sim * 40.0).round() as usize;
        let bar: String = "|".repeat(bar_len.min(40));
        println!(
            "  {:>6} ({:.1}x) [{:<40}] cos={:.6}",
            r.method, compression, bar, r.cosine_sim,
        );
    }
}

// ============================================================================
// Main
// ============================================================================
