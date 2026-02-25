//! # Recipe: Accelerated Quantized Matrix Multiplication
//!
//! **Category**: Acceleration
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## Learning Objective
//! Demonstrate how INT8 and INT4 quantized matrix multiplication reduces memory
//! bandwidth while preserving inference accuracy. Compares FP32 baseline, FP16
//! simulated, INT8 (scale + zero-point), and INT4 (packed 2-per-byte) approaches.
//!
//! ## Run Command
//! ```bash
//! cargo run --example acceleration_quantized_matmul --release
//! ```
//!
//! ## Toyota Way Principles
//! - **Muda** (Waste elimination): 4-8x memory reduction via quantization
//! - **Jidoka** (Quality built-in): Error metrics validate precision tradeoff
//! - **Genchi Genbutsu** (Go and see): Concrete throughput numbers per method

use apr_cookbook::prelude::*;
use rand::Rng;
use std::fmt;

// ============================================================================
// Constants
// ============================================================================

const DIM: usize = 1024;

// ============================================================================
// Data Structures
// ============================================================================

/// Quantization method for weight representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantMethod {
    /// Full 32-bit floating point (baseline).
    FP32,
    /// Simulated 16-bit floating point (truncated mantissa).
    FP16,
    /// 8-bit integer with affine scale + zero-point.
    INT8,
    /// 4-bit integer packed 2 values per byte.
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
    /// Quantization method used.
    pub method: QuantMethod,
    /// Raw quantized bytes.
    pub data: Vec<u8>,
    /// Scale factor for dequantization.
    pub scale: f64,
    /// Zero-point offset for affine quantization.
    pub zero_point: i32,
    /// Number of rows in the weight matrix.
    pub rows: usize,
    /// Number of columns in the weight matrix.
    pub cols: usize,
}

/// Result of a quantized matmul operation.
pub struct MatmulResult {
    /// Quantization method used.
    pub method: QuantMethod,
    /// Output vector from the matmul.
    pub output: Vec<f64>,
    /// Maximum absolute error vs FP32 baseline.
    pub max_error: f64,
    /// Mean absolute error vs FP32 baseline.
    pub mean_error: f64,
    /// Cosine similarity to FP32 baseline output.
    pub cosine_sim: f64,
    /// Memory footprint of the weight matrix in bytes.
    pub memory_bytes: usize,
    /// Estimated throughput in GB/s.
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

fn quantize_fp32(weights: &[f64], rows: usize, cols: usize) -> QuantizedWeights {
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

fn quantize_fp16(weights: &[f64], rows: usize, cols: usize) -> QuantizedWeights {
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

fn quantize_int8(weights: &[f64], rows: usize, cols: usize) -> QuantizedWeights {
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

fn quantize_int4(weights: &[f64], rows: usize, cols: usize) -> QuantizedWeights {
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
fn dequantize_weight(qw: &QuantizedWeights, index: usize) -> f64 {
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

/// Perform matrix-vector multiplication: output = W * input.
///
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

fn print_comparison_table(results: &[MatmulResult]) {
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

fn print_tradeoff_curve(results: &[MatmulResult]) {
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

fn main() -> Result<()> {
    println!("=== APR Cookbook: Accelerated Quantized Matrix Multiplication ===\n");

    let mut ctx = RecipeContext::new("acceleration_quantized_matmul")?;

    // Generate deterministic weights (1024x1024) and input (1024)
    println!("1. Generating synthetic data");
    println!("   Weight matrix: {}x{}", DIM, DIM);
    println!("   Input vector:  {}\n", DIM);

    let weights: Vec<f64> = (0..DIM * DIM)
        .map(|_| ctx.rng().gen_range(-1.0..1.0))
        .collect();
    let input: Vec<f64> = (0..DIM).map(|_| ctx.rng().gen_range(-1.0..1.0)).collect();

    // Run all variants
    println!("2. Running quantized matmul variants\n");
    let results = run_comparison(&weights, &input);

    // Print comparison table
    println!("3. Comparison Table\n");
    print_comparison_table(&results);

    // Print tradeoff curve
    println!("\n4. Accuracy vs Compression\n");
    print_tradeoff_curve(&results);

    // Record metrics
    for r in &results {
        let prefix = format!("{}", r.method);
        ctx.record_metric(&format!("{}_memory_bytes", prefix), r.memory_bytes as i64);
        ctx.record_float_metric(&format!("{}_cosine_sim", prefix), r.cosine_sim);
        ctx.record_float_metric(&format!("{}_max_error", prefix), r.max_error);
        ctx.record_float_metric(&format!("{}_mean_error", prefix), r.mean_error);
    }

    println!();
    ctx.report()?;

    println!("\n[SUCCESS] Quantized matmul comparison complete.");
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx() -> RecipeContext {
        RecipeContext::new("test_quantized_matmul").expect("context creation must succeed")
    }

    fn make_small_data(ctx: &mut RecipeContext) -> (Vec<f64>, Vec<f64>) {
        let weights: Vec<f64> = (0..64).map(|_| ctx.rng().gen_range(-1.0..1.0)).collect();
        let input: Vec<f64> = (0..8).map(|_| ctx.rng().gen_range(-1.0..1.0)).collect();
        (weights, input)
    }

    #[test]
    fn test_fp32_roundtrip_exact() {
        let mut ctx = make_ctx();
        let (weights, _) = make_small_data(&mut ctx);
        let qw = quantize_weights(&weights, 8, 8, QuantMethod::FP32);
        for (i, &original) in weights.iter().enumerate() {
            let restored = dequantize_weight(&qw, i);
            // FP32 roundtrip loses f64 precision but f32 must match
            assert!(
                (original as f32 - restored as f32).abs() < 1e-7,
                "FP32 roundtrip mismatch at index {}: {} vs {}",
                i,
                original,
                restored,
            );
        }
    }

    #[test]
    fn test_int8_quantization_bounded_error() {
        let mut ctx = make_ctx();
        let (weights, _) = make_small_data(&mut ctx);
        let qw = quantize_weights(&weights, 8, 8, QuantMethod::INT8);
        for (i, &original) in weights.iter().enumerate() {
            let restored = dequantize_weight(&qw, i);
            let err = (original - restored).abs();
            // INT8 with 256 levels over range [-1,1] gives step ~0.008
            assert!(
                err < 0.02,
                "INT8 error {} at index {} too large (original={}, restored={})",
                err,
                i,
                original,
                restored,
            );
        }
    }

    #[test]
    fn test_int4_quantization_bounded_error() {
        let mut ctx = make_ctx();
        let (weights, _) = make_small_data(&mut ctx);
        let qw = quantize_weights(&weights, 8, 8, QuantMethod::INT4);
        for (i, &original) in weights.iter().enumerate() {
            let restored = dequantize_weight(&qw, i);
            let err = (original - restored).abs();
            // INT4 with 16 levels over range [-1,1] gives step ~0.133
            assert!(
                err < 0.2,
                "INT4 error {} at index {} too large (original={}, restored={})",
                err,
                i,
                original,
                restored,
            );
        }
    }

    #[test]
    fn test_int4_packing_roundtrip() {
        // Verify 2-per-byte packing/unpacking is correct
        let weights = vec![0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.0, 0.8];
        let qw = quantize_weights(&weights, 2, 4, QuantMethod::INT4);
        // Packed length should be ceil(8/2) = 4 bytes
        assert_eq!(qw.data.len(), 4, "INT4 should pack 2 values per byte");
        // Dequantized values should be close to originals
        for (i, &original) in weights.iter().enumerate() {
            let restored = dequantize_weight(&qw, i);
            assert!(
                (original - restored).abs() < 0.2,
                "INT4 pack roundtrip failed at {}: {} vs {}",
                i,
                original,
                restored,
            );
        }
    }

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&a, &a);
        assert!(
            (sim - 1.0).abs() < 1e-10,
            "Identical vectors should have cosine similarity 1.0, got {}",
            sim,
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-10,
            "Orthogonal vectors should have cosine similarity 0.0, got {}",
            sim,
        );
    }

    #[test]
    fn test_memory_footprint_ordering() {
        let mut ctx = make_ctx();
        let weights: Vec<f64> = (0..DIM * DIM)
            .map(|_| ctx.rng().gen_range(-1.0..1.0))
            .collect();
        let fp32 = quantize_weights(&weights, DIM, DIM, QuantMethod::FP32);
        let fp16 = quantize_weights(&weights, DIM, DIM, QuantMethod::FP16);
        let int8 = quantize_weights(&weights, DIM, DIM, QuantMethod::INT8);
        let int4 = quantize_weights(&weights, DIM, DIM, QuantMethod::INT4);
        assert!(
            fp32.data.len() > fp16.data.len(),
            "FP32 ({}) should use more memory than FP16 ({})",
            fp32.data.len(),
            fp16.data.len(),
        );
        assert!(
            fp16.data.len() > int8.data.len(),
            "FP16 ({}) should use more memory than INT8 ({})",
            fp16.data.len(),
            int8.data.len(),
        );
        assert!(
            int8.data.len() > int4.data.len(),
            "INT8 ({}) should use more memory than INT4 ({})",
            int8.data.len(),
            int4.data.len(),
        );
    }

    #[test]
    fn test_matmul_output_length() {
        let mut ctx = make_ctx();
        let (weights, input) = make_small_data(&mut ctx);
        let qw = quantize_weights(&weights, 8, 8, QuantMethod::FP32);
        let output = quantized_matmul(&qw, &input);
        assert_eq!(
            output.len(),
            8,
            "Output should have {} rows, got {}",
            8,
            output.len(),
        );
    }

    #[test]
    fn test_error_increases_with_compression() {
        let mut ctx = make_ctx();
        let weights: Vec<f64> = (0..64).map(|_| ctx.rng().gen_range(-1.0..1.0)).collect();
        let input: Vec<f64> = (0..8).map(|_| ctx.rng().gen_range(-1.0..1.0)).collect();

        let fp32_qw = quantize_weights(&weights, 8, 8, QuantMethod::FP32);
        let baseline = quantized_matmul(&fp32_qw, &input);

        let int8_qw = quantize_weights(&weights, 8, 8, QuantMethod::INT8);
        let int8_out = quantized_matmul(&int8_qw, &input);
        let int8_err = max_abs_error(&int8_out, &baseline);

        let int4_qw = quantize_weights(&weights, 8, 8, QuantMethod::INT4);
        let int4_out = quantized_matmul(&int4_qw, &input);
        let int4_err = max_abs_error(&int4_out, &baseline);

        assert!(
            int4_err >= int8_err,
            "INT4 error ({}) should be >= INT8 error ({})",
            int4_err,
            int8_err,
        );
    }

    #[test]
    fn test_quant_method_display() {
        assert_eq!(format!("{}", QuantMethod::FP32), "FP32");
        assert_eq!(format!("{}", QuantMethod::FP16), "FP16");
        assert_eq!(format!("{}", QuantMethod::INT8), "INT8");
        assert_eq!(format!("{}", QuantMethod::INT4), "INT4");
    }
}
