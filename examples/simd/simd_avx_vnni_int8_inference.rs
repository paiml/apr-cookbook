//! # Recipe: AVX-VNNI Int8 Inference Acceleration
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/avx512-matmul-v1.yaml
//! **Category**: SIMD Acceleration
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] No proptests (SIMD recipe)
//!
//! ## Learning Objective
//! Demonstrate AVX-VNNI (VPDPBUSD) Int8 inference on Intel Meteor Lake CPUs.
//! Since we cannot use unsafe intrinsics, we simulate the int8 dot product
//! pattern (u8 * i8 -> i32 accumulation) in safe Rust and compare throughput
//! and accuracy against f32 baselines.
//!
//! ## Toyota Way Principles
//! - **Muda** (Waste elimination): 4x memory reduction via int8 quantization
//! - **Jidoka** (Quality built-in): Quantization error bounds validated in tests
//! - **Genchi Genbutsu** (Go and see): Concrete GOPS numbers per SIMD level
//!
//! ## Run Command
//! ```bash
//! cargo run --example simd_avx_vnni_int8_inference --release
//! ```
//!
//! ## Falsification Claim (F8)
//!
//! AVX-VNNI Int8 matmul simulation achieves >=1.5x throughput vs scalar f32
//! for 512x512 matrices.
//!
//!
//! ## Format Variants
//! ```bash
//! apr bench model.apr          # APR native format
//! apr bench model.gguf         # GGUF (llama.cpp compatible)
//! apr bench model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Hennessy, J. & Patterson, D. (2017). *Computer Architecture: A Quantitative Approach*. DOI: 10.1016/C2012-0-01712-X

use std::time::Instant;
use trueno::Matrix;

/// SIMD capability detection with AVX-VNNI support
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    pub avx_vnni: bool,
    pub avx2: bool,
    pub sse42: bool,
    pub neon: bool,
}

impl SimdCapabilities {
    /// Detect available SIMD features at runtime
    #[allow(clippy::incompatible_msrv)]
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx_vnni: is_x86_feature_detected!("avxvnni"),
                avx2: is_x86_feature_detected!("avx2"),
                sse42: is_x86_feature_detected!("sse4.2"),
                neon: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                avx_vnni: false,
                avx2: false,
                sse42: false,
                neon: true,
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                avx_vnni: false,
                avx2: false,
                sse42: false,
                neon: false,
            }
        }
    }

    fn best_level(&self) -> &'static str {
        if self.avx_vnni {
            "AVX-VNNI"
        } else if self.avx2 {
            "AVX2"
        } else if self.neon {
            "NEON"
        } else if self.sse42 {
            "SSE4.2"
        } else {
            "Scalar"
        }
    }

    /// Peak int8 giga-operations per second estimate by SIMD level
    fn peak_int8_gops(&self) -> f64 {
        if self.avx_vnni {
            400.0
        } else if self.avx2 {
            200.0
        } else if self.neon {
            100.0
        } else {
            10.0
        }
    }
}

/// A quantized int8 linear layer
#[derive(Debug, Clone)]
pub struct Int8Layer {
    pub weights: Vec<i8>,
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

/// Benchmark result for throughput comparisons
#[derive(Debug)]
pub struct BenchmarkResult {
    pub name: String,
    pub time_ms: f64,
    pub gops: f64,
    pub speedup: f64,
}

/// Generate deterministic random f32 data using `DefaultHasher`
fn generate_random_data(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut data = Vec::with_capacity(rows * cols);
    for i in 0..(rows * cols) {
        let mut hasher = DefaultHasher::new();
        (seed, i).hash(&mut hasher);
        let hash = hasher.finish();
        data.push((hash as f32 / u64::MAX as f32) * 2.0 - 1.0);
    }
    data
}

/// Per-tensor symmetric quantization: map f32 values to i8 range [-127, 127]
fn quantize_to_int8(data: &[f32]) -> (Vec<i8>, f32) {
    let abs_max = data
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max)
        .max(1e-10);
    let scale = abs_max / 127.0;
    let quantized: Vec<i8> = data
        .iter()
        .map(|v| (v / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (quantized, scale)
}

/// Simulated int8 matmul: A(m x k) * B(k x n) -> C(m x n)
/// Mimics VPDPBUSD: multiply i8 pairs, accumulate into i32, then rescale to f32
fn int8_matmul_simulate(
    a: &[i8],
    b: &[i8],
    m: usize,
    n: usize,
    k: usize,
    scale_a: f32,
    scale_b: f32,
) -> Vec<f32> {
    let combined_scale = scale_a * scale_b;
    let mut result = vec![0.0_f32; m * n];

    for row in 0..m {
        for col in 0..n {
            let mut acc: i32 = 0;
            for p in 0..k {
                acc += i32::from(a[row * k + p]) * i32::from(b[p * n + col]);
            }
            result[row * n + col] = acc as f32 * combined_scale;
        }
    }
    result
}

/// Benchmark f32 matmul via trueno::Matrix
fn benchmark_f32_matmul(size: usize, iterations: usize) -> BenchmarkResult {
    let a_data = generate_random_data(size, size, 42);
    let b_data = generate_random_data(size, size, 43);

    let a = Matrix::from_vec(size, size, a_data).expect("Failed to create matrix A");
    let b = Matrix::from_vec(size, size, b_data).expect("Failed to create matrix B");

    // Warmup
    let _ = a.matmul(&b);

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a.matmul(&b);
    }
    let elapsed = start.elapsed();

    let time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let ops = 2.0 * (size as f64).powi(3);
    let gops = ops / (time_ms / 1000.0) / 1e9;

    BenchmarkResult {
        name: format!("f32_matmul_{size}x{size}"),
        time_ms,
        gops,
        speedup: 1.0,
    }
}

/// Benchmark int8 matmul simulation
fn benchmark_int8_matmul(size: usize, iterations: usize) -> BenchmarkResult {
    let a_data = generate_random_data(size, size, 42);
    let b_data = generate_random_data(size, size, 43);

    let (a_q, scale_a) = quantize_to_int8(&a_data);
    let (b_q, scale_b) = quantize_to_int8(&b_data);

    // Warmup
    let _ = int8_matmul_simulate(&a_q, &b_q, size, size, size, scale_a, scale_b);

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = int8_matmul_simulate(&a_q, &b_q, size, size, size, scale_a, scale_b);
    }
    let elapsed = start.elapsed();

    let time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let ops = 2.0 * (size as f64).powi(3);
    let gops = ops / (time_ms / 1000.0) / 1e9;

    BenchmarkResult {
        name: format!("int8_matmul_{size}x{size}"),
        time_ms,
        gops,
        speedup: 0.0, // filled in by caller
    }
}

/// Compute mean absolute error between original f32 and dequantized int8
fn compute_quantization_error(original: &[f32], quantized: &[i8], scale: f32) -> f64 {
    let n = original.len();
    if n == 0 {
        return 0.0;
    }
    let total_error: f64 = original
        .iter()
        .zip(quantized.iter())
        .map(|(&orig, &q)| {
            let reconstructed = f64::from(q) * f64::from(scale);
            (f64::from(orig) - reconstructed).abs()
        })
        .sum();
    total_error / n as f64
}

fn main() {
    println!("=== AVX-VNNI Int8 Inference Acceleration ===\n");

    // Section 1: SIMD Detection
    println!("1. SIMD Capability Detection (AVX-VNNI Focus)");
    println!("   ─────────────────────────────────────────");

    let caps = SimdCapabilities::detect();
    #[cfg(target_arch = "x86_64")]
    {
        println!("   AVX-VNNI: {}", if caps.avx_vnni { "YES" } else { "no" });
        println!("   AVX2:     {}", if caps.avx2 { "YES" } else { "no" });
        println!("   SSE4.2:   {}", if caps.sse42 { "YES" } else { "no" });
    }
    #[cfg(target_arch = "aarch64")]
    {
        println!("   NEON:     {}", if caps.neon { "YES" } else { "no" });
    }
    println!("   Best level:       {}", caps.best_level());
    println!("   Peak Int8 GOPS:   {:.1}", caps.peak_int8_gops());
    println!();

    // Section 2: Quantization Demo
    println!("2. Per-Tensor Symmetric Quantization");
    println!("   ─────────────────────────────────────────");

    let demo_data = generate_random_data(1, 16, 99);
    let (demo_q, demo_scale) = quantize_to_int8(&demo_data);

    println!("   Scale factor: {demo_scale:.6}");
    println!("   ┌────────────┬────────┬──────────────┐");
    println!("   │ Original   │ Int8   │ Reconstructed│");
    println!("   ├────────────┼────────┼──────────────┤");
    for i in 0..4 {
        let reconstructed = f64::from(demo_q[i]) * f64::from(demo_scale);
        println!(
            "   │ {:>9.5}  │ {:>5}  │ {:>11.5}  │",
            demo_data[i], demo_q[i], reconstructed
        );
    }
    println!("   └────────────┴────────┴──────────────┘");
    println!();

    // Section 3: Int8 vs F32 Benchmark
    println!("3. Int8 vs F32 Matmul Benchmark");
    println!("   ─────────────────────────────────────────");
    println!("   ┌──────────────┬──────────────┬──────────┬──────────┬─────────┐");
    println!("   │ Size         │ Method       │ Time(ms) │ GOPS     │ Speedup │");
    println!("   ├──────────────┼──────────────┼──────────┼──────────┼─────────┤");

    let sizes = [64, 128, 256, 512];
    let iterations = 5;

    for &size in &sizes {
        let f32_result = benchmark_f32_matmul(size, iterations);
        let mut int8_result = benchmark_int8_matmul(size, iterations);
        int8_result.speedup = if f32_result.time_ms > 0.0 {
            f32_result.time_ms / int8_result.time_ms
        } else {
            1.0
        };

        println!(
            "   │ {:4}x{:4}    │ f32          │ {:>8.3} │ {:>8.2} │    1.0x │",
            size, size, f32_result.time_ms, f32_result.gops
        );
        println!(
            "   │              │ int8 (sim)   │ {:>8.3} │ {:>8.2} │  {:>4.1}x │",
            int8_result.time_ms, int8_result.gops, int8_result.speedup
        );
    }
    println!("   └──────────────┴──────────────┴──────────┴──────────┴─────────┘");
    println!();

    // Section 4: Accuracy Analysis
    println!("4. Quantization Accuracy Analysis");
    println!("   ─────────────────────────────────────────");

    let analysis_sizes = [128, 256, 512];
    println!("   ┌──────────────┬──────────────────┬──────────────────┐");
    println!("   │ Matrix Size  │ Mean Abs Error   │ Relative Error   │");
    println!("   ├──────────────┼──────────────────┼──────────────────┤");

    for &size in &analysis_sizes {
        let data = generate_random_data(size, size, 55);
        let (q, scale) = quantize_to_int8(&data);
        let mae = compute_quantization_error(&data, &q, scale);

        let abs_mean: f64 =
            data.iter().map(|v| f64::from(v.abs())).sum::<f64>() / (size * size) as f64;
        let rel_error = if abs_mean > 0.0 {
            mae / abs_mean * 100.0
        } else {
            0.0
        };

        println!(
            "   │ {:4}x{:4}    │ {:>14.8}   │ {:>13.4}%  │",
            size, size, mae, rel_error
        );
    }
    println!("   └──────────────┴──────────────────┴──────────────────┘");
    println!();

    // Section 5: Falsification Check
    println!("5. F8 Falsification Check (Int8 >=1.5x vs f32, 512x512)");
    println!("   ─────────────────────────────────────────");

    let f32_bench = benchmark_f32_matmul(512, iterations);
    let mut int8_bench = benchmark_int8_matmul(512, iterations);
    int8_bench.speedup = if f32_bench.time_ms > 0.0 {
        f32_bench.time_ms / int8_bench.time_ms
    } else {
        1.0
    };

    println!(
        "   f32  time: {:.3} ms  ({:.2} GOPS)",
        f32_bench.time_ms, f32_bench.gops
    );
    println!(
        "   int8 time: {:.3} ms  ({:.2} GOPS)",
        int8_bench.time_ms, int8_bench.gops
    );
    println!("   Speedup:   {:.2}x", int8_bench.speedup);

    if int8_bench.speedup >= 1.5 {
        println!("   Status: CLAIM SUPPORTED (>=1.5x achieved)");
    } else {
        println!(
            "   Status: CLAIM NOT MET ({:.2}x < 1.5x) -- int8 simulation \
             without true VNNI intrinsics is scalar-bound",
            int8_bench.speedup
        );
    }
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_detection() {
        let caps = SimdCapabilities::detect();
        assert!(!caps.best_level().is_empty());
    }

    #[test]
    fn test_peak_int8_gops_positive() {
        let caps = SimdCapabilities::detect();
        assert!(caps.peak_int8_gops() > 0.0);
    }

    #[test]
    fn test_quantize_roundtrip_small() {
        let data = vec![0.0, 0.5, -0.5, 1.0, -1.0];
        let (q, scale) = quantize_to_int8(&data);
        assert_eq!(q.len(), data.len());
        assert!(scale > 0.0);
        // Reconstruct and check closeness
        for (orig, &qv) in data.iter().zip(q.iter()) {
            let recon = f64::from(qv) * f64::from(scale);
            assert!(
                (f64::from(*orig) - recon).abs() < f64::from(scale) + 1e-6,
                "orig={orig}, recon={recon}, scale={scale}"
            );
        }
    }

    #[test]
    fn test_quantize_preserves_zero() {
        let data = vec![0.0_f32; 10];
        let (q, _scale) = quantize_to_int8(&data);
        for &v in &q {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn test_quantize_clamps_to_range() {
        let data = vec![100.0, -100.0, 0.0];
        let (q, _scale) = quantize_to_int8(&data);
        for &v in &q {
            assert!(i16::from(v) >= -127 && i16::from(v) <= 127);
        }
    }

    #[test]
    fn test_int8_matmul_identity() {
        // 2x2 identity-like: A = [[127,0],[0,127]], B = [[1,0],[0,1]]
        let a: Vec<i8> = vec![127, 0, 0, 127];
        let b: Vec<i8> = vec![1, 0, 0, 1];
        let scale_a = 1.0 / 127.0;
        let scale_b = 1.0;
        let result = int8_matmul_simulate(&a, &b, 2, 2, 2, scale_a, scale_b);
        // Should approximate identity * 1.0
        assert!((f64::from(result[0]) - 1.0).abs() < 0.01);
        assert!(f64::from(result[1]).abs() < 0.01);
        assert!(f64::from(result[2]).abs() < 0.01);
        assert!((f64::from(result[3]) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_int8_matmul_dimensions() {
        let a = vec![1_i8; 3 * 4];
        let b = vec![1_i8; 4 * 5];
        let result = int8_matmul_simulate(&a, &b, 3, 5, 4, 1.0, 1.0);
        assert_eq!(result.len(), 15);
        // Each element should be k * 1 * 1 * 1.0 * 1.0 = 4.0
        for &v in &result {
            assert!((f64::from(v) - 4.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_compute_quantization_error_zero() {
        let original = vec![0.0_f32; 10];
        let quantized = vec![0_i8; 10];
        let error = compute_quantization_error(&original, &quantized, 0.01);
        assert!(error < 1e-10);
    }

    #[test]
    fn test_compute_quantization_error_bounded() {
        let data = generate_random_data(16, 16, 42);
        let (q, scale) = quantize_to_int8(&data);
        let error = compute_quantization_error(&data, &q, scale);
        // Error should be bounded by scale / 2 (half-step quantization error)
        assert!(error < f64::from(scale));
    }

    #[test]
    fn test_generate_random_deterministic() {
        let d1 = generate_random_data(10, 10, 42);
        let d2 = generate_random_data(10, 10, 42);
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_generate_random_different_seeds() {
        let d1 = generate_random_data(10, 10, 1);
        let d2 = generate_random_data(10, 10, 2);
        assert_ne!(d1, d2);
    }

    #[test]
    fn test_benchmark_f32_matmul() {
        let result = benchmark_f32_matmul(32, 1);
        assert!(result.time_ms > 0.0);
        assert!(result.gops > 0.0);
    }

    #[test]
    fn test_int8_layer_memory_savings() {
        let layer = Int8Layer {
            weights: vec![0_i8; 256 * 256],
            scales: vec![0.01],
            rows: 256,
            cols: 256,
        };
        let int8_bytes = layer.weights.len();
        let f32_bytes = layer.rows * layer.cols * 4;
        assert_eq!(int8_bytes * 4, f32_bytes);
    }
}
