//! Trueno SIMD Operations Example
//!
//! Demonstrates SIMD-accelerated matrix operations using trueno 0.11.
//! Part of APR-024 specification.
//!
//! # Trueno SIMD Features
//!
//! - **AVX-512**: 512-bit vectors, 80+ GFLOPS matmul
//! - **AVX2**: 256-bit vectors, ~40 GFLOPS matmul
//! - **NEON**: ARM SIMD, efficient on Apple Silicon
//! - **Scalar**: Pure Rust fallback
//!
//! # Running
//!
//! ```bash
//! cargo run --example trueno_simd_ops --release
//! ```
//!
//! # Falsification Claim (F7)
//!
//! AVX-512 matmul achieves ≥80 GFLOPS on compatible hardware.
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

/// SIMD capability detection
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    pub avx512f: bool,
    pub avx2: bool,
    pub sse42: bool,
    pub neon: bool,
}

impl SimdCapabilities {
    /// Detect available SIMD features
    fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx512f: is_x86_feature_detected!("avx512f"),
                avx2: is_x86_feature_detected!("avx2"),
                sse42: is_x86_feature_detected!("sse4.2"),
                neon: false,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                avx512f: false,
                avx2: false,
                sse42: false,
                neon: true,
            }
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                avx512f: false,
                avx2: false,
                sse42: false,
                neon: false,
            }
        }
    }

    fn best_level(&self) -> &'static str {
        if self.avx512f {
            "AVX-512"
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

    fn peak_gflops(&self) -> f64 {
        if self.avx512f {
            80.0
        } else if self.avx2 {
            40.0
        } else if self.neon {
            30.0
        } else {
            5.0
        }
    }
}

/// Benchmark result
#[derive(Debug)]
pub struct BenchmarkResult {
    pub name: String,
    pub time_ms: f64,
    pub gflops: f64,
    pub efficiency: f64,
}

/// Generate random f32 matrix data
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

/// Matrix multiplication benchmark using trueno
fn benchmark_matmul(m: usize, n: usize, k: usize, iterations: usize) -> BenchmarkResult {
    let a_data = generate_random_data(m, k, 42);
    let b_data = generate_random_data(k, n, 43);

    let a = Matrix::from_vec(m, k, a_data).expect("Failed to create matrix A");
    let b = Matrix::from_vec(k, n, b_data).expect("Failed to create matrix B");

    // Warmup
    let _ = a.matmul(&b);

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a.matmul(&b);
    }
    let elapsed = start.elapsed();

    let time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let gflops = flops / (time_ms / 1000.0) / 1e9;

    let caps = SimdCapabilities::detect();
    let efficiency = gflops / caps.peak_gflops() * 100.0;

    BenchmarkResult {
        name: format!("matmul_{}x{}x{}", m, n, k),
        time_ms,
        gflops,
        efficiency,
    }
}

/// Element-wise operations benchmark (manual SIMD via slice access)
fn benchmark_elementwise(size: usize, iterations: usize) -> BenchmarkResult {
    let a_data = generate_random_data(1, size, 42);
    let b_data = generate_random_data(1, size, 43);

    let a = Matrix::from_vec(1, size, a_data).expect("Failed to create matrix A");
    let b = Matrix::from_vec(1, size, b_data).expect("Failed to create matrix B");

    // Element-wise add via slice access (auto-vectorized by LLVM)
    let elementwise_add = |a: &Matrix<f32>, b: &Matrix<f32>| -> Vec<f32> {
        a.as_slice()
            .iter()
            .zip(b.as_slice().iter())
            .map(|(x, y)| x + y)
            .collect()
    };

    // Element-wise mul via slice access (auto-vectorized by LLVM)
    let elementwise_mul = |a: &Matrix<f32>, b: &Matrix<f32>| -> Vec<f32> {
        a.as_slice()
            .iter()
            .zip(b.as_slice().iter())
            .map(|(x, y)| x * y)
            .collect()
    };

    // Warmup
    let _ = elementwise_add(&a, &b);

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = elementwise_add(&a, &b);
        let _ = elementwise_mul(&a, &b);
    }
    let elapsed = start.elapsed();

    let time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let ops = size as f64 * 2.0;
    let gflops = ops / (time_ms / 1000.0) / 1e9;

    let caps = SimdCapabilities::detect();
    let efficiency = (gflops / caps.peak_gflops() * 100.0).min(100.0);

    BenchmarkResult {
        name: format!("elementwise_{}", size),
        time_ms,
        gflops,
        efficiency,
    }
}

/// Reduction benchmark (sum)
fn benchmark_reduction(size: usize, iterations: usize) -> BenchmarkResult {
    let a_data = generate_random_data(1, size, 42);
    let a = Matrix::from_vec(1, size, a_data).expect("Failed to create matrix");

    // Warmup
    let _ = a.as_slice().iter().sum::<f32>();

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _sum: f32 = a.as_slice().iter().sum();
    }
    let elapsed = start.elapsed();

    let time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let ops = size as f64;
    let gflops = ops / (time_ms / 1000.0) / 1e9;

    let caps = SimdCapabilities::detect();
    let efficiency = (gflops / caps.peak_gflops() * 100.0).min(100.0);

    BenchmarkResult {
        name: format!("reduction_{}", size),
        time_ms,
        gflops,
        efficiency,
    }
}

/// Softmax benchmark
fn benchmark_softmax(batch: usize, seq_len: usize, iterations: usize) -> BenchmarkResult {
    let data = generate_random_data(batch, seq_len, 42);
    let logits = Matrix::from_vec(batch, seq_len, data).expect("Failed to create matrix");

    let compute_softmax = |m: &Matrix<f32>| -> Vec<f32> {
        let rows = m.rows();
        let cols = m.cols();
        let mut result = vec![0.0f32; rows * cols];

        for i in 0..rows {
            let row_start = i * cols;
            let row_end = row_start + cols;
            let row = &m.as_slice()[row_start..row_end];

            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);

            let mut sum = 0.0f32;
            for (j, &val) in row.iter().enumerate() {
                let exp_val = (val - max_val).exp();
                result[row_start + j] = exp_val;
                sum += exp_val;
            }

            for j in 0..cols {
                result[row_start + j] /= sum;
            }
        }

        result
    };

    // Warmup
    let _ = compute_softmax(&logits);

    // Benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = compute_softmax(&logits);
    }
    let elapsed = start.elapsed();

    let time_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    let ops = (batch * seq_len) as f64 * 3.0;
    let gflops = ops / (time_ms / 1000.0) / 1e9;

    let caps = SimdCapabilities::detect();
    let efficiency = (gflops / caps.peak_gflops() * 100.0).min(100.0);

    BenchmarkResult {
        name: format!("softmax_{}x{}", batch, seq_len),
        time_ms,
        gflops,
        efficiency,
    }
}

fn main() {
    println!("=== Trueno SIMD Operations Example ===\n");

    // Section 1: SIMD Detection
    println!("1. SIMD Capability Detection");
    println!("   ─────────────────────────────────────────");

    let caps = SimdCapabilities::detect();
    #[cfg(target_arch = "x86_64")]
    {
        println!("   • AVX-512F: {}", if caps.avx512f { "✓" } else { "✗" });
        println!("   • AVX2:     {}", if caps.avx2 { "✓" } else { "✗" });
        println!("   • SSE4.2:   {}", if caps.sse42 { "✓" } else { "✗" });
    }
    #[cfg(target_arch = "aarch64")]
    {
        println!("   • NEON:     {}", if caps.neon { "✓" } else { "✗" });
    }
    println!("   Best level: {}", caps.best_level());
    println!("   Peak GFLOPS: {:.1}", caps.peak_gflops());
    println!();

    // Section 2: Matrix Multiplication
    println!("2. Matrix Multiplication (trueno::Matrix::matmul)");
    println!("   ─────────────────────────────────────────");
    println!("   ┌──────────────┬────────────┬────────────┬────────────┐");
    println!("   │ Size         │ Time (ms)  │ GFLOPS     │ Efficiency │");
    println!("   ├──────────────┼────────────┼────────────┼────────────┤");

    for size in [64, 128, 256, 512] {
        let result = benchmark_matmul(size, size, size, 10);
        println!(
            "   │ {:4}x{:4}    │ {:8.3}   │ {:8.2}   │ {:6.1}%    │",
            size, size, result.time_ms, result.gflops, result.efficiency
        );
    }
    println!("   └──────────────┴────────────┴────────────┴────────────┘");
    println!();

    // Section 3: Element-wise Operations
    println!("3. Element-wise Operations (add, mul)");
    println!("   ─────────────────────────────────────────");
    println!("   ┌──────────────┬────────────┬────────────┬────────────┐");
    println!("   │ Size         │ Time (ms)  │ GFLOPS     │ Efficiency │");
    println!("   ├──────────────┼────────────┼────────────┼────────────┤");

    for size in [10_000, 100_000, 1_000_000] {
        let result = benchmark_elementwise(size, 100);
        println!(
            "   │ {:>10}   │ {:8.3}   │ {:8.2}   │ {:6.1}%    │",
            size, result.time_ms, result.gflops, result.efficiency
        );
    }
    println!("   └──────────────┴────────────┴────────────┴────────────┘");
    println!();

    // Section 4: Reduction
    println!("4. Reduction Operations (sum)");
    println!("   ─────────────────────────────────────────");
    println!("   ┌──────────────┬────────────┬────────────┬────────────┐");
    println!("   │ Size         │ Time (ms)  │ GFLOPS     │ Efficiency │");
    println!("   ├──────────────┼────────────┼────────────┼────────────┤");

    for size in [10_000, 100_000, 1_000_000] {
        let result = benchmark_reduction(size, 100);
        println!(
            "   │ {:>10}   │ {:8.3}   │ {:8.2}   │ {:6.1}%    │",
            size, result.time_ms, result.gflops, result.efficiency
        );
    }
    println!("   └──────────────┴────────────┴────────────┴────────────┘");
    println!();

    // Section 5: Softmax
    println!("5. Softmax (Attention Component)");
    println!("   ─────────────────────────────────────────");
    println!("   ┌──────────────┬────────────┬────────────┬────────────┐");
    println!("   │ Batch×Seq    │ Time (ms)  │ GFLOPS     │ Efficiency │");
    println!("   ├──────────────┼────────────┼────────────┼────────────┤");

    for (batch, seq) in [(1, 512), (8, 512), (32, 512)] {
        let result = benchmark_softmax(batch, seq, 100);
        println!(
            "   │ {:4}×{:4}    │ {:8.3}   │ {:8.2}   │ {:6.1}%    │",
            batch, seq, result.time_ms, result.gflops, result.efficiency
        );
    }
    println!("   └──────────────┴────────────┴────────────┴────────────┘");
    println!();

    // Section 6: F7 Check
    println!("6. F7 Falsification Check (AVX-512 ≥80 GFLOPS)");
    println!("   ─────────────────────────────────────────");

    let large_matmul = benchmark_matmul(512, 512, 512, 5);
    println!("   Benchmark: 512x512 matmul");
    println!("   Achieved: {:.2} GFLOPS", large_matmul.gflops);

    if caps.avx512f {
        if large_matmul.gflops >= 80.0 {
            println!("   Status: ✓ CLAIM SUPPORTED");
        } else {
            println!("   Status: ⚠ Below threshold");
        }
    } else {
        println!("   Status: N/A (AVX-512 not available)");
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
    fn test_peak_gflops_positive() {
        let caps = SimdCapabilities::detect();
        assert!(caps.peak_gflops() > 0.0);
    }

    #[test]
    fn test_matmul_benchmark() {
        let result = benchmark_matmul(32, 32, 32, 1);
        assert!(result.time_ms > 0.0);
        assert!(result.gflops > 0.0);
    }

    #[test]
    fn test_elementwise_benchmark() {
        let result = benchmark_elementwise(1000, 1);
        assert!(result.time_ms > 0.0);
    }

    #[test]
    fn test_reduction_benchmark() {
        let result = benchmark_reduction(1000, 1);
        assert!(result.time_ms > 0.0);
    }

    #[test]
    fn test_softmax_benchmark() {
        let result = benchmark_softmax(4, 128, 1);
        assert!(result.time_ms > 0.0);
    }

    #[test]
    fn test_trueno_matrix_creation() {
        let data = generate_random_data(10, 10, 42);
        let m = Matrix::from_vec(10, 10, data).unwrap();
        assert_eq!(m.rows(), 10);
        assert_eq!(m.cols(), 10);
    }

    #[test]
    fn test_trueno_matmul() {
        let a_data = generate_random_data(4, 8, 42);
        let b_data = generate_random_data(8, 6, 43);
        let a = Matrix::from_vec(4, 8, a_data).unwrap();
        let b = Matrix::from_vec(8, 6, b_data).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.rows(), 4);
        assert_eq!(c.cols(), 6);
    }

    #[test]
    fn test_trueno_elementwise() {
        let a_data = generate_random_data(1, 100, 42);
        let b_data = generate_random_data(1, 100, 43);
        let a = Matrix::from_vec(1, 100, a_data).unwrap();
        let b = Matrix::from_vec(1, 100, b_data).unwrap();
        // Element-wise via slice access
        let c: Vec<f32> = a
            .as_slice()
            .iter()
            .zip(b.as_slice().iter())
            .map(|(x, y)| x + y)
            .collect();
        assert_eq!(c.len(), 100);
    }

    #[test]
    fn test_trueno_reduction() {
        let data = generate_random_data(1, 100, 42);
        let a = Matrix::from_vec(1, 100, data).unwrap();
        let sum: f32 = a.as_slice().iter().sum();
        assert!(sum.is_finite());
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
    fn test_f7_gflops_positive() {
        let result = benchmark_matmul(128, 128, 128, 1);
        assert!(result.gflops > 0.0);
    }
}
