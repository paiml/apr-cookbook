#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
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
    pub fn detect() -> Self {
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

    pub fn best_level(&self) -> &'static str {
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

    pub fn peak_gflops(&self) -> f64 {
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
pub fn generate_random_data(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
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
pub fn benchmark_matmul(m: usize, n: usize, k: usize, iterations: usize) -> BenchmarkResult {
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
pub fn benchmark_elementwise(size: usize, iterations: usize) -> BenchmarkResult {
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
pub fn benchmark_reduction(size: usize, iterations: usize) -> BenchmarkResult {
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
pub fn benchmark_softmax(batch: usize, seq_len: usize, iterations: usize) -> BenchmarkResult {
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
