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
    pub fn detect() -> Self {
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

    pub fn best_level(&self) -> &'static str {
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
    pub fn peak_int8_gops(&self) -> f64 {
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

/// Per-tensor symmetric quantization: map f32 values to i8 range [-127, 127]
pub fn quantize_to_int8(data: &[f32]) -> (Vec<i8>, f32) {
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

// Simulated int8 matmul: A(m x k) * B(k x n) -> C(m x n)
/// Mimics VPDPBUSD: multiply i8 pairs, accumulate into i32, then rescale to f32
pub fn int8_matmul_simulate(
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
pub fn benchmark_f32_matmul(size: usize, iterations: usize) -> BenchmarkResult {
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
pub fn benchmark_int8_matmul(size: usize, iterations: usize) -> BenchmarkResult {
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
pub fn compute_quantization_error(original: &[f32], quantized: &[i8], scale: f32) -> f64 {
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
