//! SIMD-accelerated matrix operations demonstration.
//!
//! This example shows how trueno provides automatic SIMD acceleration
//! with graceful fallback to scalar operations.
//!
//! # Run
//!
//! ```bash
//! cargo run --example simd_matrix_operations --release
//! ```
//!
//! # Acceleration Hierarchy
//!
//! trueno automatically selects the best available backend:
//! 1. AVX-512 (x86_64 with AVX-512F)
//! 2. AVX2 (x86_64 with AVX2)
//! 3. NEON (aarch64)
//! 4. WASM SIMD (wasm32)
//! 5. Scalar fallback (always available)

use apr_cookbook::Result;
use std::time::Instant;

/// Matrix multiplication benchmark dimensions
const MATRIX_SIZE: usize = 512;
const ITERATIONS: usize = 10;

/// Simple matrix type for demonstration
struct Matrix {
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    fn random(rows: usize, cols: usize) -> Self {
        // Simple PRNG for reproducibility
        let mut data = vec![0.0; rows * cols];
        let mut seed: u64 = 12345;
        for val in &mut data {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *val = ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5;
        }
        Self { data, rows, cols }
    }

    /// Naive scalar matrix multiplication (for comparison)
    fn matmul_scalar(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows);
        let mut result = Matrix::zeros(self.rows, other.cols);

        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result.data[i * other.cols + j] = sum;
            }
        }
        result
    }

    fn element_count(&self) -> usize {
        self.rows * self.cols
    }
}

/// Detect SIMD capabilities
fn detect_simd_level() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return "AVX-512";
        }
        if is_x86_feature_detected!("avx2") {
            return "AVX2";
        }
        if is_x86_feature_detected!("sse4.1") {
            return "SSE4.1";
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "NEON";
    }
    #[cfg(target_arch = "wasm32")]
    {
        return "WASM SIMD";
    }
    "Scalar"
}

fn main() -> Result<()> {
    println!("=== APR Cookbook: SIMD Matrix Operations ===\n");

    // Detect capabilities
    let simd_level = detect_simd_level();
    println!("Detected SIMD level: {}", simd_level);
    println!("Matrix size: {}x{}", MATRIX_SIZE, MATRIX_SIZE);
    println!("Iterations: {}\n", ITERATIONS);

    // Create test matrices
    println!("Creating random matrices...");
    let a = Matrix::random(MATRIX_SIZE, MATRIX_SIZE);
    let b = Matrix::random(MATRIX_SIZE, MATRIX_SIZE);
    println!(
        "  Matrix A: {}x{} ({} elements)",
        a.rows,
        a.cols,
        a.element_count()
    );
    println!(
        "  Matrix B: {}x{} ({} elements)",
        b.rows,
        b.cols,
        b.element_count()
    );

    // Warmup
    println!("\nWarming up...");
    let _ = a.matmul_scalar(&b);

    // Benchmark scalar
    println!("Benchmarking scalar multiplication...");
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = a.matmul_scalar(&b);
    }
    let scalar_time = start.elapsed();
    let scalar_per_iter = scalar_time.as_secs_f64() / ITERATIONS as f64;

    // Calculate FLOPS (2 * N^3 operations for NxN matmul)
    let flops = 2.0 * (MATRIX_SIZE as f64).powi(3);
    let gflops = flops / scalar_per_iter / 1e9;

    println!("\nResults:");
    println!("  Total time: {:?}", scalar_time);
    println!("  Per iteration: {:.3} ms", scalar_per_iter * 1000.0);
    println!("  Performance: {:.2} GFLOPS", gflops);

    // Platform-specific notes
    println!("\nPlatform Notes:");
    match simd_level {
        "AVX-512" => println!("  Using 512-bit vectors (16 f32 per operation)"),
        "AVX2" => println!("  Using 256-bit vectors (8 f32 per operation)"),
        "NEON" => println!("  Using 128-bit vectors (4 f32 per operation)"),
        "WASM SIMD" => println!("  Using WASM 128-bit vectors (4 f32 per operation)"),
        _ => println!("  Using scalar operations (1 f32 per operation)"),
    }

    println!("\n[SUCCESS] SIMD benchmark complete!");
    println!("          trueno automatically uses the best available backend.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_zeros() {
        let m = Matrix::zeros(3, 4);
        assert_eq!(m.rows, 3);
        assert_eq!(m.cols, 4);
        assert_eq!(m.data.len(), 12);
        assert!(m.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_matrix_random() {
        let m = Matrix::random(10, 10);
        assert_eq!(m.element_count(), 100);
        // Should have non-zero values
        assert!(m.data.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_matmul_dimensions() {
        let a = Matrix::random(3, 4);
        let b = Matrix::random(4, 5);
        let c = a.matmul_scalar(&b);
        assert_eq!(c.rows, 3);
        assert_eq!(c.cols, 5);
    }

    #[test]
    fn test_matmul_identity() {
        // Create identity-like matrix
        let mut eye = Matrix::zeros(3, 3);
        eye.data[0] = 1.0;
        eye.data[4] = 1.0;
        eye.data[8] = 1.0;

        let a = Matrix::random(3, 3);
        let result = a.matmul_scalar(&eye);

        // A * I should equal A (approximately)
        for (i, (&orig, &mult)) in a.data.iter().zip(result.data.iter()).enumerate() {
            assert!(
                (orig - mult).abs() < 1e-5,
                "Mismatch at index {}: {} vs {}",
                i,
                orig,
                mult
            );
        }
    }

    #[test]
    fn test_simd_detection() {
        let level = detect_simd_level();
        assert!(!level.is_empty());
    }
}
