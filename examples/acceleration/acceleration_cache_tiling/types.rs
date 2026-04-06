#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
use apr_cookbook::Result;
#[allow(unused_imports)]
use std::time::Instant;
use trueno::Matrix;

pub const MATRIX_SIZE: usize = 512;
pub const ITERATIONS: usize = 5;
pub const TILE_SIZES: [usize; 6] = [8, 16, 32, 64, 128, 256];

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Hardware cache hierarchy parameters.
#[derive(Debug, Clone, Copy)]
pub struct CacheInfo {
    pub l1d_bytes: usize,
    pub l2_bytes: usize,
    pub l3_bytes: usize,
    pub line_size: usize,
}

/// Result of a single tiling benchmark run.
#[derive(Debug, Clone)]
pub struct TilingResult {
    pub tile_size: usize,
    pub time_ms: f64,
    pub gflops: f64,
    pub efficiency_vs_best: f64,
}

// ---------------------------------------------------------------------------
// Cache detection (hard-coded for Intel Core Ultra 7 155H)
// ---------------------------------------------------------------------------

pub fn detect_cache_sizes() -> CacheInfo {
    CacheInfo {
        l1d_bytes: 32 * 1024,
        l2_bytes: 2 * 1024 * 1024,
        l3_bytes: 24 * 1024 * 1024,
        line_size: 64,
    }
}

// ---------------------------------------------------------------------------
// Deterministic data generation
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Matrix multiplication kernels
// ---------------------------------------------------------------------------

/// Naive ijk matrix multiplication.
pub fn naive_matmul(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0_f32;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

/// Cache-aware 6-loop tiled matmul (ii,jj,kk,i,j,k). Boundary-safe via `min()`.
pub fn tiled_matmul(a: &[f32], b: &[f32], c: &mut [f32], n: usize, tile: usize) {
    for v in c.iter_mut() {
        *v = 0.0;
    }
    let mut ii = 0;
    while ii < n {
        let ie = (ii + tile).min(n);
        let mut jj = 0;
        while jj < n {
            let je = (jj + tile).min(n);
            let mut kk = 0;
            while kk < n {
                let ke = (kk + tile).min(n);
                for i in ii..ie {
                    for j in jj..je {
                        let mut s = c[i * n + j];
                        for k in kk..ke {
                            s += a[i * n + k] * b[k * n + j];
                        }
                        c[i * n + j] = s;
                    }
                }
                kk += tile;
            }
            jj += tile;
        }
        ii += tile;
    }
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

pub fn compute_gflops(n: usize, time_ms: f64) -> f64 {
    2.0 * (n as f64).powi(3) / (time_ms / 1000.0) / 1e9
}

pub fn make_result(tile: usize, time_ms: f64, n: usize) -> TilingResult {
    TilingResult {
        tile_size: tile,
        time_ms,
        gflops: compute_gflops(n, time_ms),
        efficiency_vs_best: 0.0,
    }
}

pub fn benchmark_naive(n: usize, iters: usize) -> TilingResult {
    let (a, b) = (
        generate_random_data(n, n, 42),
        generate_random_data(n, n, 43),
    );
    let mut c = vec![0.0_f32; n * n];
    naive_matmul(&a, &b, &mut c, n);
    let t = Instant::now();
    for _ in 0..iters {
        naive_matmul(&a, &b, &mut c, n);
    }
    make_result(0, t.elapsed().as_secs_f64() * 1000.0 / iters as f64, n)
}

pub fn benchmark_tiled(n: usize, tile: usize, iters: usize) -> TilingResult {
    let (a, b) = (
        generate_random_data(n, n, 42),
        generate_random_data(n, n, 43),
    );
    let mut c = vec![0.0_f32; n * n];
    tiled_matmul(&a, &b, &mut c, n, tile);
    let t = Instant::now();
    for _ in 0..iters {
        tiled_matmul(&a, &b, &mut c, n, tile);
    }
    make_result(tile, t.elapsed().as_secs_f64() * 1000.0 / iters as f64, n)
}

pub fn benchmark_trueno(n: usize, iters: usize) -> TilingResult {
    let a = Matrix::from_vec(n, n, generate_random_data(n, n, 42)).expect("matrix A");
    let b = Matrix::from_vec(n, n, generate_random_data(n, n, 43)).expect("matrix B");
    let _ = a.matmul(&b);
    let t = Instant::now();
    for _ in 0..iters {
        let _ = a.matmul(&b);
    }
    make_result(0, t.elapsed().as_secs_f64() * 1000.0 / iters as f64, n)
}

// ---------------------------------------------------------------------------
// Cache analysis utilities
// ---------------------------------------------------------------------------

/// Optimal tile: three tile blocks (A, B, C) of tile^2 * 4 bytes must fit.
pub fn optimal_tile_for_cache(cache_bytes: usize) -> usize {
    let floats = cache_bytes / (3 * size_of::<f32>());
    (floats as f64).sqrt().max(1.0) as usize
}

/// Working set in bytes for three tile-sized f32 blocks.
pub fn working_set_bytes(tile: usize) -> usize {
    3 * tile * tile * size_of::<f32>()
}

/// Classify which cache level a tile's working set fits into.
pub fn classify_cache_level(tile: usize) -> &'static str {
    let (bytes, c) = (working_set_bytes(tile), detect_cache_sizes());
    if bytes <= c.l1d_bytes {
        "L1"
    } else if bytes <= c.l2_bytes {
        "L2"
    } else if bytes <= c.l3_bytes {
        "L3"
    } else {
        "DRAM"
    }
}

pub fn fmt_bytes(b: usize) -> String {
    if b >= 1024 * 1024 {
        format!("{} MB", b / (1024 * 1024))
    } else if b >= 1024 {
        format!("{} KB", b / 1024)
    } else {
        format!("{b} B")
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
