//! # Recipe: Cache-Aware Tiled Matrix Multiplication
//!
//! **Category**: Acceleration
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: trueno (SIMD backend)
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A - cache hierarchy varies)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] 12 tests
//!
//! ## Learning Objective
//! Demonstrate how cache-aware tiling transforms a naive O(n^3) matrix multiply
//! from memory-bound to compute-bound by keeping working sets within L1/L2/L3.
//! Sweeps tile sizes, classifies each by cache level, and compares against
//! trueno's optimized SIMD matmul.
//!
//! ## Toyota Way Principles
//! - **Kaizen** (Continuous improvement): Iteratively refine tile size toward
//!   cache-optimal configuration
//! - **Muda** (Waste elimination): Eliminate cache misses by fitting tiles in L1
//! - **Genchi Genbutsu** (Go and see): Measure real GFLOPS, not theoretical peaks
//!
//! ## Run Command
//! ```bash
//! cargo run --example acceleration_cache_tiling --release
//! ```
//!
//! ## References
//! - Hennessy, J. & Patterson, D. (2017). *Computer Architecture: A Quantitative Approach*. DOI: 10.1016/C2012-0-01712-X

#![deny(unsafe_code)]
#![deny(clippy::todo, clippy::unimplemented, clippy::panic)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::time::Instant;
use trueno::Matrix;

use apr_cookbook::Result;

const MATRIX_SIZE: usize = 512;
const ITERATIONS: usize = 5;
const TILE_SIZES: [usize; 6] = [8, 16, 32, 64, 128, 256];

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Hardware cache hierarchy parameters.
#[derive(Debug, Clone, Copy)]
struct CacheInfo {
    l1d_bytes: usize,
    l2_bytes: usize,
    l3_bytes: usize,
    line_size: usize,
}

/// Result of a single tiling benchmark run.
#[derive(Debug, Clone)]
struct TilingResult {
    tile_size: usize,
    time_ms: f64,
    gflops: f64,
    efficiency_vs_best: f64,
}

// ---------------------------------------------------------------------------
// Cache detection (hard-coded for Intel Core Ultra 7 155H)
// ---------------------------------------------------------------------------

fn detect_cache_sizes() -> CacheInfo {
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

// ---------------------------------------------------------------------------
// Matrix multiplication kernels
// ---------------------------------------------------------------------------

/// Naive ijk matrix multiplication.
fn naive_matmul(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
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
fn tiled_matmul(a: &[f32], b: &[f32], c: &mut [f32], n: usize, tile: usize) {
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

fn compute_gflops(n: usize, time_ms: f64) -> f64 {
    2.0 * (n as f64).powi(3) / (time_ms / 1000.0) / 1e9
}

fn make_result(tile: usize, time_ms: f64, n: usize) -> TilingResult {
    TilingResult {
        tile_size: tile,
        time_ms,
        gflops: compute_gflops(n, time_ms),
        efficiency_vs_best: 0.0,
    }
}

fn benchmark_naive(n: usize, iters: usize) -> TilingResult {
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

fn benchmark_tiled(n: usize, tile: usize, iters: usize) -> TilingResult {
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

fn benchmark_trueno(n: usize, iters: usize) -> TilingResult {
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
fn optimal_tile_for_cache(cache_bytes: usize) -> usize {
    let floats = cache_bytes / (3 * size_of::<f32>());
    (floats as f64).sqrt().max(1.0) as usize
}

/// Working set in bytes for three tile-sized f32 blocks.
fn working_set_bytes(tile: usize) -> usize {
    3 * tile * tile * size_of::<f32>()
}

/// Classify which cache level a tile's working set fits into.
fn classify_cache_level(tile: usize) -> &'static str {
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

fn fmt_bytes(b: usize) -> String {
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

fn main() -> Result<()> {
    println!("=== APR Cookbook: Cache-Aware Tiled Matrix Multiplication ===\n");

    // 1. Cache Hierarchy Detection
    let cache = detect_cache_sizes();
    let sep = |w| println!("  {}", "-".repeat(w));
    println!("1. Cache Hierarchy Detection");
    sep(60);
    println!(
        "  L1d: {} KB  |  L2: {} KB  |  L3: {} KB  |  Line: {} B",
        cache.l1d_bytes / 1024,
        cache.l2_bytes / 1024,
        cache.l3_bytes / 1024,
        cache.line_size
    );
    let l1t = optimal_tile_for_cache(cache.l1d_bytes);
    let l2t = optimal_tile_for_cache(cache.l2_bytes);
    let l3t = optimal_tile_for_cache(cache.l3_bytes);
    println!("\n  Optimal tile per cache level:");
    for (nm, sz, tl) in [
        ("L1d", cache.l1d_bytes, l1t),
        ("L2", cache.l2_bytes, l2t),
        ("L3", cache.l3_bytes, l3t),
    ] {
        println!(
            "    {nm} ({:>6} KB) -> tile={tl:>4}  ws={:>8}",
            sz / 1024,
            fmt_bytes(working_set_bytes(tl))
        );
    }

    // 2. Naive vs Tiled Benchmark
    println!("\n2. Naive vs Tiled Benchmark ({MATRIX_SIZE}x{MATRIX_SIZE}, {ITERATIONS} iters)");
    sep(60);
    let naive = benchmark_naive(MATRIX_SIZE, ITERATIONS);
    let mut tiled: Vec<_> = TILE_SIZES
        .iter()
        .map(|&t| benchmark_tiled(MATRIX_SIZE, t, ITERATIONS))
        .collect();
    let best = tiled.iter().map(|r| r.gflops).fold(naive.gflops, f64::max);
    let mut nd = naive;
    if best > 0.0 {
        nd.efficiency_vs_best = nd.gflops / best;
        for r in &mut tiled {
            r.efficiency_vs_best = r.gflops / best;
        }
    }
    println!(
        "  {:>10}  {:>10}  {:>8}  {:>10}",
        "Tile", "Time(ms)", "GFLOPS", "Efficiency"
    );
    sep(44);
    let row = |label: &str, r: &TilingResult| {
        println!(
            "  {:>10}  {:>10.3}  {:>8.2}  {:>9.1}%",
            label,
            r.time_ms,
            r.gflops,
            r.efficiency_vs_best * 100.0
        );
    };
    row("naive", &nd);
    for r in &tiled {
        row(&r.tile_size.to_string(), r);
    }

    // 3. Cache Level Analysis
    println!("\n3. Cache Level Analysis");
    sep(56);
    println!(
        "  {:>10}  {:>12}  {:>6}  {:>8}",
        "Tile", "Working Set", "Level", "GFLOPS"
    );
    sep(42);
    for r in &tiled {
        let ws = working_set_bytes(r.tile_size);
        println!(
            "  {:>10}  {:>12}  {:>6}  {:>8.2}",
            r.tile_size,
            fmt_bytes(ws),
            classify_cache_level(r.tile_size),
            r.gflops
        );
    }

    // 4. Trueno Comparison
    println!("\n4. Trueno SIMD Comparison");
    sep(56);
    let trueno = benchmark_trueno(MATRIX_SIZE, ITERATIONS);
    let bt = tiled.iter().max_by(|a, b| {
        a.gflops
            .partial_cmp(&b.gflops)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let bt = bt.cloned().unwrap_or(make_result(0, 0.0, MATRIX_SIZE));
    println!("  {:>16}  {:>10}  {:>8}", "Kernel", "Time(ms)", "GFLOPS");
    sep(40);
    println!(
        "  {:>16}  {:>10.3}  {:>8.2}",
        "naive", nd.time_ms, nd.gflops
    );
    println!(
        "  {:>16}  {:>10.3}  {:>8.2}",
        format!("tiled({})", bt.tile_size),
        bt.time_ms,
        bt.gflops
    );
    println!(
        "  {:>16}  {:>10.3}  {:>8.2}",
        "trueno::matmul", trueno.time_ms, trueno.gflops
    );
    if nd.gflops > 0.0 {
        println!(
            "\n  Speedup (best tiled vs naive): {:.2}x",
            bt.gflops / nd.gflops
        );
        println!(
            "  Speedup (trueno vs naive):     {:.2}x",
            trueno.gflops / nd.gflops
        );
    }

    // 5. Optimal Tile Recommendations
    println!("\n5. Optimal Tile Recommendations");
    sep(56);
    println!(
        "  {:>8}  {:>10}  {:>12}",
        "Cache", "Opt.Tile", "Working Set"
    );
    sep(36);
    for (nm, tl) in [("L1d", l1t), ("L2", l2t), ("L3", l3t)] {
        println!(
            "  {:>8}  {:>10}  {:>10} KB",
            nm,
            tl,
            working_set_bytes(tl) / 1024
        );
    }
    println!("\n  Key insight: L1-fitting tiles (~{l1t}) minimise cache misses per block,");
    println!(
        "  but for {MATRIX_SIZE}x{MATRIX_SIZE} the L2-fitting tile (~{l2t}) often wins due to"
    );
    println!("  less loop overhead and better amortisation of tile setup cost.\n");
    println!("[SUCCESS] Cache tiling benchmark complete.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_info_sizes() {
        let c = detect_cache_sizes();
        assert_eq!(c.l1d_bytes, 32 * 1024);
        assert_eq!(c.l2_bytes, 2 * 1024 * 1024);
        assert_eq!(c.l3_bytes, 24 * 1024 * 1024);
        assert_eq!(c.line_size, 64);
    }

    #[test]
    fn test_naive_matmul_identity() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut c = [0.0_f32; 9];
        naive_matmul(&a, &eye, &mut c, 3);
        for i in 0..9 {
            assert!((a[i] - c[i]).abs() < 1e-6, "idx {i}");
        }
    }

    #[test]
    fn test_tiled_matmul_identity() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let eye = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut c = [0.0_f32; 9];
        tiled_matmul(&a, &eye, &mut c, 3, 2);
        for i in 0..9 {
            assert!((a[i] - c[i]).abs() < 1e-6, "idx {i}");
        }
    }

    #[test]
    fn test_naive_vs_tiled_agreement() {
        let n = 16;
        let (a, b) = (
            generate_random_data(n, n, 100),
            generate_random_data(n, n, 200),
        );
        let (mut cn, mut ct) = (vec![0.0_f32; n * n], vec![0.0_f32; n * n]);
        naive_matmul(&a, &b, &mut cn, n);
        tiled_matmul(&a, &b, &mut ct, n, 4);
        for i in 0..cn.len() {
            assert!((cn[i] - ct[i]).abs() < 1e-3, "idx {i}");
        }
    }

    #[test]
    fn test_tiled_boundary_handling() {
        let n = 7; // not divisible by tile=4
        let (a, b) = (
            generate_random_data(n, n, 300),
            generate_random_data(n, n, 301),
        );
        let (mut cn, mut ct) = (vec![0.0_f32; n * n], vec![0.0_f32; n * n]);
        naive_matmul(&a, &b, &mut cn, n);
        tiled_matmul(&a, &b, &mut ct, n, 4);
        for i in 0..cn.len() {
            assert!((cn[i] - ct[i]).abs() < 1e-3, "idx {i}");
        }
    }

    #[test]
    fn test_optimal_tile_for_l1() {
        let t = optimal_tile_for_cache(32 * 1024); // sqrt(32768/12) ~ 52
        assert!((45..=60).contains(&t), "L1 tile {t} outside 45..60");
    }

    #[test]
    fn test_optimal_tile_for_l2() {
        let t = optimal_tile_for_cache(2 * 1024 * 1024); // sqrt(2097152/12) ~ 418
        assert!((380..=450).contains(&t), "L2 tile {t} outside 380..450");
    }

    #[test]
    fn test_classify_l1_l2_l3_dram() {
        assert_eq!(classify_cache_level(8), "L1"); // 768 B
        assert_eq!(classify_cache_level(128), "L2"); // 192 KB
        assert_eq!(classify_cache_level(512), "L3"); // 3 MB
        assert_eq!(classify_cache_level(2048), "DRAM"); // 48 MB
    }

    #[test]
    fn test_working_set_bytes() {
        assert_eq!(working_set_bytes(1), 12);
        assert_eq!(working_set_bytes(10), 1200);
        assert_eq!(working_set_bytes(32), 12288);
    }

    #[test]
    fn test_generate_random_deterministic() {
        let d1 = generate_random_data(10, 10, 42);
        assert_eq!(d1, generate_random_data(10, 10, 42));
        assert_ne!(d1, generate_random_data(10, 10, 99));
    }

    #[test]
    fn test_compute_gflops() {
        let gf = compute_gflops(64, 1.0); // 2*64^3 / 0.001 / 1e9 = 0.524288
        assert!((gf - 0.524_288).abs() < 1e-4, "got {gf}");
    }

    #[test]
    fn test_benchmarks_return_positive() {
        let (n, t) = (benchmark_naive(16, 1), benchmark_tiled(16, 4, 1));
        assert!(n.time_ms > 0.0 && n.gflops > 0.0);
        assert!(t.time_ms > 0.0 && t.gflops > 0.0);
    }
}
