#![allow(unused_imports)]
//! # Recipe: Cache-Aware Tiled Matrix Multiplication
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/avx512-matmul-v1.yaml
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
//!
//! ## Format Variants
//! ```bash
//! apr bench model.apr          # APR native format
//! apr bench model.gguf         # GGUF (llama.cpp compatible)
//! apr bench model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Hennessy, J. & Patterson, D. (2017). *Computer Architecture: A Quantitative Approach*. DOI: 10.1016/C2012-0-01712-X

#![deny(unsafe_code)]
#![deny(clippy::todo, clippy::unimplemented, clippy::panic)]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::time::Instant;
use trueno::Matrix;

use apr_cookbook::Result;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
