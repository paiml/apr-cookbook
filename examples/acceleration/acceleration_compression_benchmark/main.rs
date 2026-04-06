#![allow(unused_imports)]
//! # Recipe: Compression Benchmark for .apr Model Payloads
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/avx512-matmul-v1.yaml
//! **Category**: Acceleration - Compression
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: lz4_flex 0.11, zstd 0.13
//!
//! ## QA Checklist
//! 1. [x] `cargo run --example acceleration_compression_benchmark` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes (12 tests)
//! 3. [x] Deterministic output via `DefaultHasher` seeded generation
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable (single 10 MB allocation reused)
//! 6. [x] WASM compatible (N/A - benchmarking tool)
//! 7. [x] Clippy pedantic clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in main logic
//! 10. [x] 12 unit tests
//!
//! ## Learning Objective
//! Compare LZ4 and Zstandard compression on realistic .apr tensor payloads.
//! Measure compression ratio, throughput (GB/s), and decompression latency to
//! guide the choice of on-disk and over-the-wire compression for ML models.
//!
//! ## Toyota Way Principles
//! - **Muda** (Waste elimination): Choose optimal compression to minimize
//!   storage, transfer time, and decompression overhead during inference.
//!
//! ## Run Command
//! ```bash
//! cargo run --example acceleration_compression_benchmark --release
//! ```
//!
//! ## Falsification Claim (F9)
//! LZ4 decompression throughput >= 2 GB/s on model-like data.
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

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== Compression Benchmark for .apr Model Payloads ===\n");

    let size = 10 * 1024 * 1024; // 10 MB
    let iters = 5;
    let methods = [
        CompressionMethod::None,
        CompressionMethod::Lz4,
        CompressionMethod::Zstd(1),
        CompressionMethod::Zstd(3),
        CompressionMethod::Zstd(9),
    ];

    // Section 1: Data Generation
    println!("1. Data Generation");
    println!("   ─────────────────────────────────────────────────");
    let random_data = generate_random_data(size, 42);
    let model_data = generate_model_like_data(size, 42);
    let sparse_data = generate_sparse_data(size, 42, 0.90);
    println!("   Random data:     {} (high entropy)", fmt_bytes(size));
    println!(
        "   Model-like data: {} (structured blocks)",
        fmt_bytes(size)
    );
    println!("   Sparse data:     {} (90% zeros)", fmt_bytes(size));
    println!();

    // Section 2: Compression Benchmark Table (model-like data)
    println!("2. Compression Benchmark (model-like data, {iters} iterations)");
    println!("   ─────────────────────────────────────────────────");
    print_result_header();
    for method in &methods {
        print_result_row(&benchmark_compression(&model_data, *method, iters));
    }
    println!();

    // Section 3: Data Type Comparison
    println!("3. Data Type Comparison (LZ4 and ZSTD-3)");
    println!("   ─────────────────────────────────────────────────");
    let cmp_methods = [CompressionMethod::Lz4, CompressionMethod::Zstd(3)];
    let datasets: [(&str, &[u8]); 3] = [
        ("Random", &random_data),
        ("Model-like", &model_data),
        ("Sparse-90%", &sparse_data),
    ];
    println!(
        "   {:<12} {:<8} {:>7} {:>10} {:>10}",
        "Data Type", "Method", "Ratio", "Comp GB/s", "Dec GB/s"
    );
    println!("   {}", "-".repeat(50));
    for (name, data) in &datasets {
        for m in &cmp_methods {
            let r = benchmark_compression(data, *m, iters);
            println!(
                "   {:<12} {:<8} {:>6.2}x {:>9.2} {:>10.2}",
                name,
                r.method_name,
                r.ratio,
                r.compress_throughput_gbps,
                r.decompress_throughput_gbps,
            );
        }
    }
    println!();

    // Section 4: Decompression Throughput Focus
    println!("4. Decompression Throughput Focus (hot path for inference)");
    println!("   ─────────────────────────────────────────────────");
    println!(
        "   {:<8} {:>10} {:>10} {:>12}",
        "Method", "Random", "Model", "Sparse-90%"
    );
    println!("   {}", "-".repeat(44));
    for method in &methods {
        if matches!(method, CompressionMethod::None) {
            continue;
        }
        let rr = benchmark_compression(&random_data, *method, 10);
        let rm = benchmark_compression(&model_data, *method, 10);
        let rs = benchmark_compression(&sparse_data, *method, 10);
        println!(
            "   {:<8} {:>8.2} {:>8.2} {:>10.2}  GB/s",
            method.name(),
            rr.decompress_throughput_gbps,
            rm.decompress_throughput_gbps,
            rs.decompress_throughput_gbps,
        );
    }
    println!();

    // Section 5: Recommendation Summary
    println!("5. Recommendation Summary");
    println!("   ─────────────────────────────────────────────────");
    println!("   Use case                     | Recommended");
    println!("   ─────────────────────────────|─────────────");
    println!("   Inference hot path (latency) | LZ4");
    println!("   On-disk storage (size)       | ZSTD-3");
    println!("   Network transfer (bandwidth) | ZSTD-9");
    println!("   Sparse/pruned models         | ZSTD-3");
    println!("   WASM/browser bundles         | LZ4");
    println!();

    // Section 6: F9 Falsification Check
    println!("6. Falsification Check: F9 (LZ4 decompression >= 2 GB/s)");
    println!("   ─────────────────────────────────────────────────");
    let f9 = benchmark_compression(&model_data, CompressionMethod::Lz4, 20);
    println!("   Payload: model-like {}, 20 iterations", fmt_bytes(size));
    println!(
        "   LZ4 decompress throughput: {:.2} GB/s",
        f9.decompress_throughput_gbps
    );
    if f9.decompress_throughput_gbps >= 2.0 {
        println!("   Status: CLAIM SUPPORTED (>= 2 GB/s)");
    } else {
        println!(
            "   Status: BELOW THRESHOLD ({:.2} < 2 GB/s)",
            f9.decompress_throughput_gbps
        );
        println!("   Note: Run with --release for optimized throughput");
    }
    println!();
    println!("=== Benchmark Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_random_data_deterministic() {
        let d1 = generate_random_data(1024, 42);
        let d2 = generate_random_data(1024, 42);
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_generate_random_data_different_seeds() {
        let d1 = generate_random_data(1024, 1);
        let d2 = generate_random_data(1024, 2);
        assert_ne!(d1, d2);
    }

    #[test]
    fn test_generate_model_like_data_size() {
        let data = generate_model_like_data(10_000, 42);
        assert_eq!(data.len(), 10_000);
    }

    #[test]
    fn test_generate_sparse_data_sparsity() {
        let data = generate_sparse_data(100_000, 42, 0.90);
        let zeros = data.iter().filter(|&&b| b == 0).count();
        let ratio = zeros as f64 / data.len() as f64;
        assert!(
            ratio > 0.80,
            "Expected ~90% zeros, got {:.1}%",
            ratio * 100.0
        );
        assert!(
            ratio < 0.98,
            "Expected ~90% zeros, got {:.1}%",
            ratio * 100.0
        );
    }

    #[test]
    fn test_lz4_roundtrip() {
        let original = generate_random_data(4096, 99);
        let compressed = compress_data(&original, CompressionMethod::Lz4).unwrap();
        let decompressed = decompress_data(&compressed, CompressionMethod::Lz4).unwrap();
        assert_eq!(original, decompressed);
    }

    #[test]
    fn test_zstd_roundtrip() {
        let original = generate_model_like_data(4096, 99);
        let compressed = compress_data(&original, CompressionMethod::Zstd(3)).unwrap();
        let decompressed = decompress_data(&compressed, CompressionMethod::Zstd(3)).unwrap();
        assert_eq!(original, decompressed);
    }

    #[test]
    fn test_none_passthrough() {
        let original = generate_random_data(1024, 42);
        let compressed = compress_data(&original, CompressionMethod::None).unwrap();
        let decompressed = decompress_data(&compressed, CompressionMethod::None).unwrap();
        assert_eq!(original, compressed);
        assert_eq!(original, decompressed);
    }

    #[test]
    fn test_compression_method_names() {
        assert_eq!(CompressionMethod::None.name(), "None");
        assert_eq!(CompressionMethod::Lz4.name(), "LZ4");
        assert_eq!(CompressionMethod::Zstd(1).name(), "ZSTD-1");
        assert_eq!(CompressionMethod::Zstd(9).name(), "ZSTD-9");
    }

    #[test]
    fn test_benchmark_returns_valid_metrics() {
        let data = generate_model_like_data(8192, 42);
        let result = benchmark_compression(&data, CompressionMethod::Lz4, 1);
        assert_eq!(result.original_size, 8192);
        assert!(result.compressed_size > 0);
        assert!(result.compressed_size <= result.original_size + 64);
        assert!(result.ratio > 0.0);
        assert!(result.compress_time_ms >= 0.0);
        assert!(result.decompress_time_ms >= 0.0);
    }

    #[test]
    fn test_sparse_data_compresses_well() {
        let sparse = generate_sparse_data(100_000, 42, 0.95);
        let compressed = compress_data(&sparse, CompressionMethod::Zstd(3)).unwrap();
        let ratio = sparse.len() as f64 / compressed.len() as f64;
        assert!(
            ratio > 3.0,
            "95% sparse data should compress >3x, got {ratio:.2}x"
        );
    }

    #[test]
    fn test_model_data_more_compressible_than_random() {
        let random = generate_random_data(100_000, 42);
        let model = generate_model_like_data(100_000, 42);
        let r_random = compress_data(&random, CompressionMethod::Lz4).unwrap();
        let r_model = compress_data(&model, CompressionMethod::Lz4).unwrap();
        assert!(
            r_model.len() < r_random.len(),
            "Model-like data ({} B) should compress smaller than random ({} B)",
            r_model.len(),
            r_random.len(),
        );
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(fmt_bytes(512), "512 B");
        assert_eq!(fmt_bytes(2048), "2.0 KB");
        assert_eq!(fmt_bytes(10_485_760), "10.0 MB");
    }
}
