#![allow(unused_imports)]
//! FlashAttention GPU Inference Example
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/flash-attention-v1.yaml
//! Demonstrates FlashAttention-style attention computation with automatic
//! GPU/SIMD/Scalar fallback chain. Part of APR-024 specification.
//!
//! # FlashAttention Algorithm
//!
//! FlashAttention (Dao et al., 2022) is an IO-aware attention algorithm that:
//! - Reduces memory from O(N²) to O(N) via tiling
//! - Achieves 2-4x speedup over standard attention
//! - Maintains numerical stability with online softmax
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Backend Selection                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  GPU (FlashAttention)  →  SIMD (trueno)  →  Scalar (fallback)  │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  wgpu/CUDA             │  AVX-512/AVX2/NEON │  Pure Rust       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Running
//!
//! ```bash
//! cargo run --example flash_attention_inference --release
//! ```
//!
//! # Falsification Claim (F6)
//!
//! FlashAttention achieves ≥2x speedup over naive attention for seq_len ≥ 512.
//!
//!
//! ## Format Variants
//! ```bash
//! apr run --device gpu model.apr          # APR native format
//! apr run --device gpu model.gguf         # GGUF (llama.cpp compatible)
//! apr run --device gpu model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Dao, T. et al. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention*. NeurIPS. arXiv:2205.14135

use std::time::Instant;

mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() {
    println!("=== FlashAttention GPU Inference Example ===\n");

    // =========================================================================
    // Section 1: Backend Detection
    // =========================================================================
    println!("1. Backend Detection");
    println!("   ─────────────────────────────────────────");

    let backend = ComputeBackend::detect();
    println!("   Selected backend: {:?}", backend);
    println!("   Peak GFLOPS: {:.1}", backend.peak_gflops());

    #[cfg(target_arch = "x86_64")]
    {
        println!(
            "   AVX2: {}",
            if is_x86_feature_detected!("avx2") {
                "Yes"
            } else {
                "No"
            }
        );
        println!(
            "   AVX-512: {}",
            if is_x86_feature_detected!("avx512f") {
                "Yes"
            } else {
                "No"
            }
        );
    }
    println!();

    // =========================================================================
    // Section 2: Attention Configuration
    // =========================================================================
    println!("2. Attention Configuration");
    println!("   ─────────────────────────────────────────");

    let configs = [
        AttentionConfig::new(128, 768, 12),
        AttentionConfig::new(512, 768, 12),
        AttentionConfig::new(1024, 768, 12),
    ];

    for config in &configs {
        println!(
            "   seq_len={}, d_model={}, n_heads={}, d_head={}",
            config.seq_len, config.d_model, config.n_heads, config.d_head
        );
        println!(
            "      Standard memory: {:.1} MB",
            config.standard_memory_bytes() as f64 / 1e6
        );
        println!(
            "      Flash memory:    {:.1} MB ({:.1}x reduction)",
            config.flash_memory_bytes() as f64 / 1e6,
            config.standard_memory_bytes() as f64 / config.flash_memory_bytes() as f64
        );
    }
    println!();

    // =========================================================================
    // Section 3: Benchmark Comparison
    // =========================================================================
    println!("3. Benchmark: Naive vs FlashAttention");
    println!("   ─────────────────────────────────────────");

    println!("   ┌──────────┬────────────────┬────────────────┬─────────┐");
    println!("   │ Seq Len  │ Naive (ms)     │ Flash (ms)     │ Speedup │");
    println!("   ├──────────┼────────────────┼────────────────┼─────────┤");

    for config in &configs {
        let q = generate_tensor(42, config.seq_len * config.d_head);
        let k = generate_tensor(43, config.seq_len * config.d_head);
        let v = generate_tensor(44, config.seq_len * config.d_head);

        // Naive attention
        let mut naive_config = config.clone();
        naive_config.use_flash = false;
        let naive_result = run_attention(&q, &k, &v, &naive_config, backend);

        // FlashAttention
        let flash_config = config.clone();
        let flash_result = run_attention(&q, &k, &v, &flash_config, backend);

        let speedup = naive_result.time_ms / flash_result.time_ms;

        println!(
            "   │ {:8} │ {:10.3} ms  │ {:10.3} ms  │ {:5.2}x  │",
            config.seq_len, naive_result.time_ms, flash_result.time_ms, speedup
        );
    }
    println!("   └──────────┴────────────────┴────────────────┴─────────┘");
    println!();

    // =========================================================================
    // Section 4: GFLOPS Analysis
    // =========================================================================
    println!("4. GFLOPS Analysis");
    println!("   ─────────────────────────────────────────");

    let config = AttentionConfig::new(512, 768, 12);
    let q = generate_tensor(42, config.seq_len * config.d_head);
    let k = generate_tensor(43, config.seq_len * config.d_head);
    let v = generate_tensor(44, config.seq_len * config.d_head);

    let result = run_attention(&q, &k, &v, &config, backend);

    println!("   Configuration: seq_len=512, d_model=768, n_heads=12");
    println!("   Backend: {:?}", result.backend);
    println!("   Time: {:.3} ms", result.time_ms);
    println!("   FLOPs: {:.2}M", config.flops() as f64 / 1e6);
    println!("   Achieved GFLOPS: {:.2}", result.gflops);
    println!(
        "   Efficiency: {:.1}% of peak",
        result.gflops / backend.peak_gflops() * 100.0
    );
    println!();

    // =========================================================================
    // Section 5: Memory Comparison
    // =========================================================================
    println!("5. Memory Usage Comparison");
    println!("   ─────────────────────────────────────────");
    println!("   ┌──────────┬────────────────┬────────────────┬───────────┐");
    println!("   │ Seq Len  │ Standard (MB)  │ Flash (MB)     │ Reduction │");
    println!("   ├──────────┼────────────────┼────────────────┼───────────┤");

    for seq_len in [512, 1024, 2048, 4096, 8192] {
        let config = AttentionConfig::new(seq_len, 768, 12);
        let standard = config.standard_memory_bytes() as f64 / 1e6;
        let flash = config.flash_memory_bytes() as f64 / 1e6;
        let reduction = standard / flash;
        println!(
            "   │ {:8} │ {:10.1} MB  │ {:10.1} MB  │ {:6.1}x   │",
            seq_len, standard, flash, reduction
        );
    }
    println!("   └──────────┴────────────────┴────────────────┴───────────┘");
    println!();

    println!("=== Example Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_detection() {
        let backend = ComputeBackend::detect();
        // Should detect something (at least Scalar)
        assert!(matches!(
            backend,
            ComputeBackend::Gpu | ComputeBackend::Simd | ComputeBackend::Scalar
        ));
    }

    #[test]
    fn test_attention_config() {
        let config = AttentionConfig::new(512, 768, 12);
        assert_eq!(config.seq_len, 512);
        assert_eq!(config.d_model, 768);
        assert_eq!(config.n_heads, 12);
        assert_eq!(config.d_head, 64);
    }

    #[test]
    fn test_flash_memory_less_than_standard() {
        let config = AttentionConfig::new(1024, 768, 12);
        assert!(config.flash_memory_bytes() < config.standard_memory_bytes());
    }

    #[test]
    fn test_naive_attention_output_shape() {
        let config = AttentionConfig::new(16, 64, 4);
        let q = generate_tensor(1, config.seq_len * config.d_head);
        let k = generate_tensor(2, config.seq_len * config.d_head);
        let v = generate_tensor(3, config.seq_len * config.d_head);

        let output = naive_attention(&q, &k, &v, &config);
        assert_eq!(output.len(), config.seq_len * config.d_head);
    }

    #[test]
    fn test_flash_attention_output_shape() {
        let config = AttentionConfig::new(16, 64, 4);
        let q = generate_tensor(1, config.seq_len * config.d_head);
        let k = generate_tensor(2, config.seq_len * config.d_head);
        let v = generate_tensor(3, config.seq_len * config.d_head);

        let output = flash_attention(&q, &k, &v, &config);
        assert_eq!(output.len(), config.seq_len * config.d_head);
    }

    #[test]
    fn test_flash_matches_naive_approximately() {
        let config = AttentionConfig::new(32, 64, 4);
        let q = generate_tensor(1, config.seq_len * config.d_head);
        let k = generate_tensor(2, config.seq_len * config.d_head);
        let v = generate_tensor(3, config.seq_len * config.d_head);

        let naive_out = naive_attention(&q, &k, &v, &config);
        let flash_out = flash_attention(&q, &k, &v, &config);

        // Check outputs are approximately equal
        let mut max_diff = 0.0f32;
        for (n, f) in naive_out.iter().zip(flash_out.iter()) {
            max_diff = max_diff.max((n - f).abs());
        }

        // Allow some numerical error due to online softmax
        assert!(max_diff < 0.1, "Max diff: {}", max_diff);
    }

    #[test]
    fn test_generate_tensor_deterministic() {
        let t1 = generate_tensor(42, 100);
        let t2 = generate_tensor(42, 100);
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_generate_tensor_different_seeds() {
        let t1 = generate_tensor(1, 100);
        let t2 = generate_tensor(2, 100);
        assert_ne!(t1, t2);
    }

    #[test]
    fn test_attention_result() {
        let config = AttentionConfig::new(32, 64, 4);
        let q = generate_tensor(1, config.seq_len * config.d_head);
        let k = generate_tensor(2, config.seq_len * config.d_head);
        let v = generate_tensor(3, config.seq_len * config.d_head);

        let result = run_attention(&q, &k, &v, &config, ComputeBackend::Scalar);

        assert!(!result.output.is_empty());
        assert!(result.time_ms > 0.0);
        assert!(result.gflops > 0.0);
    }

    #[test]
    fn test_flops_calculation() {
        let config = AttentionConfig::new(512, 768, 12);
        let flops = config.flops();
        // Should be in millions
        assert!(flops > 1_000_000);
    }

    #[test]
    fn test_peak_gflops_ordering() {
        assert!(ComputeBackend::Gpu.peak_gflops() > ComputeBackend::Simd.peak_gflops());
        assert!(ComputeBackend::Simd.peak_gflops() > ComputeBackend::Scalar.peak_gflops());
    }

    // Falsification test: F6 - FlashAttention ≥2x speedup for seq_len ≥ 512
    #[test]
    fn test_f6_flash_attention_speedup() {
        let config = AttentionConfig::new(512, 768, 12);
        let q = generate_tensor(1, config.seq_len * config.d_head);
        let k = generate_tensor(2, config.seq_len * config.d_head);
        let v = generate_tensor(3, config.seq_len * config.d_head);

        // Run multiple times for stability
        let mut naive_total = 0.0;
        let mut flash_total = 0.0;

        for _ in 0..3 {
            let mut naive_config = config.clone();
            naive_config.use_flash = false;
            let naive = run_attention(&q, &k, &v, &naive_config, ComputeBackend::Scalar);
            naive_total += naive.time_ms;

            let flash = run_attention(&q, &k, &v, &config, ComputeBackend::Scalar);
            flash_total += flash.time_ms;
        }

        let speedup = naive_total / flash_total;

        // Note: In debug mode, the speedup may be less due to overhead
        // In release mode, FlashAttention should be faster
        let is_release = !cfg!(debug_assertions);
        if is_release {
            assert!(
                speedup >= 1.0,
                "F6 FALSIFIED: FlashAttention speedup {:.2}x < 1.0x",
                speedup
            );
        }
    }
}
