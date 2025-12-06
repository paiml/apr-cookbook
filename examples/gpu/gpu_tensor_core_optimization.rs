//! # Recipe: Tensor Core Optimization
//!
//! **Category**: GPU Acceleration
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
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Optimize for NVIDIA Tensor Cores with mixed precision.
//!
//! ## Run Command
//! ```bash
//! cargo run --example gpu_tensor_core_optimization
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("gpu_tensor_core_optimization")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Tensor Core optimization with mixed precision");
    println!();

    // Check Tensor Core support
    let tc_info = check_tensor_core_support();

    println!("Tensor Core Support:");
    println!("  Generation: {}", tc_info.generation);
    println!("  FP16 support: {}", tc_info.fp16_support);
    println!("  BF16 support: {}", tc_info.bf16_support);
    println!("  INT8 support: {}", tc_info.int8_support);
    println!("  Peak TFLOPS (FP16): {}", tc_info.peak_tflops_fp16);
    println!();

    // Benchmark different precisions
    let matrix_size = 4096;

    println!(
        "Matrix Multiplication Benchmark ({}x{})",
        matrix_size, matrix_size
    );
    println!("{:-<65}", "");
    println!(
        "{:<12} {:>12} {:>12} {:>12} {:>12}",
        "Precision", "Time(ms)", "TFLOPS", "Memory", "Accuracy"
    );
    println!("{:-<65}", "");

    let precisions = vec![
        Precision::FP32,
        Precision::FP16,
        Precision::BF16,
        Precision::INT8,
    ];

    let mut results = Vec::new();
    for precision in &precisions {
        let result = benchmark_precision(*precision, matrix_size)?;
        results.push(result.clone());

        println!(
            "{:<12} {:>10.2}ms {:>10.1} {:>10}MB {:>12}",
            format!("{:?}", precision),
            result.time_ms,
            result.tflops,
            result.memory_mb,
            result.accuracy_status
        );
    }
    println!("{:-<65}", "");

    // Record best results
    let fp16_result = results.iter().find(|r| r.precision == Precision::FP16);
    if let Some(r) = fp16_result {
        ctx.record_float_metric("fp16_tflops", r.tflops);
        ctx.record_float_metric("fp16_time_ms", r.time_ms);
    }

    // Speedup analysis
    let fp32_time = results
        .iter()
        .find(|r| r.precision == Precision::FP32)
        .map_or(1.0, |r| r.time_ms);

    println!();
    println!("Speedup over FP32:");
    for result in &results {
        let speedup = fp32_time / result.time_ms;
        println!("  {:?}: {:.2}x", result.precision, speedup);
    }

    // Memory savings
    println!();
    println!("Memory Savings over FP32:");
    let fp32_memory = results
        .iter()
        .find(|r| r.precision == Precision::FP32)
        .map_or(1, |r| r.memory_mb);

    for result in &results {
        let savings = ((f64::from(fp32_memory) - f64::from(result.memory_mb)) / f64::from(fp32_memory)) * 100.0;
        if savings > 0.0 {
            println!("  {:?}: {:.0}% reduction", result.precision, savings);
        }
    }

    // Save results
    let results_path = ctx.path("tensor_core_benchmark.json");
    save_results(&results_path, &results)?;
    println!();
    println!("Results saved to: {:?}", results_path);

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TensorCoreInfo {
    generation: String,
    fp16_support: bool,
    bf16_support: bool,
    int8_support: bool,
    peak_tflops_fp16: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum Precision {
    FP32,
    FP16,
    BF16,
    INT8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResult {
    precision: Precision,
    time_ms: f64,
    tflops: f64,
    memory_mb: u32,
    accuracy_status: String,
}

fn check_tensor_core_support() -> TensorCoreInfo {
    // Simulated Tensor Core detection (Ampere generation)
    TensorCoreInfo {
        generation: "Ampere (Simulated)".to_string(),
        fp16_support: true,
        bf16_support: true,
        int8_support: true,
        peak_tflops_fp16: 312,
    }
}

fn benchmark_precision(precision: Precision, size: u32) -> Result<BenchmarkResult> {
    // FLOPs for matrix multiplication: 2 * N^3
    let flops = 2.0 * f64::from(size).powi(3);

    // Simulated performance based on precision
    let (tflops, memory_factor, accuracy) = match precision {
        Precision::FP32 => (19.5, 4.0, "exact"),
        Precision::FP16 => (156.0, 2.0, "~0.1% loss"),
        Precision::BF16 => (156.0, 2.0, "~0.05% loss"),
        Precision::INT8 => (312.0, 1.0, "~1% loss"),
    };

    let time_ms = (flops / (tflops * 1e12)) * 1000.0;
    let memory_mb =
        ((f64::from(size) * f64::from(size) * memory_factor) / (1024.0 * 1024.0)) as u32 * 2 + 10;

    Ok(BenchmarkResult {
        precision,
        time_ms,
        tflops,
        memory_mb,
        accuracy_status: accuracy.to_string(),
    })
}

fn save_results(path: &std::path::Path, results: &[BenchmarkResult]) -> Result<()> {
    let json = serde_json::to_string_pretty(results)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    std::fs::write(path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_core_info() {
        let info = check_tensor_core_support();
        assert!(info.fp16_support);
        assert!(info.peak_tflops_fp16 > 0);
    }

    #[test]
    fn test_benchmark_fp32() {
        let result = benchmark_precision(Precision::FP32, 1024).unwrap();
        assert_eq!(result.precision, Precision::FP32);
        assert!(result.time_ms > 0.0);
    }

    #[test]
    fn test_fp16_faster_than_fp32() {
        let fp32 = benchmark_precision(Precision::FP32, 1024).unwrap();
        let fp16 = benchmark_precision(Precision::FP16, 1024).unwrap();

        assert!(fp16.time_ms < fp32.time_ms);
    }

    #[test]
    fn test_int8_fastest() {
        let fp32 = benchmark_precision(Precision::FP32, 1024).unwrap();
        let int8 = benchmark_precision(Precision::INT8, 1024).unwrap();

        assert!(int8.time_ms < fp32.time_ms);
    }

    #[test]
    fn test_memory_savings() {
        let fp32 = benchmark_precision(Precision::FP32, 1024).unwrap();
        let fp16 = benchmark_precision(Precision::FP16, 1024).unwrap();

        assert!(fp16.memory_mb < fp32.memory_mb);
    }

    #[test]
    fn test_deterministic() {
        let r1 = benchmark_precision(Precision::FP16, 1024).unwrap();
        let r2 = benchmark_precision(Precision::FP16, 1024).unwrap();

        assert_eq!(r1.time_ms, r2.time_ms);
        assert_eq!(r1.tflops, r2.tflops);
    }

    #[test]
    fn test_save_results() {
        let ctx = RecipeContext::new("test_tc_save").unwrap();
        let path = ctx.path("results.json");

        let results = vec![benchmark_precision(Precision::FP16, 512).unwrap()];
        save_results(&path, &results).unwrap();

        assert!(path.exists());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_fp16_always_faster(size in 256u32..2048) {
            let fp32 = benchmark_precision(Precision::FP32, size).unwrap();
            let fp16 = benchmark_precision(Precision::FP16, size).unwrap();

            prop_assert!(fp16.time_ms < fp32.time_ms);
        }

        #[test]
        fn prop_tflops_positive(size in 128u32..1024) {
            for precision in [Precision::FP32, Precision::FP16, Precision::BF16, Precision::INT8] {
                let result = benchmark_precision(precision, size).unwrap();
                prop_assert!(result.tflops > 0.0);
            }
        }

        #[test]
        fn prop_larger_size_more_time(size1 in 256u32..512, size2 in 513u32..1024) {
            let r1 = benchmark_precision(Precision::FP16, size1).unwrap();
            let r2 = benchmark_precision(Precision::FP16, size2).unwrap();

            prop_assert!(r2.time_ms > r1.time_ms);
        }
    }
}
