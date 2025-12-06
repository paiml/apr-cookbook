//! # Recipe: Multi-GPU Inference
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
//! Distribute inference across multiple GPUs.
//!
//! ## Run Command
//! ```bash
//! cargo run --example gpu_multi_gpu_inference
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("gpu_multi_gpu_inference")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Multi-GPU inference distribution");
    println!();

    // Detect GPUs
    let gpus = detect_gpus();
    ctx.record_metric("gpu_count", gpus.len() as i64);

    println!("Detected GPUs:");
    for gpu in &gpus {
        println!("  GPU {}: {} ({}GB)", gpu.id, gpu.name, gpu.memory_gb);
    }
    println!();

    // Configure multi-GPU strategy
    let strategies = vec![
        DistributionStrategy::DataParallel,
        DistributionStrategy::PipelineParallel,
        DistributionStrategy::TensorParallel,
    ];

    // Model config
    let model_config = ModelConfig {
        total_params_b: 7.0, // 7B parameter model
        layers: 32,
        batch_size: 64,
    };

    println!(
        "Model: {:.0}B parameters, {} layers",
        model_config.total_params_b, model_config.layers
    );
    println!("Batch size: {}", model_config.batch_size);
    println!();

    println!("Strategy Comparison ({} GPUs):", gpus.len());
    println!("{:-<70}", "");
    println!(
        "{:<20} {:>12} {:>12} {:>12} {:>10}",
        "Strategy", "Time(ms)", "Throughput", "Efficiency", "Memory/GPU"
    );
    println!("{:-<70}", "");

    let mut results = Vec::new();
    for strategy in &strategies {
        let result = benchmark_strategy(&gpus, &model_config, *strategy)?;
        results.push(result.clone());

        println!(
            "{:<20} {:>10.2}ms {:>10.0}/s {:>10.0}% {:>8}GB",
            format!("{:?}", strategy),
            result.total_time_ms,
            result.throughput,
            result.efficiency * 100.0,
            result.memory_per_gpu_gb
        );
    }
    println!("{:-<70}", "");

    // Best strategy
    let best = results
        .iter()
        .max_by(|a, b| {
            a.throughput
                .partial_cmp(&b.throughput)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| CookbookError::invalid_format("No results"))?;

    ctx.record_float_metric("best_throughput", best.throughput);
    ctx.record_float_metric("best_efficiency", best.efficiency);

    println!();
    println!("Best Strategy: {:?}", best.strategy);
    println!("  Throughput: {:.0} samples/sec", best.throughput);
    println!("  Efficiency: {:.0}%", best.efficiency * 100.0);

    // Scaling analysis
    println!();
    println!("Scaling Analysis:");
    let single_gpu_throughput = benchmark_strategy(
        &gpus[..1],
        &model_config,
        DistributionStrategy::DataParallel,
    )?
    .throughput;
    let multi_gpu_throughput = best.throughput;
    let scaling_factor = multi_gpu_throughput / single_gpu_throughput;

    println!("  Single GPU: {:.0} samples/sec", single_gpu_throughput);
    println!(
        "  {} GPUs: {:.0} samples/sec",
        gpus.len(),
        multi_gpu_throughput
    );
    println!(
        "  Scaling factor: {:.2}x (ideal: {}x)",
        scaling_factor,
        gpus.len()
    );

    // Save results
    let results_path = ctx.path("multi_gpu_benchmark.json");
    save_results(&results_path, &results)?;
    println!();
    println!("Results saved to: {:?}", results_path);

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GpuDevice {
    id: u32,
    name: String,
    memory_gb: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelConfig {
    total_params_b: f64,
    layers: u32,
    batch_size: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum DistributionStrategy {
    DataParallel,
    PipelineParallel,
    TensorParallel,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResult {
    strategy: DistributionStrategy,
    total_time_ms: f64,
    throughput: f64,
    efficiency: f64,
    memory_per_gpu_gb: u32,
}

fn detect_gpus() -> Vec<GpuDevice> {
    // Simulated 4-GPU setup
    (0..4)
        .map(|id| GpuDevice {
            id,
            name: format!("GPU {} (Simulated)", id),
            memory_gb: 24,
        })
        .collect()
}

fn benchmark_strategy(
    gpus: &[GpuDevice],
    model: &ModelConfig,
    strategy: DistributionStrategy,
) -> Result<BenchmarkResult> {
    let gpu_count = gpus.len() as f64;

    // Base time for single GPU
    let base_time_ms = model.total_params_b * 10.0 * f64::from(model.batch_size) / 1000.0;

    // Strategy-specific performance characteristics
    let (speedup, _overhead, memory_factor) = match strategy {
        DistributionStrategy::DataParallel => {
            // Good scaling but communication overhead
            let overhead = 1.0 + 0.1 * (gpu_count - 1.0);
            (gpu_count / overhead, overhead, 1.0)
        }
        DistributionStrategy::PipelineParallel => {
            // Linear memory scaling but bubble overhead
            let bubble_overhead = 1.0 + (gpu_count - 1.0) / f64::from(model.layers);
            (
                gpu_count / bubble_overhead,
                bubble_overhead,
                1.0 / gpu_count,
            )
        }
        DistributionStrategy::TensorParallel => {
            // Best for large models but high communication
            let comm_overhead = 1.0 + 0.15 * (gpu_count - 1.0);
            (gpu_count / comm_overhead, comm_overhead, 1.0 / gpu_count)
        }
    };

    let total_time = base_time_ms / speedup;
    let throughput = (f64::from(model.batch_size) / total_time) * 1000.0;
    let efficiency = speedup / gpu_count;

    let base_memory = (model.total_params_b * 2.0) as u32; // ~2GB per B params
    let memory_per_gpu = ((f64::from(base_memory) * memory_factor) as u32).max(1);

    Ok(BenchmarkResult {
        strategy,
        total_time_ms: total_time,
        throughput,
        efficiency,
        memory_per_gpu_gb: memory_per_gpu,
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
    fn test_detect_gpus() {
        let gpus = detect_gpus();
        assert_eq!(gpus.len(), 4);
    }

    #[test]
    fn test_data_parallel() {
        let gpus = detect_gpus();
        let model = ModelConfig {
            total_params_b: 7.0,
            layers: 32,
            batch_size: 32,
        };

        let result = benchmark_strategy(&gpus, &model, DistributionStrategy::DataParallel).unwrap();

        assert!(result.throughput > 0.0);
        assert!(result.efficiency > 0.0 && result.efficiency <= 1.0);
    }

    #[test]
    fn test_pipeline_parallel_memory() {
        let gpus = detect_gpus();
        let model = ModelConfig {
            total_params_b: 7.0,
            layers: 32,
            batch_size: 32,
        };

        let data_parallel =
            benchmark_strategy(&gpus, &model, DistributionStrategy::DataParallel).unwrap();
        let pipeline =
            benchmark_strategy(&gpus, &model, DistributionStrategy::PipelineParallel).unwrap();

        // Pipeline parallel should use less memory per GPU
        assert!(pipeline.memory_per_gpu_gb <= data_parallel.memory_per_gpu_gb);
    }

    #[test]
    fn test_more_gpus_more_throughput() {
        let model = ModelConfig {
            total_params_b: 7.0,
            layers: 32,
            batch_size: 32,
        };

        let gpus_2: Vec<_> = detect_gpus().into_iter().take(2).collect();
        let gpus_4 = detect_gpus();

        let result_2 =
            benchmark_strategy(&gpus_2, &model, DistributionStrategy::DataParallel).unwrap();
        let result_4 =
            benchmark_strategy(&gpus_4, &model, DistributionStrategy::DataParallel).unwrap();

        assert!(result_4.throughput > result_2.throughput);
    }

    #[test]
    fn test_deterministic() {
        let gpus = detect_gpus();
        let model = ModelConfig {
            total_params_b: 7.0,
            layers: 32,
            batch_size: 32,
        };

        let r1 = benchmark_strategy(&gpus, &model, DistributionStrategy::TensorParallel).unwrap();
        let r2 = benchmark_strategy(&gpus, &model, DistributionStrategy::TensorParallel).unwrap();

        assert_eq!(r1.throughput, r2.throughput);
    }

    #[test]
    fn test_save_results() {
        let ctx = RecipeContext::new("test_multi_gpu_save").unwrap();
        let path = ctx.path("results.json");

        let results = vec![BenchmarkResult {
            strategy: DistributionStrategy::DataParallel,
            total_time_ms: 10.0,
            throughput: 100.0,
            efficiency: 0.9,
            memory_per_gpu_gb: 12,
        }];

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
        fn prop_efficiency_bounded(batch in 1u32..128) {
            let gpus = detect_gpus();
            let model = ModelConfig {
                total_params_b: 7.0,
                layers: 32,
                batch_size: batch,
            };

            for strategy in [
                DistributionStrategy::DataParallel,
                DistributionStrategy::PipelineParallel,
                DistributionStrategy::TensorParallel,
            ] {
                let result = benchmark_strategy(&gpus, &model, strategy).unwrap();
                prop_assert!(result.efficiency > 0.0);
                prop_assert!(result.efficiency <= 1.0);
            }
        }

        #[test]
        fn prop_throughput_positive(batch in 1u32..64) {
            let gpus = detect_gpus();
            let model = ModelConfig {
                total_params_b: 7.0,
                layers: 32,
                batch_size: batch,
            };

            let result = benchmark_strategy(&gpus, &model, DistributionStrategy::DataParallel).unwrap();
            prop_assert!(result.throughput > 0.0);
        }
    }
}
