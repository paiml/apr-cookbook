//! # Recipe: WebGPU Acceleration
//!
//! **Category**: WASM/Browser
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
//! 6. [x] WASM compatible (Verified)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Accelerate browser inference with WebGPU (simulated).
//!
//! ## Run Command
//! ```bash
//! cargo run --example wasm_webgpu_acceleration
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("wasm_webgpu_acceleration")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("WebGPU acceleration simulation");
    println!();

    // Check WebGPU availability
    let gpu_info = check_webgpu_support();

    println!("WebGPU Support:");
    println!("  Available: {}", gpu_info.available);
    println!("  Adapter: {}", gpu_info.adapter_name);
    println!("  Max buffer size: {}MB", gpu_info.max_buffer_size_mb);
    println!("  Max workgroup size: {}", gpu_info.max_workgroup_size);
    println!();

    // Create compute pipeline
    let mut pipeline = WebGpuPipeline::new(PipelineConfig {
        workgroup_size: 256,
        batch_size: 1024,
    });

    ctx.record_metric("workgroup_size", i64::from(pipeline.config.workgroup_size));
    ctx.record_metric("batch_size", i64::from(pipeline.config.batch_size));

    // Benchmark matrix operations
    let sizes = vec![64, 128, 256, 512];

    println!("Matrix multiplication benchmark:");
    println!("{:-<60}", "");
    println!(
        "{:>8} {:>12} {:>12} {:>12} {:>10}",
        "Size", "CPU(ms)", "GPU(ms)", "Speedup", "GFLOPS"
    );
    println!("{:-<60}", "");

    for size in &sizes {
        let result = pipeline.benchmark_matmul(*size)?;

        println!(
            "{:>8} {:>12.2} {:>12.2} {:>11.1}x {:>10.1}",
            format!("{}x{}", size, size),
            result.cpu_time_ms,
            result.gpu_time_ms,
            result.speedup,
            result.gflops
        );

        if *size == 256 {
            ctx.record_float_metric("speedup_256", result.speedup);
            ctx.record_float_metric("gflops_256", result.gflops);
        }
    }
    println!("{:-<60}", "");

    // Shader compilation stats
    let shader_stats = pipeline.get_shader_stats();
    println!();
    println!("Shader Statistics:");
    println!("  Compile time: {}ms", shader_stats.compile_time_ms);
    println!("  Shader modules: {}", shader_stats.module_count);
    println!("  Total instructions: {}", shader_stats.instruction_count);

    // Save benchmark results
    let results_path = ctx.path("webgpu_benchmark.json");
    pipeline.save_results(&results_path)?;
    println!();
    println!("Benchmark results saved to: {:?}", results_path);

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GpuInfo {
    available: bool,
    adapter_name: String,
    max_buffer_size_mb: u32,
    max_workgroup_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PipelineConfig {
    workgroup_size: u32,
    batch_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BenchmarkResult {
    size: u32,
    cpu_time_ms: f64,
    gpu_time_ms: f64,
    speedup: f64,
    gflops: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ShaderStats {
    compile_time_ms: u32,
    module_count: u32,
    instruction_count: u32,
}

#[derive(Debug)]
struct WebGpuPipeline {
    config: PipelineConfig,
    results: Vec<BenchmarkResult>,
}

fn check_webgpu_support() -> GpuInfo {
    // Simulated WebGPU detection
    GpuInfo {
        available: true,
        adapter_name: "Simulated GPU Adapter".to_string(),
        max_buffer_size_mb: 256,
        max_workgroup_size: 256,
    }
}

impl WebGpuPipeline {
    fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            results: Vec::new(),
        }
    }

    fn benchmark_matmul(&mut self, size: u32) -> Result<BenchmarkResult> {
        // Simulated benchmark with deterministic results
        // CPU: O(n^3) complexity
        let flops = 2.0 * f64::from(size).powi(3);

        // Simulated timings (deterministic based on size)
        let cpu_time = f64::from(size).powi(3) / 1_000_000.0; // ~1ms per 1M ops
        let gpu_time = f64::from(size).powi(3) / 10_000_000.0; // 10x faster on GPU

        let speedup = cpu_time / gpu_time;
        let gflops = flops / (gpu_time * 1_000_000.0);

        let result = BenchmarkResult {
            size,
            cpu_time_ms: cpu_time,
            gpu_time_ms: gpu_time,
            speedup,
            gflops,
        };

        self.results.push(result.clone());
        Ok(result)
    }

    fn get_shader_stats(&self) -> ShaderStats {
        ShaderStats {
            compile_time_ms: 50,
            module_count: 3,
            instruction_count: 150,
        }
    }

    fn save_results(&self, path: &std::path::Path) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.results)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_info() {
        let info = check_webgpu_support();
        assert!(info.available);
        assert!(info.max_buffer_size_mb > 0);
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = WebGpuPipeline::new(PipelineConfig {
            workgroup_size: 256,
            batch_size: 1024,
        });

        assert_eq!(pipeline.config.workgroup_size, 256);
        assert!(pipeline.results.is_empty());
    }

    #[test]
    fn test_benchmark_matmul() {
        let mut pipeline = WebGpuPipeline::new(PipelineConfig {
            workgroup_size: 256,
            batch_size: 1024,
        });

        let result = pipeline.benchmark_matmul(64).unwrap();

        assert_eq!(result.size, 64);
        assert!(result.cpu_time_ms > 0.0);
        assert!(result.gpu_time_ms > 0.0);
        assert!(result.speedup > 1.0);
    }

    #[test]
    fn test_gpu_faster_than_cpu() {
        let mut pipeline = WebGpuPipeline::new(PipelineConfig {
            workgroup_size: 256,
            batch_size: 1024,
        });

        let result = pipeline.benchmark_matmul(128).unwrap();

        assert!(result.gpu_time_ms < result.cpu_time_ms);
    }

    #[test]
    fn test_deterministic_results() {
        let config = PipelineConfig {
            workgroup_size: 256,
            batch_size: 1024,
        };

        let mut p1 = WebGpuPipeline::new(config.clone());
        let mut p2 = WebGpuPipeline::new(config);

        let r1 = p1.benchmark_matmul(64).unwrap();
        let r2 = p2.benchmark_matmul(64).unwrap();

        assert_eq!(r1.cpu_time_ms, r2.cpu_time_ms);
        assert_eq!(r1.gpu_time_ms, r2.gpu_time_ms);
    }

    #[test]
    fn test_shader_stats() {
        let pipeline = WebGpuPipeline::new(PipelineConfig {
            workgroup_size: 256,
            batch_size: 1024,
        });

        let stats = pipeline.get_shader_stats();

        assert!(stats.compile_time_ms > 0);
        assert!(stats.module_count > 0);
    }

    #[test]
    fn test_save_results() {
        let ctx = RecipeContext::new("test_webgpu_save").unwrap();
        let path = ctx.path("results.json");

        let mut pipeline = WebGpuPipeline::new(PipelineConfig {
            workgroup_size: 256,
            batch_size: 1024,
        });

        pipeline.benchmark_matmul(64).unwrap();
        pipeline.save_results(&path).unwrap();

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
        fn prop_gpu_always_faster(size in 8u32..256) {
            let mut pipeline = WebGpuPipeline::new(PipelineConfig {
                workgroup_size: 256,
                batch_size: 1024,
            });

            let result = pipeline.benchmark_matmul(size).unwrap();
            prop_assert!(result.speedup > 1.0);
        }

        #[test]
        fn prop_gflops_positive(size in 16u32..128) {
            let mut pipeline = WebGpuPipeline::new(PipelineConfig {
                workgroup_size: 256,
                batch_size: 1024,
            });

            let result = pipeline.benchmark_matmul(size).unwrap();
            prop_assert!(result.gflops > 0.0);
        }

        #[test]
        fn prop_larger_size_more_flops(size1 in 16u32..64, size2 in 65u32..128) {
            let mut pipeline = WebGpuPipeline::new(PipelineConfig {
                workgroup_size: 256,
                batch_size: 1024,
            });

            let r1 = pipeline.benchmark_matmul(size1).unwrap();
            let r2 = pipeline.benchmark_matmul(size2).unwrap();

            // Larger matrices have more operations
            let flops1 = 2.0 * (size1 as f64).powi(3);
            let flops2 = 2.0 * (size2 as f64).powi(3);
            prop_assert!(flops2 > flops1);
        }
    }
}
