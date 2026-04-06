#![allow(unused_imports)]
//! # Recipe: GPU PTX Kernel Analysis
//!
//! **Category**: GPU Acceleration
//! **CLI Equivalent**: `apr ptx`
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/flash-attention-v1.yaml
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
//! 10. [x] No unsafe code
//!
//! ## Learning Objective
//! Mirror `apr ptx_map` + `apr ptx_explain` — map a 7B model inference to
//! its 12-step CUDA PTX kernel execution sequence, compute roofline
//! analysis per kernel, and detect performance issues (low occupancy,
//! excessive shared memory, uncoalesced access patterns).
//!
//! ## Run Command
//! ```bash
//! cargo run --example gpu_ptx_analysis
//! ```
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

use apr_cookbook::prelude::*;
use std::fmt;

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("gpu_ptx_analysis")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("CUDA PTX kernel mapping and roofline analysis for 7B model inference");
    println!();

    let device = DeviceSpec::a100_80gb();
    let kernels = build_kernel_sequence();

    // Section 1: Kernel map table
    section_kernel_map(&kernels);

    // Section 2: Roofline analysis
    section_roofline(&kernels, &device);

    // Section 3: Utilization summary
    section_utilization(&kernels, &device);

    // Section 4: Performance warnings
    let warnings = section_warnings(&kernels, &device);

    // Record metrics
    let total_flops: u64 = kernels.iter().map(|k| k.flops).sum();
    ctx.record_metric("kernel_count", kernels.len() as i64);
    ctx.record_metric("total_flops", total_flops as i64);
    ctx.record_metric("warning_count", warnings.len() as i64);
    ctx.record_float_metric("ridge_point", device.ridge_point());

    let mem_bound_count = kernels
        .iter()
        .filter(|k| compute_roofline(k, &device).memory_bound)
        .count();
    ctx.record_metric("memory_bound_kernels", mem_bound_count as i64);
    ctx.record_metric(
        "compute_bound_kernels",
        (kernels.len() - mem_bound_count) as i64,
    );

    ctx.report()?;

    println!();
    println!("=== GPU PTX Analysis Recipe Complete ===");

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_sequence_has_12_steps() {
        let kernels = build_kernel_sequence();
        assert_eq!(kernels.len(), 12);
    }

    #[test]
    fn test_kernel_names_are_unique() {
        let kernels = build_kernel_sequence();
        let mut names: Vec<&str> = kernels.iter().map(|k| k.name.as_str()).collect();
        let original_len = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), original_len, "kernel names must be unique");
    }

    #[test]
    fn test_kernel_sequence_expected_order() {
        let kernels = build_kernel_sequence();
        let names: Vec<&str> = kernels.iter().map(|k| k.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "embed_lookup",
                "qkv_projection",
                "rotary_embedding",
                "flash_attention",
                "attention_output",
                "gate_proj",
                "up_proj",
                "silu_activation",
                "down_proj",
                "rmsnorm",
                "lm_head",
                "softmax_sample",
            ],
        );
    }

    #[test]
    fn test_total_threads_computation() {
        let kernel = KernelStep {
            name: "test".to_string(),
            grid_dim: [4, 2, 1],
            block_dim: [128, 1, 1],
            registers: 32,
            shared_mem_bytes: 0,
            flops: 1000,
        };
        // 4*2*1 * 128*1*1 = 1024
        assert_eq!(kernel.total_threads(), 1024);
    }

    #[test]
    fn test_roofline_memory_bound_kernel() {
        let device = DeviceSpec::a100_80gb();
        // Element-wise kernel: low arithmetic intensity
        let kernel = KernelStep {
            name: "elementwise".to_string(),
            grid_dim: [1024, 1, 1],
            block_dim: [256, 1, 1],
            registers: 16,
            shared_mem_bytes: 0,
            flops: 1024 * 256, // low flops relative to data movement
        };
        let metrics = compute_roofline(&kernel, &device);
        assert!(
            metrics.memory_bound,
            "element-wise kernel should be memory-bound",
        );
    }

    #[test]
    fn test_roofline_compute_bound_kernel() {
        let device = DeviceSpec::a100_80gb();
        // Large tiled GEMM: shared memory staging gives sqrt(flops)-based
        // byte estimate, yielding high arithmetic intensity.
        let kernel = KernelStep {
            name: "big_gemm".to_string(),
            grid_dim: [128, 128, 1],
            block_dim: [128, 1, 1],
            registers: 48,
            shared_mem_bytes: 32 * 1024,   // tiled with shared mem
            flops: 2 * 4096 * 4096 * 4096, // ~274 GFLOP
        };
        let metrics = compute_roofline(&kernel, &device);
        // With tiling heuristic: bytes = sqrt(274G)*4 ~ 2M, AI ~ 130k
        assert!(
            metrics.compute_bound,
            "large tiled GEMM should be compute-bound; AI={:.2}",
            metrics.arithmetic_intensity,
        );
    }

    #[test]
    fn test_warning_low_occupancy() {
        let device = DeviceSpec::a100_80gb();
        let kernel = KernelStep {
            name: "high_reg".to_string(),
            grid_dim: [1, 1, 1],
            block_dim: [128, 1, 1],
            registers: 128, // way above 64
            shared_mem_bytes: 0,
            flops: 100,
        };
        let warnings = detect_warnings(&kernel, &device);
        assert!(
            warnings
                .iter()
                .any(|w| w.category == WarningCategory::LowOccupancy),
            "should warn about low occupancy for high register usage",
        );
    }

    #[test]
    fn test_warning_excessive_shared_memory() {
        let device = DeviceSpec::a100_80gb();
        let kernel = KernelStep {
            name: "big_shmem".to_string(),
            grid_dim: [1, 1, 1],
            block_dim: [128, 1, 1],
            registers: 32,
            shared_mem_bytes: 64 * 1024, // exceeds 48KB
            flops: 100,
        };
        let warnings = detect_warnings(&kernel, &device);
        assert!(
            warnings
                .iter()
                .any(|w| w.category == WarningCategory::ExcessiveSharedMemory),
            "should warn about excessive shared memory",
        );
    }

    #[test]
    fn test_flash_attention_triggers_both_warnings() {
        let device = DeviceSpec::a100_80gb();
        let kernels = build_kernel_sequence();
        let flash = kernels
            .iter()
            .find(|k| k.name == "flash_attention")
            .expect("flash_attention must exist in kernel sequence");
        let warnings = detect_warnings(flash, &device);

        let has_occupancy = warnings
            .iter()
            .any(|w| w.category == WarningCategory::LowOccupancy);
        let has_shmem = warnings
            .iter()
            .any(|w| w.category == WarningCategory::ExcessiveSharedMemory);

        assert!(
            has_occupancy,
            "flash_attention should trigger low occupancy warning"
        );
        assert!(
            has_shmem,
            "flash_attention should trigger excessive shmem warning"
        );
    }

    #[test]
    fn test_format_flops_scales() {
        assert_eq!(format_flops(500), "500 FLOP");
        assert_eq!(format_flops(1_500_000), "1.50 MFLOP");
        assert_eq!(format_flops(2_500_000_000), "2.50 GFLOP");
        assert_eq!(format_flops(3_000_000_000_000), "3.00 TFLOP");
    }

    #[test]
    fn test_format_bytes_scales() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(2048), "2.05 KB");
        assert_eq!(format_bytes(5_000_000), "5.00 MB");
        assert_eq!(format_bytes(1_500_000_000), "1.50 GB");
    }
}
