#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use super::types::*;

#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use std::fmt;

///   hidden_dim=4096, n_heads=32, head_dim=128, ffn_dim=11008, vocab=32000
pub fn build_kernel_sequence() -> Vec<KernelStep> {
    vec![
        // 1. Embedding lookup (vocab=32000, dim=4096, batch*seq=2048)
        KernelStep {
            name: "embed_lookup".to_string(),
            grid_dim: [2048, 1, 1],
            block_dim: [256, 1, 1],
            registers: 24,
            shared_mem_bytes: 0,
            flops: 2048 * 4096, // trivial gather
        },
        // 2. QKV projection (4096 -> 3*4096, GEMM)
        KernelStep {
            name: "qkv_projection".to_string(),
            grid_dim: [64, 48, 1],
            block_dim: [128, 1, 1],
            registers: 48,
            shared_mem_bytes: 32 * 1024,
            flops: 2 * 2048 * 4096 * 12288, // 2*M*K*N
        },
        // 3. Rotary positional embedding
        KernelStep {
            name: "rotary_embedding".to_string(),
            grid_dim: [2048, 32, 1],
            block_dim: [128, 1, 1],
            registers: 32,
            shared_mem_bytes: 4096,
            flops: 2048 * 32 * 128 * 6, // sin/cos per head
        },
        // 4. Flash attention (causal, FP16)
        KernelStep {
            name: "flash_attention".to_string(),
            grid_dim: [2048, 32, 1],
            block_dim: [128, 1, 1],
            registers: 72, // high register pressure -- will trigger warning
            shared_mem_bytes: 52 * 1024, // exceeds 48KB -- will trigger warning
            flops: 2 * 2048 * 2048 * 128 * 32, // 2*S*S*D*H
        },
        // 5. Attention output projection (4096 -> 4096)
        KernelStep {
            name: "attention_output".to_string(),
            grid_dim: [64, 32, 1],
            block_dim: [128, 1, 1],
            registers: 48,
            shared_mem_bytes: 32 * 1024,
            flops: 2 * 2048 * 4096 * 4096,
        },
        // 6. Gate projection (4096 -> 11008, SwiGLU gate)
        KernelStep {
            name: "gate_proj".to_string(),
            grid_dim: [86, 32, 1],
            block_dim: [128, 1, 1],
            registers: 48,
            shared_mem_bytes: 32 * 1024,
            flops: 2 * 2048 * 4096 * 11008,
        },
        // 7. Up projection (4096 -> 11008)
        KernelStep {
            name: "up_proj".to_string(),
            grid_dim: [86, 32, 1],
            block_dim: [128, 1, 1],
            registers: 48,
            shared_mem_bytes: 32 * 1024,
            flops: 2 * 2048 * 4096 * 11008,
        },
        // 8. SiLU activation (element-wise on 11008-wide vector)
        KernelStep {
            name: "silu_activation".to_string(),
            grid_dim: [2048, 1, 1],
            block_dim: [256, 1, 1],
            registers: 16,
            shared_mem_bytes: 0,
            flops: 2048 * 11008 * 4, // sigmoid + mul per element
        },
        // 9. Down projection (11008 -> 4096)
        KernelStep {
            name: "down_proj".to_string(),
            grid_dim: [32, 86, 1],
            block_dim: [128, 1, 1],
            registers: 48,
            shared_mem_bytes: 32 * 1024,
            flops: 2 * 2048 * 11008 * 4096,
        },
        // 10. RMSNorm
        KernelStep {
            name: "rmsnorm".to_string(),
            grid_dim: [2048, 1, 1],
            block_dim: [256, 1, 1],
            registers: 20,
            shared_mem_bytes: 1024,
            flops: 2048 * 4096 * 5, // square + mean + rsqrt + mul + add
        },
        // 11. LM head (4096 -> 32000, final logits)
        KernelStep {
            name: "lm_head".to_string(),
            grid_dim: [250, 32, 1],
            block_dim: [128, 1, 1],
            registers: 48,
            shared_mem_bytes: 32 * 1024,
            flops: 2 * 2048 * 4096 * 32000,
        },
        // 12. Softmax + sampling
        KernelStep {
            name: "softmax_sample".to_string(),
            grid_dim: [2048, 1, 1],
            block_dim: [256, 1, 1],
            registers: 24,
            shared_mem_bytes: 2048,
            flops: 2048 * 32000 * 3, // exp + sum + div per token
        },
    ]
}

/// - Non-tiled compute kernels: flops/50 as conservative estimate
pub fn estimate_bytes_transferred(kernel: &KernelStep) -> u64 {
    let threads = kernel.total_threads();
    let flops = kernel.flops;
    let has_tiling = kernel.shared_mem_bytes >= 4096;
    let is_compute_heavy = flops > threads * 100;

    if !is_compute_heavy {
        // Memory-heavy kernel (element-wise, norm, softmax, etc.)
        // Each thread reads and writes ~4 bytes (FP16 read + FP16 write)
        threads * 4
    } else if has_tiling {
        // Tiled GEMM: shared memory enables O(sqrt(N)) global memory reads.
        // For a tiled MxKxN GEMM, bytes ~ (M*K + K*N) * 2 (FP16).
        // With 2*M*K*N flops, AI ~ N (or M). Approximate as sqrt(flops)*4.
        let sqrt_flops = (flops as f64).sqrt() as u64;
        sqrt_flops.max(1) * 4
    } else {
        // Compute-heavy but no tiling: conservative estimate
        flops / 50
    }
}

/// Compute roofline metrics for a kernel given device specs.
pub fn compute_roofline(kernel: &KernelStep, device: &DeviceSpec) -> RooflineMetrics {
    let bytes = estimate_bytes_transferred(kernel);
    let arithmetic_intensity = if bytes == 0 {
        0.0
    } else {
        kernel.flops as f64 / bytes as f64
    };

    let ridge = device.ridge_point();
    let memory_bound = arithmetic_intensity < ridge;
    let compute_bound = arithmetic_intensity >= ridge;

    // Occupancy estimation: based on register pressure and block size
    let threads_per_block = f64::from(kernel.threads_per_block());
    let max_tpb = f64::from(device.max_threads_per_sm);

    // Warps per block
    let warps_per_block = (threads_per_block / 32.0).ceil();
    // Max warps per SM
    let max_warps_per_sm = max_tpb / 32.0;

    // Register-limited warps: each SM has 65536 registers
    let regs_per_warp = f64::from(kernel.registers) * 32.0;
    let reg_limited_warps = if regs_per_warp > 0.0 {
        (65536.0 / regs_per_warp).floor()
    } else {
        max_warps_per_sm
    };

    // Shared-memory-limited blocks per SM
    let shmem_per_sm = f64::from(device.max_shared_mem_per_sm);
    let shmem_limited_blocks = if kernel.shared_mem_bytes == 0 {
        // No shared memory limit
        (max_warps_per_sm / warps_per_block).floor()
    } else {
        (shmem_per_sm / f64::from(kernel.shared_mem_bytes)).floor()
    };

    let blocks_per_sm = shmem_limited_blocks.min((reg_limited_warps / warps_per_block).floor());
    let active_warps = (blocks_per_sm * warps_per_block).min(max_warps_per_sm);
    let occupancy_pct = (active_warps / max_warps_per_sm * 100.0).clamp(0.0, 100.0);

    RooflineMetrics {
        arithmetic_intensity,
        memory_bound,
        compute_bound,
        occupancy_pct,
    }
}

/// Detect performance warnings for a kernel.
pub fn detect_warnings(kernel: &KernelStep, device: &DeviceSpec) -> Vec<PtxWarning> {
    let mut warnings = Vec::new();

    // Low occupancy: registers > device threshold
    if kernel.registers > device.max_registers_full_occupancy {
        warnings.push(PtxWarning {
            kernel_name: kernel.name.clone(),
            category: WarningCategory::LowOccupancy,
            message: format!(
                "registers/thread={} exceeds {} limit; reduce register pressure or use launch_bounds",
                kernel.registers, device.max_registers_full_occupancy,
            ),
        });
    }

    // Excessive shared memory
    if kernel.shared_mem_bytes > device.max_shared_mem_per_sm {
        warnings.push(PtxWarning {
            kernel_name: kernel.name.clone(),
            category: WarningCategory::ExcessiveSharedMemory,
            message: format!(
                "shared_mem={}KB exceeds {}KB default limit; requires cudaFuncSetAttribute opt-in",
                kernel.shared_mem_bytes / 1024,
                device.max_shared_mem_per_sm / 1024,
            ),
        });
    }

    // Uncoalesced access heuristic: non-power-of-2 block width with shared_mem=0
    // signals potential scatter/gather pattern
    let bx = kernel.block_dim[0];
    if kernel.shared_mem_bytes == 0 && bx > 0 && bx.is_power_of_two() {
        // Power of 2 block -- typically coalesced, skip
    } else if kernel.shared_mem_bytes == 0 && bx > 0 {
        warnings.push(PtxWarning {
            kernel_name: kernel.name.clone(),
            category: WarningCategory::UncoalescedAccess,
            message: format!(
                "block_dim.x={} is not power-of-2 with no shared mem staging; may cause uncoalesced global reads",
                bx,
            ),
        });
    }

    warnings
}

/// Section 1: Print the PTX kernel map table.
pub fn section_kernel_map(kernels: &[KernelStep]) {
    println!("--- Section 1: PTX Kernel Execution Map (7B Model, 1 Layer) ---");
    println!();
    println!(
        "  {:<3} {:<20} {:>14} {:>12} {:>5} {:>8} {:>14}",
        "#", "Kernel", "Grid", "Block", "Regs", "Shmem", "Est. FLOPs",
    );
    println!("  {:-<80}", "");

    for (i, k) in kernels.iter().enumerate() {
        let grid_str = format!("({},{},{})", k.grid_dim[0], k.grid_dim[1], k.grid_dim[2],);
        let block_str = format!("({},{},{})", k.block_dim[0], k.block_dim[1], k.block_dim[2],);
        let shmem_str = if k.shared_mem_bytes >= 1024 {
            format!("{}KB", k.shared_mem_bytes / 1024)
        } else {
            format!("{}B", k.shared_mem_bytes)
        };
        let flops_str = format_flops(k.flops);

        println!(
            "  {:<3} {:<20} {:>14} {:>12} {:>5} {:>8} {:>14}",
            i + 1,
            k.name,
            grid_str,
            block_str,
            k.registers,
            shmem_str,
            flops_str,
        );
    }

    let total_flops: u64 = kernels.iter().map(|k| k.flops).sum();
    println!("  {:-<80}", "");
    println!("  Total estimated FLOPs: {}", format_flops(total_flops));
    println!();
}

/// Section 2: Roofline analysis per kernel.
pub fn section_roofline(kernels: &[KernelStep], device: &DeviceSpec) {
    println!("--- Section 2: Roofline Analysis ({}) ---", device.name);
    println!();
    println!(
        "  Ridge point: {:.2} FLOP/B (peak {:.1} TFLOP/s, BW {:.3} TB/s)",
        device.ridge_point(),
        device.peak_tflops_fp16,
        device.peak_bandwidth_tb_s,
    );
    println!();
    println!(
        "  {:<20} {:>10} {:>10} {:>14}",
        "Kernel", "AI (F/B)", "Occupancy", "Bound",
    );
    println!("  {:-<58}", "");

    for k in kernels {
        let metrics = compute_roofline(k, device);
        let bound_str = if metrics.memory_bound && metrics.compute_bound {
            "balanced"
        } else if metrics.memory_bound {
            "mem-bound"
        } else {
            "compute-bound"
        };
        println!(
            "  {:<20} {:>10.2} {:>9.1}% {:>14}",
            k.name, metrics.arithmetic_intensity, metrics.occupancy_pct, bound_str,
        );
    }
    println!();
}

/// Section 3: Bandwidth and compute utilization summary.
pub fn section_utilization(kernels: &[KernelStep], device: &DeviceSpec) {
    println!("--- Section 3: Utilization Summary ---");
    println!();

    let total_flops: u64 = kernels.iter().map(|k| k.flops).sum();
    let total_bytes: u64 = kernels.iter().map(estimate_bytes_transferred).sum();

    let overall_ai = if total_bytes == 0 {
        0.0
    } else {
        total_flops as f64 / total_bytes as f64
    };

    // Estimate time at peak throughput
    let compute_time_s = total_flops as f64 / (device.peak_tflops_fp16 * 1e12);
    let memory_time_s = total_bytes as f64 / (device.peak_bandwidth_tb_s * 1e12);
    let estimated_time_s = compute_time_s.max(memory_time_s);

    let compute_util = if estimated_time_s > 0.0 {
        (compute_time_s / estimated_time_s * 100.0).min(100.0)
    } else {
        0.0
    };
    let memory_util = if estimated_time_s > 0.0 {
        (memory_time_s / estimated_time_s * 100.0).min(100.0)
    } else {
        0.0
    };

    println!("  Total FLOPs:          {}", format_flops(total_flops));
    println!("  Total bytes xferred:  {}", format_bytes(total_bytes));
    println!("  Overall AI:           {:.2} FLOP/B", overall_ai);
    println!(
        "  Est. latency (peak):  {:.4} ms",
        estimated_time_s * 1000.0,
    );
    println!("  Compute utilization:  {:.1}%", compute_util);
    println!("  Memory utilization:   {:.1}%", memory_util);
    println!();
}

/// Section 4: Performance warnings.
pub fn section_warnings(kernels: &[KernelStep], device: &DeviceSpec) -> Vec<PtxWarning> {
    println!("--- Section 4: Performance Warnings ---");
    println!();

    let mut all_warnings = Vec::new();
    for k in kernels {
        let ws = detect_warnings(k, device);
        all_warnings.extend(ws);
    }

    if all_warnings.is_empty() {
        println!("  No warnings detected.");
    } else {
        for w in &all_warnings {
            println!("  {}", w);
        }
    }
    println!();

    all_warnings
}

/// Format a FLOP count into human-readable form (GFLOP, TFLOP, etc.).
pub fn format_flops(flops: u64) -> String {
    let f = flops as f64;
    if f >= 1e12 {
        format!("{:.2} TFLOP", f / 1e12)
    } else if f >= 1e9 {
        format!("{:.2} GFLOP", f / 1e9)
    } else if f >= 1e6 {
        format!("{:.2} MFLOP", f / 1e6)
    } else {
        format!("{} FLOP", flops)
    }
}

/// Format a byte count into human-readable form.
pub fn format_bytes(bytes: u64) -> String {
    let f = bytes as f64;
    if f >= 1e12 {
        format!("{:.2} TB", f / 1e12)
    } else if f >= 1e9 {
        format!("{:.2} GB", f / 1e9)
    } else if f >= 1e6 {
        format!("{:.2} MB", f / 1e6)
    } else if f >= 1e3 {
        format!("{:.2} KB", f / 1e3)
    } else {
        format!("{} B", bytes)
    }
}
