#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use std::fmt;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// A single kernel step in the PTX execution sequence.
#[derive(Debug, Clone)]
pub struct KernelStep {
    // Kernel name (e.g. "embed_lookup", "flash_attention").
    pub name: String,
    // CUDA grid dimensions [x, y, z].
    pub grid_dim: [u32; 3],
    // CUDA block dimensions [x, y, z].
    pub block_dim: [u32; 3],
    // Number of registers per thread.
    pub registers: u32,
    // Shared memory in bytes.
    pub shared_mem_bytes: u32,
    // Estimated floating-point operations for this kernel launch.
    pub flops: u64,
}

impl KernelStep {
    /// Total number of threads launched by this kernel.
    pub fn total_threads(&self) -> u64 {
        let grid: u64 =
            u64::from(self.grid_dim[0]) * u64::from(self.grid_dim[1]) * u64::from(self.grid_dim[2]);
        let block: u64 = u64::from(self.block_dim[0])
            * u64::from(self.block_dim[1])
            * u64::from(self.block_dim[2]);
        grid * block
    }

    /// Threads per block.
    pub fn threads_per_block(&self) -> u32 {
        self.block_dim[0] * self.block_dim[1] * self.block_dim[2]
    }
}

/// Roofline model metrics for a single kernel.
#[derive(Debug, Clone)]
pub struct RooflineMetrics {
    // Arithmetic intensity: flops per byte transferred.
    pub arithmetic_intensity: f64,
    // Whether the kernel is memory-bandwidth-bound.
    pub memory_bound: bool,
    // Whether the kernel is compute-bound.
    pub compute_bound: bool,
    // Estimated SM occupancy percentage.
    pub occupancy_pct: f64,
}

impl fmt::Display for RooflineMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let bound = if self.memory_bound && self.compute_bound {
            "balanced"
        } else if self.memory_bound {
            "memory-bound"
        } else {
            "compute-bound"
        };
        write!(
            f,
            "AI={:.2} flops/B, occupancy={:.1}%, {}",
            self.arithmetic_intensity, self.occupancy_pct, bound,
        )
    }
}

/// Performance warning categories.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarningCategory {
    LowOccupancy,
    ExcessiveSharedMemory,
    UncoalescedAccess,
}

impl fmt::Display for WarningCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LowOccupancy => write!(f, "LOW_OCCUPANCY"),
            Self::ExcessiveSharedMemory => write!(f, "EXCESSIVE_SHMEM"),
            Self::UncoalescedAccess => write!(f, "UNCOALESCED"),
        }
    }
}

/// A performance warning for a specific kernel.
#[derive(Debug, Clone)]
pub struct PtxWarning {
    // Name of the kernel that triggered the warning.
    pub kernel_name: String,
    // Warning category.
    pub category: WarningCategory,
    // Human-readable explanation.
    pub message: String,
}

impl fmt::Display for PtxWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}: {}",
            self.category, self.kernel_name, self.message,
        )
    }
}

/// Device specs for roofline analysis (modeled on A100 80GB SXM).
#[derive(Debug, Clone)]
pub struct DeviceSpec {
    // Device name.
    pub name: String,
    // Peak FP16 TFLOPS.
    pub peak_tflops_fp16: f64,
    // Peak memory bandwidth in TB/s.
    pub peak_bandwidth_tb_s: f64,
    // Maximum shared memory per SM in bytes.
    pub max_shared_mem_per_sm: u32,
    // Maximum registers per thread before occupancy drops.
    pub max_registers_full_occupancy: u32,
    // Maximum threads per SM.
    pub max_threads_per_sm: u32,
}

impl DeviceSpec {
    /// A100 80GB SXM spec (deterministic reference).
    pub fn a100_80gb() -> Self {
        Self {
            name: "NVIDIA A100 80GB SXM".to_string(),
            peak_tflops_fp16: 312.0,
            peak_bandwidth_tb_s: 2.039,
            max_shared_mem_per_sm: 48 * 1024, // 48 KiB default config
            max_registers_full_occupancy: 64, // >64 regs/thread reduces occupancy
            max_threads_per_sm: 2048,
        }
    }

    /// Ridge point: arithmetic intensity where compute meets memory roof.
    pub fn ridge_point(&self) -> f64 {
        // peak_compute (TFLOP/s) / peak_bandwidth (TB/s) = flops/byte at ridge
        self.peak_tflops_fp16 / self.peak_bandwidth_tb_s
    }
}

// ---------------------------------------------------------------------------
// Kernel sequence builder
// ---------------------------------------------------------------------------

// Build the 12-step kernel execution sequence for a 7B LLaMA-style model.
//
// Each kernel's specs are deterministic, modeling a single transformer
// layer forward pass with typical 7B dimensions:

// ---------------------------------------------------------------------------
// Roofline analysis
// ---------------------------------------------------------------------------

// Estimate bytes transferred per kernel from global memory.
//
// Uses a tiered heuristic:
// - Element-wise kernels (no/low shared mem, low flops/thread): 4 bytes/thread
// - Tiled GEMM kernels (shared mem staging): sqrt(flops)*4 models the
//   O(N^2) compute vs O(N) memory access pattern of tiled matrix multiply

// ---------------------------------------------------------------------------
// Warning detection
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Section helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
