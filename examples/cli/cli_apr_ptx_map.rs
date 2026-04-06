//! # Recipe: APR PTX Source Mapping CLI
//!
//! **Category**: CLI Tools
//! **CLI Equivalent**: `apr ptx-map`
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
//! Demonstrate the `apr ptx-map` workflow: model-to-PTX source mapping.
//! Makes GPU kernel dispatch visible (Mieruka / visual management principle).
//! Maps each model layer to its dispatched PTX kernel and reports resource
//! usage, theoretical occupancy, and instruction category breakdown.
//!
//! ## Run Command
//! ```bash
//! cargo run --example cli_apr_ptx_map
//! cargo run --example cli_apr_ptx_map -- --demo
//! cargo run --example cli_apr_ptx_map -- --demo --kernel-filter attention
//! ```
//!
//!
//! ## Format Variants
//! ```bash
//! apr inspect model.apr          # APR native format
//! apr inspect model.gguf         # GGUF (llama.cpp compatible)
//! apr inspect model.safetensors  # SafeTensors (HuggingFace)
//! ```
//! ## References
//! - Amershi, S. et al. (2019). *Software Engineering for Machine Learning: A Case Study*. ICSE. DOI: 10.1109/ICSE-SEIP.2019.00042

use apr_cookbook::prelude::*;
use clap::Parser;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

fn main() -> Result<()> {
    let config = PtxMapConfig::parse();
    run_ptx_map(&config)
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Parser)]
#[command(
    name = "apr-ptx-map",
    about = "Map model layers to PTX kernel dispatches (Mieruka)"
)]
struct PtxMapConfig {
    /// Path to .apr model file
    model_path: Option<String>,

    /// Only show kernels matching this filter string
    #[arg(short = 'k', long = "kernel-filter")]
    kernel_filter: Option<String>,

    /// Run with synthetic demo model
    #[arg(long, short = 'd')]
    demo: bool,
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Maps a model layer to the GPU kernel it dispatches.
#[derive(Debug, Clone)]
struct KernelMapping {
    layer_name: String,
    kernel_name: String,
    grid_dim: [u32; 3],
    block_dim: [u32; 3],
    shared_mem_bytes: u32,
    registers_per_thread: u32,
}

impl KernelMapping {
    /// Threads per block.
    fn threads_per_block(&self) -> u32 {
        self.block_dim[0] * self.block_dim[1] * self.block_dim[2]
    }

    /// Total grid blocks.
    fn total_blocks(&self) -> u64 {
        u64::from(self.grid_dim[0]) * u64::from(self.grid_dim[1]) * u64::from(self.grid_dim[2])
    }
}

impl PtxSourceRegion {
    fn line_span(&self) -> u32 {
        self.end_line.saturating_sub(self.start_line) + 1
    }
}

/// A region within simulated PTX source for a kernel.
#[derive(Debug, Clone)]
struct PtxSourceRegion {
    kernel_name: String,
    start_line: u32,
    end_line: u32,
    instruction_count: u32,
    category: InstructionCategory,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum InstructionCategory {
    Compute,
    Memory,
    Control,
}

impl fmt::Display for InstructionCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compute => write!(f, "compute"),
            Self::Memory => write!(f, "memory"),
            Self::Control => write!(f, "control"),
        }
    }
}

// ---------------------------------------------------------------------------
// Argument parsing (test helper)
// ---------------------------------------------------------------------------

#[cfg(test)]
fn parse_args(args: &[String]) -> std::result::Result<PtxMapConfig, clap::Error> {
    PtxMapConfig::try_parse_from(args)
}

// ---------------------------------------------------------------------------
// Deterministic helpers
// ---------------------------------------------------------------------------

fn deterministic_seed(name: &str) -> u64 {
    let mut h = DefaultHasher::new();
    name.hash(&mut h);
    h.finish()
}

fn hash_to_u32(seed: u64, variant: u64) -> u32 {
    let mut h = DefaultHasher::new();
    seed.hash(&mut h);
    variant.hash(&mut h);
    (h.finish() & 0xFFFF_FFFF) as u32
}

// ---------------------------------------------------------------------------
// Occupancy model
// ---------------------------------------------------------------------------

/// SM specs modeled on A100 (sm_80).
const MAX_WARPS_PER_SM: f64 = 64.0; // 2048/32
const REGISTERS_PER_SM: u32 = 65536;
const MAX_SHARED_MEM_PER_SM: u32 = 48 * 1024; // 48 KiB default

/// Compute theoretical occupancy percentage from kernel resource usage.
fn compute_occupancy(mapping: &KernelMapping) -> f64 {
    let tpb = f64::from(mapping.threads_per_block());
    let warps_per_block = (tpb / 32.0).ceil();

    // Register-limited warps
    let regs_per_warp = f64::from(mapping.registers_per_thread) * 32.0;
    let reg_limited_warps = if regs_per_warp > 0.0 {
        (f64::from(REGISTERS_PER_SM) / regs_per_warp).floor()
    } else {
        MAX_WARPS_PER_SM
    };

    // Shared-memory-limited blocks
    let shmem_limited_blocks = if mapping.shared_mem_bytes == 0 {
        (MAX_WARPS_PER_SM / warps_per_block).floor()
    } else {
        (f64::from(MAX_SHARED_MEM_PER_SM) / f64::from(mapping.shared_mem_bytes)).floor()
    };

    let blocks_per_sm = shmem_limited_blocks.min((reg_limited_warps / warps_per_block).floor());
    let active_warps = (blocks_per_sm * warps_per_block).min(MAX_WARPS_PER_SM);
    (active_warps / MAX_WARPS_PER_SM * 100.0).clamp(0.0, 100.0)
}

// ---------------------------------------------------------------------------
// Demo model construction
// ---------------------------------------------------------------------------

/// Layer descriptor for synthetic model generation.
struct LayerSpec {
    name: &'static str,
    kernel: &'static str,
    grid: [u32; 3],
    block: [u32; 3],
    shared_kb: u32,
    regs: u32,
}

const DEMO_LAYERS: &[LayerSpec] = &[
    LayerSpec {
        name: "embed_tokens",
        kernel: "k_embed_lookup",
        grid: [2048, 1, 1],
        block: [256, 1, 1],
        shared_kb: 0,
        regs: 24,
    },
    LayerSpec {
        name: "layer_0.self_attn",
        kernel: "k_flash_attention_fwd",
        grid: [2048, 32, 1],
        block: [128, 1, 1],
        shared_kb: 48,
        regs: 72,
    },
    LayerSpec {
        name: "layer_0.attn_proj",
        kernel: "k_gemm_nt_fp16",
        grid: [64, 32, 1],
        block: [128, 1, 1],
        shared_kb: 32,
        regs: 48,
    },
    LayerSpec {
        name: "layer_0.mlp.gate",
        kernel: "k_gemm_nt_fp16",
        grid: [86, 32, 1],
        block: [128, 1, 1],
        shared_kb: 32,
        regs: 48,
    },
    LayerSpec {
        name: "layer_0.mlp.up",
        kernel: "k_gemm_nt_fp16",
        grid: [86, 32, 1],
        block: [128, 1, 1],
        shared_kb: 32,
        regs: 48,
    },
    LayerSpec {
        name: "layer_0.mlp.silu",
        kernel: "k_silu_elementwise",
        grid: [2048, 1, 1],
        block: [256, 1, 1],
        shared_kb: 0,
        regs: 16,
    },
    LayerSpec {
        name: "layer_0.mlp.down",
        kernel: "k_gemm_nt_fp16",
        grid: [32, 86, 1],
        block: [128, 1, 1],
        shared_kb: 32,
        regs: 48,
    },
    LayerSpec {
        name: "layer_0.rmsnorm",
        kernel: "k_rmsnorm",
        grid: [2048, 1, 1],
        block: [256, 1, 1],
        shared_kb: 1,
        regs: 20,
    },
    LayerSpec {
        name: "lm_head",
        kernel: "k_gemm_nt_fp16",
        grid: [250, 32, 1],
        block: [128, 1, 1],
        shared_kb: 32,
        regs: 48,
    },
    LayerSpec {
        name: "softmax_sample",
        kernel: "k_softmax_topk",
        grid: [2048, 1, 1],
        block: [256, 1, 1],
        shared_kb: 2,
        regs: 24,
    },
];

fn create_demo_mappings() -> Vec<KernelMapping> {
    DEMO_LAYERS
        .iter()
        .map(|spec| KernelMapping {
            layer_name: spec.name.to_string(),
            kernel_name: spec.kernel.to_string(),
            grid_dim: spec.grid,
            block_dim: spec.block,
            shared_mem_bytes: spec.shared_kb * 1024,
            registers_per_thread: spec.regs,
        })
        .collect()
}

/// Generate simulated PTX source regions for each kernel mapping.
fn generate_ptx_regions(mappings: &[KernelMapping]) -> Vec<PtxSourceRegion> {
    let mut regions = Vec::new();
    let mut line_cursor: u32 = 1;

    for mapping in mappings {
        let seed = deterministic_seed(&mapping.kernel_name);

        // Compute region: proportional to register count (more regs = more ALU ops)
        let compute_count = mapping.registers_per_thread * 2 + 10;
        let compute_start = line_cursor;
        let compute_end = compute_start + compute_count - 1;
        regions.push(PtxSourceRegion {
            kernel_name: mapping.kernel_name.clone(),
            start_line: compute_start,
            end_line: compute_end,
            instruction_count: compute_count,
            category: InstructionCategory::Compute,
        });
        line_cursor = compute_end + 1;

        // Memory region: proportional to shared memory usage
        let mem_count = if mapping.shared_mem_bytes > 0 {
            (mapping.shared_mem_bytes / 1024).max(4) + hash_to_u32(seed, 1) % 8
        } else {
            6 + hash_to_u32(seed, 2) % 4
        };
        let mem_start = line_cursor;
        let mem_end = mem_start + mem_count - 1;
        regions.push(PtxSourceRegion {
            kernel_name: mapping.kernel_name.clone(),
            start_line: mem_start,
            end_line: mem_end,
            instruction_count: mem_count,
            category: InstructionCategory::Memory,
        });
        line_cursor = mem_end + 1;

        // Control region: branch, barrier, return
        let ctrl_count = 4 + hash_to_u32(seed, 3) % 6;
        let ctrl_start = line_cursor;
        let ctrl_end = ctrl_start + ctrl_count - 1;
        regions.push(PtxSourceRegion {
            kernel_name: mapping.kernel_name.clone(),
            start_line: ctrl_start,
            end_line: ctrl_end,
            instruction_count: ctrl_count,
            category: InstructionCategory::Control,
        });
        line_cursor = ctrl_end + 2; // gap between kernels
    }

    regions
}

/// Create mappings from a model file path (simulated via deterministic hashing).
fn create_file_mappings(path: &str) -> Result<Vec<KernelMapping>> {
    let bytes = std::fs::read(path).map_err(|e| {
        CookbookError::invalid_format(format!("Failed to read model {}: {}", path, e))
    })?;
    let seed = deterministic_seed(path);
    let layer_count = (bytes.len() / 2048).clamp(3, 10);

    let layer_names = [
        "embed", "attn_qkv", "attn_out", "ffn_gate", "ffn_up", "ffn_down", "norm", "lm_head",
        "softmax", "residual",
    ];
    let kernel_names = [
        "k_embed_lookup",
        "k_gemm_nt_fp16",
        "k_gemm_nt_fp16",
        "k_gemm_nt_fp16",
        "k_gemm_nt_fp16",
        "k_gemm_nt_fp16",
        "k_rmsnorm",
        "k_gemm_nt_fp16",
        "k_softmax_topk",
        "k_add_residual",
    ];

    let mut mappings = Vec::new();
    for i in 0..layer_count {
        let li = i % layer_names.len();
        let grid_x = (hash_to_u32(seed, i as u64 * 10) % 256).max(1);
        let grid_y = (hash_to_u32(seed, i as u64 * 10 + 1) % 64).max(1);
        let block_x = [64, 128, 256][(hash_to_u32(seed, i as u64 * 10 + 2) % 3) as usize];
        let shared = (hash_to_u32(seed, i as u64 * 10 + 3) % 49) * 1024;
        let regs = (hash_to_u32(seed, i as u64 * 10 + 4) % 80).max(16);

        mappings.push(KernelMapping {
            layer_name: format!("layer_{}.{}", i, layer_names[li]),
            kernel_name: kernel_names[li].to_string(),
            grid_dim: [grid_x, grid_y, 1],
            block_dim: [block_x, 1, 1],
            shared_mem_bytes: shared,
            registers_per_thread: regs,
        });
    }

    Ok(mappings)
}

// ---------------------------------------------------------------------------
// Main driver
// ---------------------------------------------------------------------------

fn run_ptx_map(config: &PtxMapConfig) -> Result<()> {
    let mut ctx = RecipeContext::new("cli_apr_ptx_map")?;

    let mappings = if config.demo {
        create_demo_mappings()
    } else if let Some(path) = &config.model_path {
        create_file_mappings(path)?
    } else {
        println!("No model provided. Use --demo or specify a model path.");
        return Ok(());
    };

    // Apply kernel filter
    let filtered: Vec<&KernelMapping> = match &config.kernel_filter {
        Some(filter) => mappings
            .iter()
            .filter(|m| {
                m.kernel_name.contains(filter.as_str()) || m.layer_name.contains(filter.as_str())
            })
            .collect(),
        None => mappings.iter().collect(),
    };

    println!("APR PTX Map (Mieruka - Visual Kernel Dispatch)");
    println!("===============================================");
    println!();
    if let Some(f) = &config.kernel_filter {
        println!(
            "Filter: \"{}\" ({} of {} mappings)",
            f,
            filtered.len(),
            mappings.len()
        );
        println!();
    }

    // Kernel dispatch table
    println!(
        "  {:<24} {:<24} {:>14} {:>12} {:>10} {:>5} {:>10}",
        "LAYER", "KERNEL", "GRID", "BLOCK", "SHARED_MEM", "REGS", "OCCUPANCY%"
    );
    println!("  {:-<101}", "");

    let mut total_blocks_sum: u64 = 0;
    for m in &filtered {
        let grid_str = format!("({},{},{})", m.grid_dim[0], m.grid_dim[1], m.grid_dim[2]);
        let block_str = format!("({},{},{})", m.block_dim[0], m.block_dim[1], m.block_dim[2]);
        let shmem_str = if m.shared_mem_bytes >= 1024 {
            format!("{}KB", m.shared_mem_bytes / 1024)
        } else {
            format!("{}B", m.shared_mem_bytes)
        };
        let occupancy = compute_occupancy(m);
        total_blocks_sum += m.total_blocks();
        println!(
            "  {:<24} {:<24} {:>14} {:>12} {:>10} {:>5} {:>9.1}%",
            m.layer_name,
            m.kernel_name,
            grid_str,
            block_str,
            shmem_str,
            m.registers_per_thread,
            occupancy,
        );
    }
    println!("  {:-<101}", "");
    println!("  Total grid blocks dispatched: {}", total_blocks_sum);
    println!();

    // PTX source region breakdown
    let all_regions = generate_ptx_regions(&mappings);
    let filtered_regions: Vec<&PtxSourceRegion> = match &config.kernel_filter {
        Some(filter) => all_regions
            .iter()
            .filter(|r| r.kernel_name.contains(filter.as_str()))
            .collect(),
        None => all_regions.iter().collect(),
    };

    println!("PTX Source Region Breakdown:");
    println!(
        "  {:<24} {:<10} {:>8} {:>8} {:>6} {:>12}",
        "KERNEL", "CATEGORY", "START", "END", "SPAN", "INSTR_COUNT"
    );
    println!("  {:-<72}", "");

    for r in &filtered_regions {
        println!(
            "  {:<24} {:<10} {:>8} {:>8} {:>6} {:>12}",
            r.kernel_name,
            r.category.to_string(),
            r.start_line,
            r.end_line,
            r.line_span(),
            r.instruction_count,
        );
    }
    println!();

    // Category summary
    let total_compute: u32 = filtered_regions
        .iter()
        .filter(|r| r.category == InstructionCategory::Compute)
        .map(|r| r.instruction_count)
        .sum();
    let total_memory: u32 = filtered_regions
        .iter()
        .filter(|r| r.category == InstructionCategory::Memory)
        .map(|r| r.instruction_count)
        .sum();
    let total_control: u32 = filtered_regions
        .iter()
        .filter(|r| r.category == InstructionCategory::Control)
        .map(|r| r.instruction_count)
        .sum();
    let total_instr = total_compute + total_memory + total_control;

    println!("Instruction Category Summary:");
    let pct = |n: u32| -> f64 {
        if total_instr == 0 {
            0.0
        } else {
            f64::from(n) / f64::from(total_instr) * 100.0
        }
    };
    println!(
        "  compute: {:>5} ({:>5.1}%)",
        total_compute,
        pct(total_compute)
    );
    println!(
        "  memory:  {:>5} ({:>5.1}%)",
        total_memory,
        pct(total_memory)
    );
    println!(
        "  control: {:>5} ({:>5.1}%)",
        total_control,
        pct(total_control)
    );
    println!("  total:   {:>5}", total_instr);
    println!();

    // Occupancy summary
    let occupancies: Vec<f64> = filtered.iter().map(|m| compute_occupancy(m)).collect();
    let avg_occupancy = if occupancies.is_empty() {
        0.0
    } else {
        occupancies.iter().sum::<f64>() / occupancies.len() as f64
    };
    let min_occupancy = occupancies.iter().copied().fold(f64::INFINITY, f64::min);
    let max_occupancy = occupancies
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    println!("Occupancy Summary:");
    println!("  Average: {:.1}%", avg_occupancy);
    if !occupancies.is_empty() {
        println!("  Min:     {:.1}%", min_occupancy);
        println!("  Max:     {:.1}%", max_occupancy);
    }

    ctx.record_metric("layer_count", mappings.len() as i64);
    ctx.record_metric("filtered_count", filtered.len() as i64);
    ctx.record_metric("total_instructions", i64::from(total_instr));
    ctx.record_float_metric("avg_occupancy", avg_occupancy);

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_args_demo() {
        let args = vec!["apr-ptx-map".to_string(), "--demo".to_string()];
        let config = parse_args(&args).expect("parse ok");
        assert!(config.demo);
        assert!(config.kernel_filter.is_none());
    }

    #[test]
    fn test_parse_args_kernel_filter() {
        let args = vec![
            "apr-ptx-map".to_string(),
            "--kernel-filter".to_string(),
            "attention".to_string(),
        ];
        let config = parse_args(&args).expect("parse ok");
        assert_eq!(config.kernel_filter, Some("attention".to_string()));
    }

    #[test]
    fn test_parse_args_model_path() {
        let args = vec!["apr-ptx-map".to_string(), "model.apr".to_string()];
        let config = parse_args(&args).expect("parse ok");
        assert_eq!(config.model_path, Some("model.apr".to_string()));
    }

    #[test]
    fn test_parse_args_unknown_rejected() {
        let args = vec!["apr-ptx-map".to_string(), "--bogus".to_string()];
        assert!(parse_args(&args).is_err());
    }

    #[test]
    fn test_demo_mappings_count() {
        let mappings = create_demo_mappings();
        assert_eq!(mappings.len(), DEMO_LAYERS.len());
    }

    #[test]
    fn test_threads_per_block() {
        let m = KernelMapping {
            layer_name: "test".to_string(),
            kernel_name: "k_test".to_string(),
            grid_dim: [4, 2, 1],
            block_dim: [128, 2, 1],
            shared_mem_bytes: 0,
            registers_per_thread: 32,
        };
        assert_eq!(m.threads_per_block(), 256);
        assert_eq!(m.total_blocks(), 8);
    }

    #[test]
    fn test_occupancy_full() {
        // Low register count, no shared memory -> should yield high occupancy
        let m = KernelMapping {
            layer_name: "test".to_string(),
            kernel_name: "k_test".to_string(),
            grid_dim: [1, 1, 1],
            block_dim: [256, 1, 1],
            shared_mem_bytes: 0,
            registers_per_thread: 16,
        };
        let occ = compute_occupancy(&m);
        assert!(
            occ > 90.0,
            "low-resource kernel should have high occupancy, got {:.1}%",
            occ
        );
    }

    #[test]
    fn test_occupancy_register_limited() {
        // Very high register count -> reduced occupancy
        let m = KernelMapping {
            layer_name: "test".to_string(),
            kernel_name: "k_test".to_string(),
            grid_dim: [1, 1, 1],
            block_dim: [128, 1, 1],
            shared_mem_bytes: 0,
            registers_per_thread: 128,
        };
        let occ = compute_occupancy(&m);
        assert!(
            occ < 50.0,
            "high-register kernel should have reduced occupancy, got {:.1}%",
            occ
        );
    }

    #[test]
    fn test_occupancy_shmem_limited() {
        // Near-limit shared memory -> reduced occupancy
        let m = KernelMapping {
            layer_name: "test".to_string(),
            kernel_name: "k_test".to_string(),
            grid_dim: [1, 1, 1],
            block_dim: [128, 1, 1],
            shared_mem_bytes: 48 * 1024, // fills entire SM shmem budget
            registers_per_thread: 32,
        };
        let occ = compute_occupancy(&m);
        // Only 1 block fits per SM
        assert!(
            occ < 20.0,
            "shmem-limited kernel should have low occupancy, got {:.1}%",
            occ
        );
    }

    #[test]
    fn test_ptx_regions_cover_all_kernels() {
        let mappings = create_demo_mappings();
        let regions = generate_ptx_regions(&mappings);
        // Each mapping produces 3 regions (compute, memory, control)
        assert_eq!(regions.len(), mappings.len() * 3);
    }

    #[test]
    fn test_ptx_regions_categories() {
        let mappings = create_demo_mappings();
        let regions = generate_ptx_regions(&mappings);
        for chunk in regions.chunks(3) {
            assert_eq!(chunk[0].category, InstructionCategory::Compute);
            assert_eq!(chunk[1].category, InstructionCategory::Memory);
            assert_eq!(chunk[2].category, InstructionCategory::Control);
        }
    }

    #[test]
    fn test_ptx_regions_non_overlapping() {
        let mappings = create_demo_mappings();
        let regions = generate_ptx_regions(&mappings);
        for window in regions.windows(2) {
            assert!(
                window[1].start_line > window[0].end_line,
                "regions must not overlap: [{}-{}] vs [{}-{}]",
                window[0].start_line,
                window[0].end_line,
                window[1].start_line,
                window[1].end_line,
            );
        }
    }

    #[test]
    fn test_ptx_region_line_span() {
        let r = PtxSourceRegion {
            kernel_name: "test".to_string(),
            start_line: 10,
            end_line: 19,
            instruction_count: 10,
            category: InstructionCategory::Compute,
        };
        assert_eq!(r.line_span(), 10);
    }

    #[test]
    fn test_instruction_category_display() {
        assert_eq!(InstructionCategory::Compute.to_string(), "compute");
        assert_eq!(InstructionCategory::Memory.to_string(), "memory");
        assert_eq!(InstructionCategory::Control.to_string(), "control");
    }

    #[test]
    fn test_demo_run() {
        let config = PtxMapConfig {
            model_path: None,
            kernel_filter: None,
            demo: true,
        };
        assert!(run_ptx_map(&config).is_ok());
    }

    #[test]
    fn test_demo_run_with_filter() {
        let config = PtxMapConfig {
            model_path: None,
            kernel_filter: Some("attention".to_string()),
            demo: true,
        };
        assert!(run_ptx_map(&config).is_ok());
    }

    #[test]
    fn test_deterministic_seed_consistent() {
        let s1 = deterministic_seed("test_kernel");
        let s2 = deterministic_seed("test_kernel");
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_deterministic_seed_varies() {
        let s1 = deterministic_seed("kernel_a");
        let s2 = deterministic_seed("kernel_b");
        assert_ne!(s1, s2);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_occupancy_bounded(regs in 16u32..256, shmem_kb in 0u32..64, block_x in 1u32..9) {
            let block_size = block_x * 32; // warp-aligned
            let m = KernelMapping {
                layer_name: "prop".to_string(),
                kernel_name: "k_prop".to_string(),
                grid_dim: [1, 1, 1],
                block_dim: [block_size, 1, 1],
                shared_mem_bytes: shmem_kb * 1024,
                registers_per_thread: regs,
            };
            let occ = compute_occupancy(&m);
            prop_assert!(occ >= 0.0 && occ <= 100.0,
                "occupancy must be in [0,100], got {}", occ);
        }

        #[test]
        fn prop_ptx_regions_monotonic(n in 1usize..8) {
            let mappings: Vec<KernelMapping> = (0..n).map(|i| KernelMapping {
                layer_name: format!("layer_{}", i),
                kernel_name: format!("k_{}", i),
                grid_dim: [1, 1, 1],
                block_dim: [128, 1, 1],
                shared_mem_bytes: 4096,
                registers_per_thread: 32,
            }).collect();
            let regions = generate_ptx_regions(&mappings);
            for window in regions.windows(2) {
                prop_assert!(window[1].start_line > window[0].end_line,
                    "regions must be strictly ordered");
            }
        }
    }
}
