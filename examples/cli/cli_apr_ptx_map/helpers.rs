#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use clap::Parser;
use proptest::prelude::*;
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Parser)]
#[command(
    name = "apr-ptx-map",
    about = "Map model layers to PTX kernel dispatches (Mieruka)"
)]
pub struct PtxMapConfig {
    // Path to .apr model file
    pub model_path: Option<String>,

    /// Only show kernels matching this filter string
    #[arg(short = 'k', long = "kernel-filter")]
    pub kernel_filter: Option<String>,

    /// Run with synthetic demo model
    #[arg(long, short = 'd')]
    pub demo: bool,
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Maps a model layer to the GPU kernel it dispatches.
#[derive(Debug, Clone)]
pub struct KernelMapping {
    pub layer_name: String,
    pub kernel_name: String,
    pub grid_dim: [u32; 3],
    pub block_dim: [u32; 3],
    pub shared_mem_bytes: u32,
    pub registers_per_thread: u32,
}

impl KernelMapping {
    /// Threads per block.
    pub fn threads_per_block(&self) -> u32 {
        self.block_dim[0] * self.block_dim[1] * self.block_dim[2]
    }

    /// Total grid blocks.
    pub fn total_blocks(&self) -> u64 {
        u64::from(self.grid_dim[0]) * u64::from(self.grid_dim[1]) * u64::from(self.grid_dim[2])
    }
}

impl PtxSourceRegion {
    pub fn line_span(&self) -> u32 {
        self.end_line.saturating_sub(self.start_line) + 1
    }
}

/// A region within simulated PTX source for a kernel.
#[derive(Debug, Clone)]
pub struct PtxSourceRegion {
    pub kernel_name: String,
    pub start_line: u32,
    pub end_line: u32,
    pub instruction_count: u32,
    pub category: InstructionCategory,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstructionCategory {
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
