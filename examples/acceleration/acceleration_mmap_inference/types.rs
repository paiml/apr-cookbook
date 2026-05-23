//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use rand::Rng;
use std::time::Instant;

// ============================================================================
// Constants
// ============================================================================

/// Number of tensors in the synthetic model.
pub const NUM_TENSORS: usize = 4;

/// Size of each tensor in f64 elements (~64 KB per tensor, ~256 KB total).
pub const ELEMENTS_PER_TENSOR: usize = 8192;

/// Simulated OS page size in bytes.
pub const PAGE_SIZE: usize = 4096;

/// Layers accessed during a partial forward pass (0..3, skipping tensor 3).
pub const ACTIVE_LAYERS: usize = 3;

// ============================================================================
// Data Structures
// ============================================================================

/// Strategy used to load model weights from disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadStrategy {
    // Read entire file into a contiguous `Vec<u8>` before parsing.
    Eager,
    // Open file as a memory-mapped region; pages load on first access.
    MemoryMapped,
}

impl std::fmt::Display for LoadStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Eager => write!(f, "Eager"),
            Self::MemoryMapped => write!(f, "MemoryMapped"),
        }
    }
}

/// Metrics collected during model loading.
#[derive(Debug, Clone)]
pub struct LoadMetrics {
    // Which strategy was used.
    pub strategy: LoadStrategy,
    // Wall-clock time to open/load the model file (milliseconds).
    pub load_time_ms: f64,
    // Peak memory attributed to model data (bytes).
    pub peak_memory_bytes: usize,
    // Number of pages currently resident in memory.
    pub pages_resident: usize,
    // Total number of pages in the file.
    pub pages_total: usize,
}

/// Tracks whether an individual page was accessed during inference.
#[derive(Debug, Clone)]
pub struct PageAccess {
    // Zero-indexed page identifier.
    pub page_id: usize,
    // Byte offset of this page within the file.
    pub offset: usize,
    // Size of this page in bytes (last page may be smaller).
    pub size: usize,
    // Whether inference touched this page.
    pub accessed: bool,
    // Name of the tensor that owns this page (if any).
    pub tensor_name: String,
}

/// Result of running inference under a given loading strategy.
#[derive(Debug, Clone)]
pub struct InferenceResult {
    // Strategy that produced this result.
    pub strategy: LoadStrategy,
    // Output vector from the forward pass.
    pub output: Vec<f64>,
    // Wall-clock inference latency (milliseconds).
    pub latency_ms: f64,
}

/// A named tensor with its raw weight data.
#[derive(Debug, Clone)]
pub struct TensorRecord {
    pub name: String,
    pub data: Vec<f64>,
}

// Simulated memory-mapped file view.
//
// Wraps a byte buffer and tracks per-page access to simulate demand paging
/// without requiring OS-level mmap (keeps the example portable and testable).
pub struct MmapView {
    // Underlying byte storage (simulates the file mapping).
    pub buffer: Vec<u8>,
    // Per-page access tracking.
    pub page_accessed: Vec<bool>,
    // Total number of pages.
    pub page_count: usize,
}

impl MmapView {
    /// Create a new mmap view over raw bytes.
    pub fn new(data: Vec<u8>) -> Self {
        let page_count = data.len().div_ceil(PAGE_SIZE);
        Self {
            buffer: data,
            page_accessed: vec![false; page_count],
            page_count,
        }
    }

    /// Read a slice from the mapped region, recording page faults.
    pub fn read_range(&mut self, offset: usize, len: usize) -> &[u8] {
        let end = (offset + len).min(self.buffer.len());
        let page_start = offset / PAGE_SIZE;
        let page_end = end.div_ceil(PAGE_SIZE);
        for p in page_start..page_end.min(self.page_count) {
            self.page_accessed[p] = true;
        }
        &self.buffer[offset..end]
    }

    /// Count how many pages have been accessed so far.
    pub fn resident_pages(&self) -> usize {
        self.page_accessed.iter().filter(|&&a| a).count()
    }

    /// Build the full page access report with tensor name annotations.
    pub fn page_report(&self, tensors: &[TensorRecord]) -> Vec<PageAccess> {
        let bytes_per_tensor = ELEMENTS_PER_TENSOR * 8; // f64 = 8 bytes
        (0..self.page_count)
            .map(|p| {
                let offset = p * PAGE_SIZE;
                let remaining = self.buffer.len().saturating_sub(offset);
                let size = remaining.min(PAGE_SIZE);
                let tensor_idx = offset / bytes_per_tensor;
                let tensor_name = tensors
                    .get(tensor_idx)
                    .map_or_else(|| "padding".to_string(), |t| t.name.clone());
                PageAccess {
                    page_id: p,
                    offset,
                    size,
                    accessed: self.page_accessed[p],
                    tensor_name,
                }
            })
            .collect()
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================
