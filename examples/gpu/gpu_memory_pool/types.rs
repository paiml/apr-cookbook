#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
//! GPU Memory Pool Allocation
//!
//! Contract: contracts/recipe-iiur-v1.yaml, contracts/flash-attention-v1.yaml
//! Demonstrates memory pooling with slab allocation, fragmentation analysis,
//! multi-tenant budgets, and watermark tracking for inference workloads.
//!
//! ```bash
//! cargo run --example gpu_memory_pool
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

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationStrategy {
    Slab,
    BestFit,
    FirstFit,
}

impl fmt::Display for AllocationStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Slab => write!(f, "Slab"),
            Self::BestFit => write!(f, "BestFit"),
            Self::FirstFit => write!(f, "FirstFit"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryBlock {
    pub offset: usize,
    pub size: usize,
    pub allocated: bool,
    pub tenant_id: Option<String>,
}

#[derive(Debug, Clone)]
pub struct AllocationRequest {
    pub size: usize,
    pub tenant_id: Option<String>,
    pub alignment: usize,
}

#[derive(Debug, Clone)]
pub struct FragmentationStats {
    pub total_free: usize,
    pub largest_free_block: usize,
    pub fragment_count: usize,
    pub fragmentation_ratio: f64,
}

impl fmt::Display for FragmentationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "free={} bytes, largest={} bytes, fragments={}, ratio={:.4}",
            self.total_free, self.largest_free_block, self.fragment_count, self.fragmentation_ratio
        )
    }
}

#[derive(Debug, Clone)]
pub struct WatermarkTracker {
    pub peak_bytes: usize,
    pub current_bytes: usize,
    pub sample_count: u64,
    pub cumulative_bytes: u64,
}

impl WatermarkTracker {
    pub fn new() -> Self {
        Self {
            peak_bytes: 0,
            current_bytes: 0,
            sample_count: 0,
            cumulative_bytes: 0,
        }
    }

    pub fn record(&mut self, allocated_bytes: usize) {
        self.current_bytes = allocated_bytes;
        if allocated_bytes > self.peak_bytes {
            self.peak_bytes = allocated_bytes;
        }
        self.sample_count += 1;
        self.cumulative_bytes += allocated_bytes as u64;
    }

    pub fn average_bytes(&self) -> f64 {
        if self.sample_count == 0 {
            0.0
        } else {
            self.cumulative_bytes as f64 / self.sample_count as f64
        }
    }
}

impl fmt::Display for WatermarkTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "peak={}, current={}, avg={:.1}, samples={}",
            self.peak_bytes,
            self.current_bytes,
            self.average_bytes(),
            self.sample_count
        )
    }
}

#[derive(Debug, Clone)]
pub struct TenantBudget {
    pub max_bytes: usize,
    pub used_bytes: usize,
}

#[derive(Debug)]
pub struct MemoryPool {
    pub total_bytes: usize,
    pub blocks: Vec<MemoryBlock>,
    pub strategy: AllocationStrategy,
    pub slab_size: usize,
    pub tenant_budgets: HashMap<String, TenantBudget>,
    pub watermarks: WatermarkTracker,
    pub next_alloc_id: u64,
}

impl MemoryPool {
    pub fn new(total_bytes: usize, strategy: AllocationStrategy) -> Self {
        let slab_size = if strategy == AllocationStrategy::Slab {
            4096
        } else {
            0
        };
        Self::with_slab_size(total_bytes, strategy, slab_size)
    }

    pub fn with_slab_size(
        total_bytes: usize,
        strategy: AllocationStrategy,
        slab_size: usize,
    ) -> Self {
        Self {
            total_bytes,
            blocks: vec![MemoryBlock {
                offset: 0,
                size: total_bytes,
                allocated: false,
                tenant_id: None,
            }],
            strategy,
            slab_size: if slab_size == 0 { 4096 } else { slab_size },
            tenant_budgets: HashMap::new(),
            watermarks: WatermarkTracker::new(),
            next_alloc_id: 0,
        }
    }

    pub fn register_tenant(&mut self, tenant_id: &str, max_bytes: usize) {
        self.tenant_budgets.insert(
            tenant_id.to_string(),
            TenantBudget {
                max_bytes,
                used_bytes: 0,
            },
        );
    }

    pub fn align_up(value: usize, alignment: usize) -> usize {
        if alignment == 0 {
            return value;
        }
        (value + alignment - 1) & !(alignment - 1)
    }

    pub fn allocate(&mut self, request: &AllocationRequest) -> Result<usize, String> {
        if let Some(ref tid) = request.tenant_id {
            if let Some(budget) = self.tenant_budgets.get(tid) {
                if budget.used_bytes + request.size > budget.max_bytes {
                    return Err(format!(
                        "tenant '{tid}' budget pub exceeded: used={}, requested={}, max={}",
                        budget.used_bytes, request.size, budget.max_bytes
                    ));
                }
            }
        }
        let effective_size = match self.strategy {
            AllocationStrategy::Slab => Self::align_up(request.size, self.slab_size),
            _ => Self::align_up(request.size, request.alignment.max(1)),
        };
        let idx = match self.strategy {
            AllocationStrategy::BestFit => self.find_best_fit(effective_size),
            _ => self.find_first_fit(effective_size),
        }
        .ok_or_else(|| {
            format!(
                "OOM: cannot allocate {effective_size} bytes in {} pool of {} bytes",
                self.strategy, self.total_bytes
            )
        })?;

        let (block_offset, block_size) = (self.blocks[idx].offset, self.blocks[idx].size);
        self.blocks[idx] = MemoryBlock {
            offset: block_offset,
            size: effective_size,
            allocated: true,
            tenant_id: request.tenant_id.clone(),
        };
        let remainder = block_size - effective_size;
        if remainder > 0 {
            self.blocks.insert(
                idx + 1,
                MemoryBlock {
                    offset: block_offset + effective_size,
                    size: remainder,
                    allocated: false,
                    tenant_id: None,
                },
            );
        }
        if let Some(ref tid) = request.tenant_id {
            if let Some(budget) = self.tenant_budgets.get_mut(tid) {
                budget.used_bytes += effective_size;
            }
        }
        self.next_alloc_id += 1;
        let used = self.used_bytes();
        self.watermarks.record(used);
        Ok(block_offset)
    }

    pub fn free(&mut self, offset: usize) -> Result<(), String> {
        let idx = self
            .blocks
            .iter()
            .position(|b| b.offset == offset && b.allocated)
            .ok_or_else(|| format!("no allocated block at offset {offset}"))?;
        if let Some(ref tid) = self.blocks[idx].tenant_id {
            if let Some(budget) = self.tenant_budgets.get_mut(tid) {
                budget.used_bytes = budget.used_bytes.saturating_sub(self.blocks[idx].size);
            }
        }
        self.blocks[idx].allocated = false;
        self.blocks[idx].tenant_id = None;
        self.coalesce(idx);
        let used = self.used_bytes();
        self.watermarks.record(used);
        Ok(())
    }

    pub fn used_bytes(&self) -> usize {
        self.blocks
            .iter()
            .filter(|b| b.allocated)
            .map(|b| b.size)
            .sum()
    }
    pub fn free_bytes(&self) -> usize {
        self.total_bytes - self.used_bytes()
    }

    pub fn fragmentation_stats(&self) -> FragmentationStats {
        let free_blocks: Vec<&MemoryBlock> = self.blocks.iter().filter(|b| !b.allocated).collect();
        let total_free: usize = free_blocks.iter().map(|b| b.size).sum();
        let largest = free_blocks.iter().map(|b| b.size).max().unwrap_or(0);
        let ratio = if total_free == 0 {
            0.0
        } else {
            1.0 - (largest as f64 / total_free as f64)
        };
        FragmentationStats {
            total_free,
            largest_free_block: largest,
            fragment_count: free_blocks.len(),
            fragmentation_ratio: ratio,
        }
    }

    pub fn defragment(&mut self) -> usize {
        let mut allocated: Vec<MemoryBlock> = self
            .blocks
            .iter()
            .filter(|b| b.allocated)
            .cloned()
            .collect();
        if allocated.is_empty() {
            self.blocks = vec![MemoryBlock {
                offset: 0,
                size: self.total_bytes,
                allocated: false,
                tenant_id: None,
            }];
            return 0;
        }
        let (mut offset, mut moved) = (0usize, 0usize);
        for block in &mut allocated {
            if block.offset != offset {
                moved += 1;
            }
            block.offset = offset;
            offset += block.size;
        }
        let remaining_free = self.total_bytes - offset;
        self.blocks = allocated;
        if remaining_free > 0 {
            self.blocks.push(MemoryBlock {
                offset,
                size: remaining_free,
                allocated: false,
                tenant_id: None,
            });
        }
        moved
    }

    pub fn find_first_fit(&self, size: usize) -> Option<usize> {
        self.blocks
            .iter()
            .position(|b| !b.allocated && b.size >= size)
    }

    pub fn find_best_fit(&self, size: usize) -> Option<usize> {
        self.blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| !b.allocated && b.size >= size)
            .min_by_key(|(_, b)| b.size)
            .map(|(i, _)| i)
    }

    pub fn coalesce(&mut self, idx: usize) {
        if idx + 1 < self.blocks.len() && !self.blocks[idx + 1].allocated {
            let next = self.blocks.remove(idx + 1);
            self.blocks[idx].size += next.size;
        }
        if idx > 0 && !self.blocks[idx - 1].allocated {
            let current = self.blocks.remove(idx);
            self.blocks[idx - 1].size += current.size;
        }
    }
}
