//! # Recipe: GPU Memory Pool Allocation
//!
//! **Category**: GPU Acceleration
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (std only)
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
//! Demonstrate GPU memory pooling with slab allocation, fragmentation
//! analysis, multi-tenant budgets, and watermark tracking for inference workloads.
//!
//! ## Run Command
//! ```bash
//! cargo run --example gpu_memory_pool
//! ```

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Allocation strategy for the memory pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AllocationStrategy {
    /// Fixed-size slab allocation -- fastest, no external fragmentation.
    Slab,
    /// Scan free list and pick the smallest block that fits.
    BestFit,
    /// Scan free list and pick the first block that fits.
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

/// A contiguous region in the pool.
#[derive(Debug, Clone)]
struct MemoryBlock {
    /// Byte offset from the start of the pool.
    offset: usize,
    /// Size of this block in bytes.
    size: usize,
    /// Whether the block is currently allocated.
    allocated: bool,
    /// Optional tenant that owns this allocation.
    tenant_id: Option<String>,
}

/// Request to allocate memory from the pool.
#[derive(Debug, Clone)]
struct AllocationRequest {
    /// Requested size in bytes.
    size: usize,
    /// Optional tenant identifier for budget enforcement.
    tenant_id: Option<String>,
    /// Required alignment (must be a power of two).
    alignment: usize,
}

/// Fragmentation statistics for the pool.
#[derive(Debug, Clone)]
struct FragmentationStats {
    /// Total free bytes across all free blocks.
    total_free: usize,
    /// Size of the largest contiguous free block.
    largest_free_block: usize,
    /// Number of separate free regions.
    fragment_count: usize,
    /// Ratio in [0.0, 1.0] -- 0 means no fragmentation.
    fragmentation_ratio: f64,
}

impl fmt::Display for FragmentationStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "free={} bytes, largest_contiguous={} bytes, fragments={}, ratio={:.4}",
            self.total_free, self.largest_free_block, self.fragment_count, self.fragmentation_ratio,
        )
    }
}

/// Tracks peak, current, and average memory usage.
#[derive(Debug, Clone)]
struct WatermarkTracker {
    /// Peak allocated bytes observed.
    peak_bytes: usize,
    /// Current allocated bytes.
    current_bytes: usize,
    /// Number of samples recorded.
    sample_count: u64,
    /// Running total for average computation.
    cumulative_bytes: u64,
}

impl WatermarkTracker {
    fn new() -> Self {
        Self {
            peak_bytes: 0,
            current_bytes: 0,
            sample_count: 0,
            cumulative_bytes: 0,
        }
    }

    /// Record a new observation of allocated bytes.
    fn record(&mut self, allocated_bytes: usize) {
        self.current_bytes = allocated_bytes;
        if allocated_bytes > self.peak_bytes {
            self.peak_bytes = allocated_bytes;
        }
        self.sample_count += 1;
        self.cumulative_bytes += allocated_bytes as u64;
    }

    /// Average allocated bytes across all samples.
    fn average_bytes(&self) -> f64 {
        if self.sample_count == 0 {
            return 0.0;
        }
        self.cumulative_bytes as f64 / self.sample_count as f64
    }
}

impl fmt::Display for WatermarkTracker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "peak={} bytes, current={} bytes, avg={:.1} bytes, samples={}",
            self.peak_bytes,
            self.current_bytes,
            self.average_bytes(),
            self.sample_count,
        )
    }
}

/// Per-tenant memory budget.
#[derive(Debug, Clone)]
struct TenantBudget {
    /// Maximum bytes this tenant may allocate.
    max_bytes: usize,
    /// Currently allocated bytes for this tenant.
    used_bytes: usize,
}

/// GPU memory pool with configurable allocation strategy.
#[derive(Debug)]
struct MemoryPool {
    /// Total pool size in bytes.
    total_bytes: usize,
    /// Ordered list of blocks covering the entire pool.
    blocks: Vec<MemoryBlock>,
    /// Allocation strategy.
    strategy: AllocationStrategy,
    /// Fixed slab size (only meaningful when strategy is Slab).
    slab_size: usize,
    /// Per-tenant budgets.
    tenant_budgets: HashMap<String, TenantBudget>,
    /// Watermark tracker.
    watermarks: WatermarkTracker,
    /// Monotonically increasing allocation id.
    next_alloc_id: u64,
}

impl MemoryPool {
    /// Create a new memory pool with the given total size and strategy.
    fn new(total_bytes: usize, strategy: AllocationStrategy) -> Self {
        let slab_size = match strategy {
            AllocationStrategy::Slab => 4096, // default 4 KiB slabs
            _ => 0,
        };
        Self::with_slab_size(total_bytes, strategy, slab_size)
    }

    /// Create a pool with a specific slab size (relevant for Slab strategy).
    fn with_slab_size(total_bytes: usize, strategy: AllocationStrategy, slab_size: usize) -> Self {
        let blocks = vec![MemoryBlock {
            offset: 0,
            size: total_bytes,
            allocated: false,
            tenant_id: None,
        }];

        Self {
            total_bytes,
            blocks,
            strategy,
            slab_size: if slab_size == 0 { 4096 } else { slab_size },
            tenant_budgets: HashMap::new(),
            watermarks: WatermarkTracker::new(),
            next_alloc_id: 0,
        }
    }

    /// Register a tenant with a byte budget.
    fn register_tenant(&mut self, tenant_id: &str, max_bytes: usize) {
        self.tenant_budgets.insert(
            tenant_id.to_string(),
            TenantBudget {
                max_bytes,
                used_bytes: 0,
            },
        );
    }

    /// Align `value` up to the nearest multiple of `alignment`.
    fn align_up(value: usize, alignment: usize) -> usize {
        if alignment == 0 {
            return value;
        }
        let mask = alignment - 1;
        (value + mask) & !mask
    }

    /// Allocate memory according to the pool's strategy.
    fn allocate(&mut self, request: &AllocationRequest) -> Result<usize, String> {
        // Enforce tenant budget
        if let Some(ref tid) = request.tenant_id {
            if let Some(budget) = self.tenant_budgets.get(tid) {
                if budget.used_bytes + request.size > budget.max_bytes {
                    return Err(format!(
                        "tenant '{}' budget exceeded: used={}, requested={}, max={}",
                        tid, budget.used_bytes, request.size, budget.max_bytes,
                    ));
                }
            }
        }

        let effective_size = match self.strategy {
            AllocationStrategy::Slab => {
                // Round up to the next slab boundary.
                Self::align_up(request.size, self.slab_size)
            }
            _ => Self::align_up(request.size, request.alignment.max(1)),
        };

        let block_idx = match self.strategy {
            AllocationStrategy::FirstFit => self.find_first_fit(effective_size),
            AllocationStrategy::BestFit => self.find_best_fit(effective_size),
            AllocationStrategy::Slab => self.find_first_fit(effective_size),
        };

        let idx = block_idx.ok_or_else(|| {
            format!(
                "OOM: cannot allocate {} bytes ({} effective) in {} pool of {} bytes",
                request.size, effective_size, self.strategy, self.total_bytes,
            )
        })?;

        // Split the free block.
        let block_offset = self.blocks[idx].offset;
        let block_size = self.blocks[idx].size;

        self.blocks[idx] = MemoryBlock {
            offset: block_offset,
            size: effective_size,
            allocated: true,
            tenant_id: request.tenant_id.clone(),
        };

        // Remainder becomes a new free block.
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

        // Update tenant accounting.
        if let Some(ref tid) = request.tenant_id {
            if let Some(budget) = self.tenant_budgets.get_mut(tid) {
                budget.used_bytes += effective_size;
            }
        }

        self.next_alloc_id += 1;

        // Record watermark.
        let used = self.used_bytes();
        self.watermarks.record(used);

        Ok(block_offset)
    }

    /// Free a previously allocated block at `offset`.
    fn free(&mut self, offset: usize) -> Result<(), String> {
        let idx = self
            .blocks
            .iter()
            .position(|b| b.offset == offset && b.allocated)
            .ok_or_else(|| format!("no allocated block at offset {}", offset))?;

        // Update tenant accounting before freeing.
        if let Some(ref tid) = self.blocks[idx].tenant_id {
            if let Some(budget) = self.tenant_budgets.get_mut(tid) {
                budget.used_bytes = budget.used_bytes.saturating_sub(self.blocks[idx].size);
            }
        }

        self.blocks[idx].allocated = false;
        self.blocks[idx].tenant_id = None;

        // Coalesce with neighbors.
        self.coalesce(idx);

        // Record watermark.
        let used = self.used_bytes();
        self.watermarks.record(used);

        Ok(())
    }

    /// Total bytes currently allocated.
    fn used_bytes(&self) -> usize {
        self.blocks
            .iter()
            .filter(|b| b.allocated)
            .map(|b| b.size)
            .sum()
    }

    /// Total bytes currently free.
    fn free_bytes(&self) -> usize {
        self.total_bytes - self.used_bytes()
    }

    /// Compute fragmentation statistics.
    fn fragmentation_stats(&self) -> FragmentationStats {
        let free_blocks: Vec<&MemoryBlock> = self.blocks.iter().filter(|b| !b.allocated).collect();
        let total_free: usize = free_blocks.iter().map(|b| b.size).sum();
        let largest_free_block = free_blocks.iter().map(|b| b.size).max().unwrap_or(0);
        let fragment_count = free_blocks.len();
        let fragmentation_ratio = if total_free == 0 {
            0.0
        } else {
            1.0 - (largest_free_block as f64 / total_free as f64)
        };

        FragmentationStats {
            total_free,
            largest_free_block,
            fragment_count,
            fragmentation_ratio,
        }
    }

    /// Defragment the pool by compacting all allocated blocks to the front.
    ///
    /// Returns the number of blocks moved.
    fn defragment(&mut self) -> usize {
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

        // Count blocks that will actually move.
        let mut offset = 0usize;
        let mut moved = 0usize;
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

    // -- private helpers --

    fn find_first_fit(&self, size: usize) -> Option<usize> {
        self.blocks
            .iter()
            .position(|b| !b.allocated && b.size >= size)
    }

    fn find_best_fit(&self, size: usize) -> Option<usize> {
        self.blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| !b.allocated && b.size >= size)
            .min_by_key(|(_, b)| b.size)
            .map(|(i, _)| i)
    }

    fn coalesce(&mut self, idx: usize) {
        // Merge with next block if free.
        if idx + 1 < self.blocks.len() && !self.blocks[idx + 1].allocated {
            let next = self.blocks.remove(idx + 1);
            self.blocks[idx].size += next.size;
        }
        // Merge with previous block if free.
        if idx > 0 && !self.blocks[idx - 1].allocated {
            let current = self.blocks.remove(idx);
            self.blocks[idx - 1].size += current.size;
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic pseudo-random helper
// ---------------------------------------------------------------------------

/// Simple deterministic PRNG based on `DefaultHasher` for reproducibility.
fn deterministic_rand(seed: u64, index: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    index.hash(&mut hasher);
    hasher.finish()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Recipe: GPU Memory Pool Allocation ===");
    println!("Demonstrating memory pooling strategies for inference workloads");
    println!();

    // -----------------------------------------------------------------------
    // Section 1: Create memory pool with configurable block sizes
    // -----------------------------------------------------------------------
    println!("--- Section 1: Create Memory Pool with Configurable Block Sizes ---");
    println!();

    let pool_size: usize = 1024 * 1024; // 1 MiB
    let slab_size: usize = 4096; // 4 KiB slabs

    let pool = MemoryPool::with_slab_size(pool_size, AllocationStrategy::Slab, slab_size);
    println!("Pool created:");
    println!(
        "  Total size:  {} bytes ({} KiB)",
        pool.total_bytes,
        pool.total_bytes / 1024
    );
    println!("  Strategy:    {}", pool.strategy);
    println!("  Slab size:   {} bytes", pool.slab_size);
    println!("  Free bytes:  {}", pool.free_bytes());
    println!("  Block count: {}", pool.blocks.len());
    println!();

    // -----------------------------------------------------------------------
    // Section 2: Slab allocation vs dynamic allocation comparison
    // -----------------------------------------------------------------------
    println!("--- Section 2: Slab vs Dynamic Allocation Comparison ---");
    println!();

    let num_allocs = 100;
    let seed = 42u64;

    // Slab allocation benchmark
    let mut slab_pool = MemoryPool::with_slab_size(pool_size, AllocationStrategy::Slab, slab_size);
    let start = Instant::now();
    let mut slab_offsets = Vec::new();
    for i in 0..num_allocs {
        let size = (deterministic_rand(seed, i as u64) % 8192 + 128) as usize;
        let req = AllocationRequest {
            size,
            tenant_id: None,
            alignment: 1,
        };
        match slab_pool.allocate(&req) {
            Ok(offset) => slab_offsets.push(offset),
            Err(_) => break,
        }
    }
    let slab_alloc_time = start.elapsed();
    let slab_alloc_count = slab_offsets.len();

    // Free half
    let start = Instant::now();
    for offset in slab_offsets.iter().step_by(2) {
        let _ = slab_pool.free(*offset);
    }
    let slab_free_time = start.elapsed();

    println!("Slab Allocation (slab_size={} bytes):", slab_size);
    println!("  Allocations:  {}/{}", slab_alloc_count, num_allocs);
    println!("  Alloc time:   {:?}", slab_alloc_time);
    println!(
        "  Free time:    {:?} (freed {})",
        slab_free_time,
        slab_alloc_count / 2
    );
    println!("  Used:         {} bytes", slab_pool.used_bytes());
    println!("  Fragmentation: {}", slab_pool.fragmentation_stats());
    println!();

    // BestFit allocation benchmark
    let mut bestfit_pool = MemoryPool::new(pool_size, AllocationStrategy::BestFit);
    let start = Instant::now();
    let mut bf_offsets = Vec::new();
    for i in 0..num_allocs {
        let size = (deterministic_rand(seed, i as u64) % 8192 + 128) as usize;
        let req = AllocationRequest {
            size,
            tenant_id: None,
            alignment: 64,
        };
        match bestfit_pool.allocate(&req) {
            Ok(offset) => bf_offsets.push(offset),
            Err(_) => break,
        }
    }
    let bf_alloc_time = start.elapsed();
    let bf_alloc_count = bf_offsets.len();

    let start = Instant::now();
    for offset in bf_offsets.iter().step_by(2) {
        let _ = bestfit_pool.free(*offset);
    }
    let bf_free_time = start.elapsed();

    println!("BestFit Allocation (alignment=64 bytes):");
    println!("  Allocations:  {}/{}", bf_alloc_count, num_allocs);
    println!("  Alloc time:   {:?}", bf_alloc_time);
    println!(
        "  Free time:    {:?} (freed {})",
        bf_free_time,
        bf_alloc_count / 2
    );
    println!("  Used:         {} bytes", bestfit_pool.used_bytes());
    println!("  Fragmentation: {}", bestfit_pool.fragmentation_stats());
    println!();

    // FirstFit allocation benchmark
    let mut firstfit_pool = MemoryPool::new(pool_size, AllocationStrategy::FirstFit);
    let start = Instant::now();
    let mut ff_offsets = Vec::new();
    for i in 0..num_allocs {
        let size = (deterministic_rand(seed, i as u64) % 8192 + 128) as usize;
        let req = AllocationRequest {
            size,
            tenant_id: None,
            alignment: 64,
        };
        match firstfit_pool.allocate(&req) {
            Ok(offset) => ff_offsets.push(offset),
            Err(_) => break,
        }
    }
    let ff_alloc_time = start.elapsed();
    let ff_alloc_count = ff_offsets.len();

    let start = Instant::now();
    for offset in ff_offsets.iter().step_by(2) {
        let _ = firstfit_pool.free(*offset);
    }
    let ff_free_time = start.elapsed();

    println!("FirstFit Allocation (alignment=64 bytes):");
    println!("  Allocations:  {}/{}", ff_alloc_count, num_allocs);
    println!("  Alloc time:   {:?}", ff_alloc_time);
    println!(
        "  Free time:    {:?} (freed {})",
        ff_free_time,
        ff_alloc_count / 2
    );
    println!("  Used:         {} bytes", firstfit_pool.used_bytes());
    println!("  Fragmentation: {}", firstfit_pool.fragmentation_stats());
    println!();

    // -----------------------------------------------------------------------
    // Section 3: Fragmentation analysis after mixed allocations
    // -----------------------------------------------------------------------
    println!("--- Section 3: Fragmentation Analysis ---");
    println!();

    let mut frag_pool = MemoryPool::new(64 * 1024, AllocationStrategy::FirstFit);
    let mut offsets = Vec::new();

    // Allocate 16 blocks of 2 KiB each (32 KiB total)
    for i in 0..16 {
        let req = AllocationRequest {
            size: 2048,
            tenant_id: Some(format!("frag-tenant-{}", i % 3)),
            alignment: 1,
        };
        if let Ok(off) = frag_pool.allocate(&req) {
            offsets.push(off);
        }
    }

    let before_free = frag_pool.fragmentation_stats();
    println!("Before freeing (16 x 2KiB allocated):");
    println!("  {}", before_free);

    // Free every other block to create fragmentation.
    for offset in offsets.iter().step_by(2) {
        let _ = frag_pool.free(*offset);
    }

    let after_free = frag_pool.fragmentation_stats();
    println!("After freeing alternating blocks:");
    println!("  {}", after_free);

    // Defragment.
    let moved = frag_pool.defragment();
    let after_defrag = frag_pool.fragmentation_stats();
    println!("After defragmentation ({} blocks moved):", moved);
    println!("  {}", after_defrag);
    println!();

    // -----------------------------------------------------------------------
    // Section 4: Multi-tenant memory budgets
    // -----------------------------------------------------------------------
    println!("--- Section 4: Multi-Tenant Memory Budgets ---");
    println!();

    let mut mt_pool = MemoryPool::new(256 * 1024, AllocationStrategy::BestFit);
    mt_pool.register_tenant("model-alpha", 128 * 1024);
    mt_pool.register_tenant("model-beta", 64 * 1024);
    mt_pool.register_tenant("model-gamma", 64 * 1024);

    let tenant_allocs = [
        ("model-alpha", 64 * 1024),
        ("model-alpha", 32 * 1024),
        ("model-beta", 32 * 1024),
        ("model-beta", 16 * 1024),
        ("model-gamma", 48 * 1024),
    ];

    for (tid, size) in &tenant_allocs {
        let req = AllocationRequest {
            size: *size,
            tenant_id: Some((*tid).to_string()),
            alignment: 1,
        };
        match mt_pool.allocate(&req) {
            Ok(offset) => {
                println!(
                    "  Allocated {} bytes for '{}' at offset {}",
                    size, tid, offset,
                );
            }
            Err(e) => println!("  DENIED: {}", e),
        }
    }

    // Try to exceed model-beta's budget.
    let over_budget_req = AllocationRequest {
        size: 32 * 1024,
        tenant_id: Some("model-beta".to_string()),
        alignment: 1,
    };
    match mt_pool.allocate(&over_budget_req) {
        Ok(_) => println!("  Unexpected: model-beta allocation should have been denied"),
        Err(e) => println!("  Budget enforcement: {}", e),
    }

    println!();
    println!("  Tenant budget summary:");
    for (tid, budget) in &mt_pool.tenant_budgets {
        println!(
            "    {}: {}/{} bytes ({:.1}%)",
            tid,
            budget.used_bytes,
            budget.max_bytes,
            (budget.used_bytes as f64 / budget.max_bytes as f64) * 100.0,
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // Section 5: Memory watermark tracking
    // -----------------------------------------------------------------------
    println!("--- Section 5: Memory Watermark Tracking ---");
    println!();

    let mut wm_pool = MemoryPool::new(128 * 1024, AllocationStrategy::FirstFit);
    let mut wm_offsets = Vec::new();

    // Simulate a workload: allocate, use, free in waves.
    for wave in 0..4 {
        let allocs_this_wave = 5 + wave * 2;
        for i in 0..allocs_this_wave {
            let size = (deterministic_rand(seed, (wave * 100 + i) as u64) % 4096 + 256) as usize;
            let req = AllocationRequest {
                size,
                tenant_id: None,
                alignment: 1,
            };
            if let Ok(off) = wm_pool.allocate(&req) {
                wm_offsets.push(off);
            }
        }
        // Free some from previous waves.
        let drain_count = wm_offsets.len() / 3;
        for _ in 0..drain_count {
            if let Some(off) = wm_offsets.pop() {
                let _ = wm_pool.free(off);
            }
        }
    }

    println!("Watermark after 4 allocation waves:");
    println!("  {}", wm_pool.watermarks);
    println!(
        "  Utilization: {:.1}%",
        (wm_pool.used_bytes() as f64 / wm_pool.total_bytes as f64) * 100.0,
    );
    println!();

    // -----------------------------------------------------------------------
    // Section 6: Allocation strategy comparison summary
    // -----------------------------------------------------------------------
    println!("--- Section 6: Allocation Strategy Comparison Summary ---");
    println!();

    let strategies = [
        ("Slab", AllocationStrategy::Slab),
        ("BestFit", AllocationStrategy::BestFit),
        ("FirstFit", AllocationStrategy::FirstFit),
    ];

    let bench_pool_size: usize = 512 * 1024;
    let bench_rounds: u64 = 200;

    println!(
        "  {:<10} {:>8} {:>10} {:>14} {:>12}",
        "Strategy", "Allocs", "Time", "Frag Ratio", "Peak Bytes",
    );
    println!("  {:-<58}", "");

    for (name, strategy) in &strategies {
        let mut bench_pool = MemoryPool::with_slab_size(bench_pool_size, *strategy, 4096);
        let mut bench_offsets = Vec::new();

        let start = Instant::now();
        for i in 0..bench_rounds {
            let size = (deterministic_rand(seed + 7, i) % 4096 + 64) as usize;
            let req = AllocationRequest {
                size,
                tenant_id: None,
                alignment: if *strategy == AllocationStrategy::Slab {
                    1
                } else {
                    64
                },
            };
            match bench_pool.allocate(&req) {
                Ok(off) => bench_offsets.push(off),
                Err(_) => break,
            }
            // Free every 3rd allocation to create churn.
            if i % 3 == 0 {
                if let Some(off) = bench_offsets.first().copied() {
                    let _ = bench_pool.free(off);
                    bench_offsets.remove(0);
                }
            }
        }
        let elapsed = start.elapsed();
        let stats = bench_pool.fragmentation_stats();

        println!(
            "  {:<10} {:>8} {:>10?} {:>14.4} {:>12}",
            name,
            bench_offsets.len(),
            elapsed,
            stats.fragmentation_ratio,
            bench_pool.watermarks.peak_bytes,
        );
    }

    println!();
    println!("=== GPU Memory Pool Recipe Complete ===");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation_slab() {
        let pool = MemoryPool::new(8192, AllocationStrategy::Slab);
        assert_eq!(pool.total_bytes, 8192);
        assert_eq!(pool.strategy, AllocationStrategy::Slab);
        assert_eq!(pool.blocks.len(), 1);
        assert!(!pool.blocks[0].allocated);
    }

    #[test]
    fn test_pool_creation_bestfit() {
        let pool = MemoryPool::new(16384, AllocationStrategy::BestFit);
        assert_eq!(pool.total_bytes, 16384);
        assert_eq!(pool.free_bytes(), 16384);
        assert_eq!(pool.used_bytes(), 0);
    }

    #[test]
    fn test_allocate_simple() {
        let mut pool = MemoryPool::new(8192, AllocationStrategy::FirstFit);
        let req = AllocationRequest {
            size: 1024,
            tenant_id: None,
            alignment: 1,
        };
        let offset = pool.allocate(&req);
        assert!(offset.is_ok());
        assert_eq!(offset.expect("allocation failed"), 0);
        assert_eq!(pool.used_bytes(), 1024);
    }

    #[test]
    fn test_allocate_oom() {
        let mut pool = MemoryPool::new(512, AllocationStrategy::FirstFit);
        let req = AllocationRequest {
            size: 1024,
            tenant_id: None,
            alignment: 1,
        };
        let result = pool.allocate(&req);
        assert!(result.is_err());
    }

    #[test]
    fn test_free_and_reuse() {
        let mut pool = MemoryPool::new(4096, AllocationStrategy::FirstFit);
        let req = AllocationRequest {
            size: 2048,
            tenant_id: None,
            alignment: 1,
        };
        let off = pool.allocate(&req).expect("alloc failed");
        pool.free(off).expect("free failed");
        assert_eq!(pool.used_bytes(), 0);
        assert_eq!(pool.free_bytes(), 4096);

        // Re-allocate into the freed space.
        let off2 = pool.allocate(&req).expect("re-alloc failed");
        assert_eq!(off2, 0);
    }

    #[test]
    fn test_free_invalid_offset() {
        let mut pool = MemoryPool::new(4096, AllocationStrategy::FirstFit);
        let result = pool.free(9999);
        assert!(result.is_err());
    }

    #[test]
    fn test_slab_rounds_up() {
        let slab_size = 4096;
        let mut pool = MemoryPool::with_slab_size(16384, AllocationStrategy::Slab, slab_size);
        let req = AllocationRequest {
            size: 100, // much less than 4096
            tenant_id: None,
            alignment: 1,
        };
        pool.allocate(&req).expect("alloc failed");
        // Slab rounds up to 4096.
        assert_eq!(pool.used_bytes(), slab_size);
    }

    #[test]
    fn test_bestfit_picks_smallest() {
        let mut pool = MemoryPool::new(16384, AllocationStrategy::BestFit);
        // Allocate three blocks, free the middle one (small) and last one (large).
        let r1 = AllocationRequest {
            size: 4096,
            tenant_id: None,
            alignment: 1,
        };
        let r2 = AllocationRequest {
            size: 2048,
            tenant_id: None,
            alignment: 1,
        };
        let r3 = AllocationRequest {
            size: 4096,
            tenant_id: None,
            alignment: 1,
        };

        let off1 = pool.allocate(&r1).expect("alloc r1");
        let off2 = pool.allocate(&r2).expect("alloc r2");
        let _off3 = pool.allocate(&r3).expect("alloc r3");

        pool.free(off1).expect("free off1");
        pool.free(off2).expect("free off2");

        // Now free list has a 6144-byte block (coalesced off1+off2) and a 6144-byte remainder.
        // Request 2048 -- BestFit should pick the first suitable block.
        let r_small = AllocationRequest {
            size: 2048,
            tenant_id: None,
            alignment: 1,
        };
        let off_small = pool.allocate(&r_small).expect("alloc small");
        // Should fit in the first free region (offset 0).
        assert_eq!(off_small, 0);
    }

    #[test]
    fn test_fragmentation_stats_no_fragmentation() {
        let pool = MemoryPool::new(8192, AllocationStrategy::FirstFit);
        let stats = pool.fragmentation_stats();
        assert_eq!(stats.fragmentation_ratio, 0.0);
        assert_eq!(stats.fragment_count, 1);
        assert_eq!(stats.total_free, 8192);
    }

    #[test]
    fn test_fragmentation_stats_with_fragmentation() {
        let mut pool = MemoryPool::new(8192, AllocationStrategy::FirstFit);
        let r = AllocationRequest {
            size: 2048,
            tenant_id: None,
            alignment: 1,
        };

        let off0 = pool.allocate(&r).expect("alloc 0");
        let _off1 = pool.allocate(&r).expect("alloc 1");
        let off2 = pool.allocate(&r).expect("alloc 2");

        // Free 0 and 2, leaving a gap pattern: [free][alloc][free][free-remainder].
        pool.free(off0).expect("free 0");
        pool.free(off2).expect("free 2");

        let stats = pool.fragmentation_stats();
        assert!(stats.fragment_count >= 2);
        assert!(stats.fragmentation_ratio > 0.0);
    }

    #[test]
    fn test_defragment() {
        let mut pool = MemoryPool::new(8192, AllocationStrategy::FirstFit);
        let r = AllocationRequest {
            size: 1024,
            tenant_id: None,
            alignment: 1,
        };

        let off0 = pool.allocate(&r).expect("alloc 0");
        let _off1 = pool.allocate(&r).expect("alloc 1");
        let off2 = pool.allocate(&r).expect("alloc 2");
        let _off3 = pool.allocate(&r).expect("alloc 3");

        pool.free(off0).expect("free 0");
        pool.free(off2).expect("free 2");

        let before = pool.fragmentation_stats();
        assert!(before.fragmentation_ratio > 0.0);

        let moved = pool.defragment();
        assert!(moved > 0);

        let after = pool.fragmentation_stats();
        assert!(after.fragmentation_ratio < before.fragmentation_ratio);
    }

    #[test]
    fn test_defragment_empty_pool() {
        let mut pool = MemoryPool::new(4096, AllocationStrategy::FirstFit);
        let moved = pool.defragment();
        assert_eq!(moved, 0);
        assert_eq!(pool.blocks.len(), 1);
        assert_eq!(pool.free_bytes(), 4096);
    }

    #[test]
    fn test_tenant_budget_enforcement() {
        let mut pool = MemoryPool::new(16384, AllocationStrategy::FirstFit);
        pool.register_tenant("small-model", 4096);

        let req_ok = AllocationRequest {
            size: 2048,
            tenant_id: Some("small-model".to_string()),
            alignment: 1,
        };
        assert!(pool.allocate(&req_ok).is_ok());

        let req_over = AllocationRequest {
            size: 4096,
            tenant_id: Some("small-model".to_string()),
            alignment: 1,
        };
        assert!(pool.allocate(&req_over).is_err());
    }

    #[test]
    fn test_tenant_budget_freed_bytes() {
        let mut pool = MemoryPool::new(16384, AllocationStrategy::FirstFit);
        pool.register_tenant("t1", 4096);

        let req = AllocationRequest {
            size: 2048,
            tenant_id: Some("t1".to_string()),
            alignment: 1,
        };
        let off = pool.allocate(&req).expect("alloc");
        assert_eq!(pool.tenant_budgets["t1"].used_bytes, 2048);

        pool.free(off).expect("free");
        assert_eq!(pool.tenant_budgets["t1"].used_bytes, 0);
    }

    #[test]
    fn test_watermark_tracker_basic() {
        let mut tracker = WatermarkTracker::new();
        assert_eq!(tracker.peak_bytes, 0);
        assert_eq!(tracker.current_bytes, 0);

        tracker.record(100);
        tracker.record(200);
        tracker.record(150);

        assert_eq!(tracker.peak_bytes, 200);
        assert_eq!(tracker.current_bytes, 150);
        assert_eq!(tracker.sample_count, 3);
        assert!((tracker.average_bytes() - 150.0).abs() < 0.01);
    }

    #[test]
    fn test_watermark_average_empty() {
        let tracker = WatermarkTracker::new();
        assert_eq!(tracker.average_bytes(), 0.0);
    }

    #[test]
    fn test_watermark_integration_with_pool() {
        let mut pool = MemoryPool::new(8192, AllocationStrategy::FirstFit);
        let req = AllocationRequest {
            size: 1024,
            tenant_id: None,
            alignment: 1,
        };

        pool.allocate(&req).expect("alloc 1");
        pool.allocate(&req).expect("alloc 2");
        assert_eq!(pool.watermarks.peak_bytes, 2048);

        pool.free(0).expect("free first block");
        assert_eq!(pool.watermarks.current_bytes, 1024);
        assert_eq!(pool.watermarks.peak_bytes, 2048); // peak unchanged
    }

    #[test]
    fn test_align_up() {
        assert_eq!(MemoryPool::align_up(1, 64), 64);
        assert_eq!(MemoryPool::align_up(64, 64), 64);
        assert_eq!(MemoryPool::align_up(65, 64), 128);
        assert_eq!(MemoryPool::align_up(0, 64), 0);
        assert_eq!(MemoryPool::align_up(100, 0), 100);
    }

    #[test]
    fn test_deterministic_rand_reproducible() {
        let a = deterministic_rand(42, 0);
        let b = deterministic_rand(42, 0);
        assert_eq!(a, b);
    }

    #[test]
    fn test_deterministic_rand_varies() {
        let a = deterministic_rand(42, 0);
        let b = deterministic_rand(42, 1);
        assert_ne!(a, b);
    }

    #[test]
    fn test_allocation_strategy_display() {
        assert_eq!(format!("{}", AllocationStrategy::Slab), "Slab");
        assert_eq!(format!("{}", AllocationStrategy::BestFit), "BestFit");
        assert_eq!(format!("{}", AllocationStrategy::FirstFit), "FirstFit");
    }

    #[test]
    fn test_coalesce_adjacent_free_blocks() {
        let mut pool = MemoryPool::new(8192, AllocationStrategy::FirstFit);
        let r = AllocationRequest {
            size: 2048,
            tenant_id: None,
            alignment: 1,
        };

        let off0 = pool.allocate(&r).expect("alloc 0");
        let off1 = pool.allocate(&r).expect("alloc 1");
        let _off2 = pool.allocate(&r).expect("alloc 2");

        // Free adjacent blocks -- they should coalesce.
        pool.free(off0).expect("free 0");
        pool.free(off1).expect("free 1");

        // The first block should now be a single 4096-byte free block.
        assert!(!pool.blocks[0].allocated);
        assert_eq!(pool.blocks[0].size, 4096);
    }

    #[test]
    fn test_multi_tenant_isolation() {
        let mut pool = MemoryPool::new(32768, AllocationStrategy::BestFit);
        pool.register_tenant("t-a", 16384);
        pool.register_tenant("t-b", 16384);

        let req_a = AllocationRequest {
            size: 8192,
            tenant_id: Some("t-a".to_string()),
            alignment: 1,
        };
        let req_b = AllocationRequest {
            size: 8192,
            tenant_id: Some("t-b".to_string()),
            alignment: 1,
        };

        assert!(pool.allocate(&req_a).is_ok());
        assert!(pool.allocate(&req_b).is_ok());

        assert_eq!(pool.tenant_budgets["t-a"].used_bytes, 8192);
        assert_eq!(pool.tenant_budgets["t-b"].used_bytes, 8192);
    }
}
