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
enum AllocationStrategy {
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
struct MemoryBlock {
    offset: usize,
    size: usize,
    allocated: bool,
    tenant_id: Option<String>,
}

#[derive(Debug, Clone)]
struct AllocationRequest {
    size: usize,
    tenant_id: Option<String>,
    alignment: usize,
}

#[derive(Debug, Clone)]
struct FragmentationStats {
    total_free: usize,
    largest_free_block: usize,
    fragment_count: usize,
    fragmentation_ratio: f64,
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
struct WatermarkTracker {
    peak_bytes: usize,
    current_bytes: usize,
    sample_count: u64,
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

    fn record(&mut self, allocated_bytes: usize) {
        self.current_bytes = allocated_bytes;
        if allocated_bytes > self.peak_bytes {
            self.peak_bytes = allocated_bytes;
        }
        self.sample_count += 1;
        self.cumulative_bytes += allocated_bytes as u64;
    }

    fn average_bytes(&self) -> f64 {
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
struct TenantBudget {
    max_bytes: usize,
    used_bytes: usize,
}

#[derive(Debug)]
struct MemoryPool {
    total_bytes: usize,
    blocks: Vec<MemoryBlock>,
    strategy: AllocationStrategy,
    slab_size: usize,
    tenant_budgets: HashMap<String, TenantBudget>,
    watermarks: WatermarkTracker,
    next_alloc_id: u64,
}

impl MemoryPool {
    fn new(total_bytes: usize, strategy: AllocationStrategy) -> Self {
        let slab_size = if strategy == AllocationStrategy::Slab {
            4096
        } else {
            0
        };
        Self::with_slab_size(total_bytes, strategy, slab_size)
    }

    fn with_slab_size(total_bytes: usize, strategy: AllocationStrategy, slab_size: usize) -> Self {
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

    fn register_tenant(&mut self, tenant_id: &str, max_bytes: usize) {
        self.tenant_budgets.insert(
            tenant_id.to_string(),
            TenantBudget {
                max_bytes,
                used_bytes: 0,
            },
        );
    }

    fn align_up(value: usize, alignment: usize) -> usize {
        if alignment == 0 {
            return value;
        }
        (value + alignment - 1) & !(alignment - 1)
    }

    fn allocate(&mut self, request: &AllocationRequest) -> Result<usize, String> {
        if let Some(ref tid) = request.tenant_id {
            if let Some(budget) = self.tenant_budgets.get(tid) {
                if budget.used_bytes + request.size > budget.max_bytes {
                    return Err(format!(
                        "tenant '{tid}' budget exceeded: used={}, requested={}, max={}",
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

    fn free(&mut self, offset: usize) -> Result<(), String> {
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

    fn used_bytes(&self) -> usize {
        self.blocks
            .iter()
            .filter(|b| b.allocated)
            .map(|b| b.size)
            .sum()
    }
    fn free_bytes(&self) -> usize {
        self.total_bytes - self.used_bytes()
    }

    fn fragmentation_stats(&self) -> FragmentationStats {
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

fn deterministic_rand(seed: u64, index: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    index.hash(&mut hasher);
    hasher.finish()
}

fn bench_strategy(
    label: &str,
    pool: &mut MemoryPool,
    num_allocs: usize,
    seed: u64,
    alignment: usize,
) {
    let start = Instant::now();
    let mut offsets = Vec::new();
    for i in 0..num_allocs {
        let size = (deterministic_rand(seed, i as u64) % 8192 + 128) as usize;
        match pool.allocate(&AllocationRequest {
            size,
            tenant_id: None,
            alignment,
        }) {
            Ok(offset) => offsets.push(offset),
            Err(_) => break,
        }
    }
    let alloc_time = start.elapsed();
    let start = Instant::now();
    for offset in offsets.iter().step_by(2) {
        let _ = pool.free(*offset);
    }
    let free_time = start.elapsed();
    println!(
        "  {label}: {}/{num_allocs} allocs in {alloc_time:?}, freed {} in {free_time:?}, frag: {}",
        offsets.len(),
        offsets.len() / 2,
        pool.fragmentation_stats()
    );
}

/// Print pool creation info.
fn print_pool_creation(pool_size: usize, slab_size: usize) {
    println!("--- 1. Pool Creation ---");
    let pool = MemoryPool::with_slab_size(pool_size, AllocationStrategy::Slab, slab_size);
    println!(
        "  {}B pool, {} strategy, {} slab, {} free\n",
        pool.total_bytes,
        pool.strategy,
        pool.slab_size,
        pool.free_bytes()
    );
}

/// Run strategy comparison benchmarks.
fn run_strategy_comparison(pool_size: usize, slab_size: usize, seed: u64) {
    println!("--- 2. Strategy Comparison ---");
    let mut slab_pool = MemoryPool::with_slab_size(pool_size, AllocationStrategy::Slab, slab_size);
    bench_strategy(&format!("Slab({slab_size})"), &mut slab_pool, 100, seed, 1);
    let mut bf_pool = MemoryPool::new(pool_size, AllocationStrategy::BestFit);
    bench_strategy("BestFit(64)", &mut bf_pool, 100, seed, 64);
    let mut ff_pool = MemoryPool::new(pool_size, AllocationStrategy::FirstFit);
    bench_strategy("FirstFit(64)", &mut ff_pool, 100, seed, 64);
    println!();
}

/// Run fragmentation analysis with allocate/free/defragment cycle.
fn run_fragmentation_analysis() {
    println!("--- 3. Fragmentation Analysis ---");
    let mut frag_pool = MemoryPool::new(64 * 1024, AllocationStrategy::FirstFit);
    let mut offsets = Vec::new();
    for i in 0..16 {
        if let Ok(off) = frag_pool.allocate(&AllocationRequest {
            size: 2048,
            tenant_id: Some(format!("t-{}", i % 3)),
            alignment: 1,
        }) {
            offsets.push(off);
        }
    }
    println!("  Before free: {}", frag_pool.fragmentation_stats());
    for offset in offsets.iter().step_by(2) {
        let _ = frag_pool.free(*offset);
    }
    println!("  After free:  {}", frag_pool.fragmentation_stats());
    let moved = frag_pool.defragment();
    println!(
        "  After defrag ({moved} moved): {}\n",
        frag_pool.fragmentation_stats()
    );
}

/// Run multi-tenant budget enforcement demo.
fn run_multi_tenant_demo() {
    println!("--- 4. Multi-Tenant Budgets ---");
    let mut mt_pool = MemoryPool::new(256 * 1024, AllocationStrategy::BestFit);
    mt_pool.register_tenant("alpha", 128 * 1024);
    mt_pool.register_tenant("beta", 64 * 1024);
    mt_pool.register_tenant("gamma", 64 * 1024);
    for (tid, size) in [
        ("alpha", 64 * 1024),
        ("alpha", 32 * 1024),
        ("beta", 32 * 1024),
        ("beta", 16 * 1024),
        ("gamma", 48 * 1024),
    ] {
        match mt_pool.allocate(&AllocationRequest {
            size,
            tenant_id: Some(tid.to_string()),
            alignment: 1,
        }) {
            Ok(off) => println!("  Alloc {size}B for '{tid}' @ {off}"),
            Err(e) => println!("  DENIED: {e}"),
        }
    }
    match mt_pool.allocate(&AllocationRequest {
        size: 32 * 1024,
        tenant_id: Some("beta".to_string()),
        alignment: 1,
    }) {
        Ok(_) => println!("  Unexpected: should be denied"),
        Err(e) => println!("  Budget enforced: {e}"),
    }
    println!();
}

/// Run watermark tracking demo with allocation waves.
fn run_watermark_tracking(seed: u64) {
    println!("--- 5. Watermark Tracking ---");
    let mut wm_pool = MemoryPool::new(128 * 1024, AllocationStrategy::FirstFit);
    let mut wm_offsets = Vec::new();
    for wave in 0..4 {
        for i in 0..(5 + wave * 2) {
            let size = (deterministic_rand(seed, (wave * 100 + i) as u64) % 4096 + 256) as usize;
            if let Ok(off) = wm_pool.allocate(&AllocationRequest {
                size,
                tenant_id: None,
                alignment: 1,
            }) {
                wm_offsets.push(off);
            }
        }
        for _ in 0..wm_offsets.len() / 3 {
            if let Some(off) = wm_offsets.pop() {
                let _ = wm_pool.free(off);
            }
        }
    }
    println!(
        "  {}\n  Utilization: {:.1}%\n",
        wm_pool.watermarks,
        wm_pool.used_bytes() as f64 / wm_pool.total_bytes as f64 * 100.0
    );
}

/// Print final strategy summary table with benchmarks.
fn print_strategy_summary(seed: u64) {
    println!("--- 6. Strategy Summary ---");
    let bench_pool_size: usize = 512 * 1024;
    println!(
        "  {:<10} {:>8} {:>10} {:>14} {:>12}",
        "Strategy", "Allocs", "Time", "Frag Ratio", "Peak Bytes"
    );
    println!("  {:-<58}", "");
    for (name, strategy) in [
        ("Slab", AllocationStrategy::Slab),
        ("BestFit", AllocationStrategy::BestFit),
        ("FirstFit", AllocationStrategy::FirstFit),
    ] {
        let mut bp = MemoryPool::with_slab_size(bench_pool_size, strategy, 4096);
        let mut bo = Vec::new();
        let start = Instant::now();
        for i in 0..200u64 {
            let size = (deterministic_rand(seed + 7, i) % 4096 + 64) as usize;
            let alignment = if strategy == AllocationStrategy::Slab {
                1
            } else {
                64
            };
            match bp.allocate(&AllocationRequest {
                size,
                tenant_id: None,
                alignment,
            }) {
                Ok(off) => bo.push(off),
                Err(_) => break,
            }
            if i % 3 == 0 {
                if let Some(off) = bo.first().copied() {
                    let _ = bp.free(off);
                    bo.remove(0);
                }
            }
        }
        let elapsed = start.elapsed();
        let stats = bp.fragmentation_stats();
        println!(
            "  {:<10} {:>8} {:>10?} {:>14.4} {:>12}",
            name,
            bo.len(),
            elapsed,
            stats.fragmentation_ratio,
            bp.watermarks.peak_bytes
        );
    }
}

fn main() {
    println!("=== Recipe: GPU Memory Pool Allocation ===\n");
    let (pool_size, slab_size, seed) = (1024 * 1024, 4096, 42u64);

    print_pool_creation(pool_size, slab_size);
    run_strategy_comparison(pool_size, slab_size, seed);
    run_fragmentation_analysis();
    run_multi_tenant_demo();
    run_watermark_tracking(seed);
    print_strategy_summary(seed);
    println!("\n=== GPU Memory Pool Recipe Complete ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(size: usize) -> AllocationRequest {
        AllocationRequest {
            size,
            tenant_id: None,
            alignment: 1,
        }
    }

    #[test]
    fn test_pool_creation() {
        let pool = MemoryPool::new(8192, AllocationStrategy::Slab);
        assert_eq!(pool.total_bytes, 8192);
        assert_eq!(pool.blocks.len(), 1);
        assert_eq!(pool.free_bytes(), 8192);
    }

    #[test]
    fn test_allocate_and_free() {
        let mut pool = MemoryPool::new(4096, AllocationStrategy::FirstFit);
        let off = pool.allocate(&req(2048)).expect("alloc");
        assert_eq!(pool.used_bytes(), 2048);
        pool.free(off).expect("free");
        assert_eq!(pool.used_bytes(), 0);
        assert_eq!(pool.free_bytes(), 4096);
        let off2 = pool.allocate(&req(2048)).expect("re-alloc");
        assert_eq!(off2, 0);
    }

    #[test]
    fn test_allocate_oom() {
        let mut pool = MemoryPool::new(512, AllocationStrategy::FirstFit);
        assert!(pool.allocate(&req(1024)).is_err());
    }

    #[test]
    fn test_slab_rounds_up() {
        let mut pool = MemoryPool::with_slab_size(16384, AllocationStrategy::Slab, 4096);
        pool.allocate(&req(100)).expect("alloc");
        assert_eq!(pool.used_bytes(), 4096);
    }

    #[test]
    fn test_bestfit_picks_smallest() {
        let mut pool = MemoryPool::new(16384, AllocationStrategy::BestFit);
        let off1 = pool.allocate(&req(4096)).expect("r1");
        let off2 = pool.allocate(&req(2048)).expect("r2");
        let _off3 = pool.allocate(&req(4096)).expect("r3");
        pool.free(off1).expect("f1");
        pool.free(off2).expect("f2");
        let off_small = pool.allocate(&req(2048)).expect("small");
        assert_eq!(off_small, 0);
    }

    #[test]
    fn test_fragmentation_and_defragment() {
        let mut pool = MemoryPool::new(8192, AllocationStrategy::FirstFit);
        let off0 = pool.allocate(&req(1024)).expect("a0");
        let _off1 = pool.allocate(&req(1024)).expect("a1");
        let off2 = pool.allocate(&req(1024)).expect("a2");
        let _off3 = pool.allocate(&req(1024)).expect("a3");
        pool.free(off0).expect("f0");
        pool.free(off2).expect("f2");
        let before = pool.fragmentation_stats();
        assert!(before.fragmentation_ratio > 0.0);
        let moved = pool.defragment();
        assert!(moved > 0);
        assert!(pool.fragmentation_stats().fragmentation_ratio < before.fragmentation_ratio);
    }

    #[test]
    fn test_tenant_budget_enforcement() {
        let mut pool = MemoryPool::new(16384, AllocationStrategy::FirstFit);
        pool.register_tenant("sm", 4096);
        assert!(pool
            .allocate(&AllocationRequest {
                size: 2048,
                tenant_id: Some("sm".to_string()),
                alignment: 1
            })
            .is_ok());
        assert!(pool
            .allocate(&AllocationRequest {
                size: 4096,
                tenant_id: Some("sm".to_string()),
                alignment: 1
            })
            .is_err());
    }

    #[test]
    fn test_tenant_budget_freed() {
        let mut pool = MemoryPool::new(16384, AllocationStrategy::FirstFit);
        pool.register_tenant("t1", 4096);
        let off = pool
            .allocate(&AllocationRequest {
                size: 2048,
                tenant_id: Some("t1".to_string()),
                alignment: 1,
            })
            .expect("alloc");
        assert_eq!(pool.tenant_budgets["t1"].used_bytes, 2048);
        pool.free(off).expect("free");
        assert_eq!(pool.tenant_budgets["t1"].used_bytes, 0);
    }

    #[test]
    fn test_watermark_tracker() {
        let mut t = WatermarkTracker::new();
        t.record(100);
        t.record(200);
        t.record(150);
        assert_eq!(t.peak_bytes, 200);
        assert_eq!(t.current_bytes, 150);
        assert!((t.average_bytes() - 150.0).abs() < 0.01);
    }

    #[test]
    fn test_align_up() {
        assert_eq!(MemoryPool::align_up(1, 64), 64);
        assert_eq!(MemoryPool::align_up(64, 64), 64);
        assert_eq!(MemoryPool::align_up(65, 64), 128);
        assert_eq!(MemoryPool::align_up(100, 0), 100);
    }

    #[test]
    fn test_coalesce_adjacent() {
        let mut pool = MemoryPool::new(8192, AllocationStrategy::FirstFit);
        let off0 = pool.allocate(&req(2048)).expect("a0");
        let off1 = pool.allocate(&req(2048)).expect("a1");
        let _off2 = pool.allocate(&req(2048)).expect("a2");
        pool.free(off0).expect("f0");
        pool.free(off1).expect("f1");
        assert!(!pool.blocks[0].allocated);
        assert_eq!(pool.blocks[0].size, 4096);
    }
}
