#![allow(unused_imports)]
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

mod helpers;
mod types;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use helpers::*;
#[allow(unused_imports)]
#[allow(clippy::wildcard_imports)]
use types::*;

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
