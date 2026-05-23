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
#[allow(unused_imports, clippy::wildcard_imports)]
use super::types::*;

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::time::Instant;

pub fn deterministic_rand(seed: u64, index: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    index.hash(&mut hasher);
    hasher.finish()
}

pub fn bench_strategy(
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
pub fn print_pool_creation(pool_size: usize, slab_size: usize) {
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
pub fn run_strategy_comparison(pool_size: usize, slab_size: usize, seed: u64) {
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
pub fn run_fragmentation_analysis() {
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
pub fn run_multi_tenant_demo() {
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
pub fn run_watermark_tracking(seed: u64) {
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
pub fn print_strategy_summary(seed: u64) {
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
