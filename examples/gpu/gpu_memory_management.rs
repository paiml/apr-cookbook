//! # Recipe: GPU Memory Management
//!
//! **Category**: GPU Acceleration
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
//! Manage GPU memory efficiently to avoid OOM.
//!
//! ## Run Command
//! ```bash
//! cargo run --example gpu_memory_management
//! ```

use apr_cookbook::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("gpu_memory_management")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("GPU memory management strategies");
    println!();

    // GPU memory info
    let gpu = GpuMemoryInfo {
        total_mb: 24 * 1024, // 24GB
        reserved_mb: 512,    // Driver/system
    };

    let available = gpu.total_mb - gpu.reserved_mb;
    ctx.record_metric("gpu_total_mb", i64::from(gpu.total_mb));
    ctx.record_metric("gpu_available_mb", i64::from(available));

    println!("GPU Memory:");
    println!("  Total: {}MB ({}GB)", gpu.total_mb, gpu.total_mb / 1024);
    println!("  Reserved: {}MB", gpu.reserved_mb);
    println!("  Available: {}MB", available);
    println!();

    // Create memory pool
    let mut pool = GpuMemoryPool::new(available);

    // Simulate model loading
    let allocations = vec![
        ("model_weights", 8 * 1024),   // 8GB
        ("optimizer_state", 4 * 1024), // 4GB
        ("activations", 2 * 1024),     // 2GB
        ("gradients", 4 * 1024),       // 4GB
        ("kv_cache", 4 * 1024),        // 4GB
    ];

    println!("Memory Allocations:");
    println!("{:-<50}", "");

    for (name, size_mb) in &allocations {
        match pool.allocate(name, *size_mb) {
            Ok(handle) => {
                println!("  ✓ {} ({}MB) -> handle {}", name, size_mb, handle);
            }
            Err(e) => {
                println!("  ✗ {} ({}MB) -> {}", name, size_mb, e);
            }
        }
    }
    println!("{:-<50}", "");

    // Memory status
    let status = pool.status();
    println!();
    println!("Memory Status:");
    println!(
        "  Used: {}MB ({:.1}%)",
        status.used_mb,
        status.utilization * 100.0
    );
    println!("  Free: {}MB", status.free_mb);
    println!("  Allocations: {}", status.num_allocations);
    println!("  Fragmentation: {:.1}%", status.fragmentation * 100.0);

    ctx.record_float_metric("memory_utilization", status.utilization);

    // Demonstrate memory optimization
    println!();
    println!("Memory Optimization:");

    // Free some memory
    if let Some(handle) = pool.find_allocation("optimizer_state") {
        pool.free(handle)?;
        println!("  Freed optimizer_state (4GB)");
    }

    // Try gradient checkpointing (trade compute for memory)
    let checkpoint_savings = 2 * 1024; // Save 2GB
    println!("  Gradient checkpointing: saves {}MB", checkpoint_savings);

    // Try activation offloading
    if let Some(handle) = pool.find_allocation("activations") {
        pool.offload_to_cpu(handle)?;
        println!("  Offloaded activations to CPU");
    }

    // Final status
    let final_status = pool.status();
    println!();
    println!("Final Memory Status:");
    println!(
        "  Used: {}MB ({:.1}%)",
        final_status.used_mb,
        final_status.utilization * 100.0
    );
    println!("  Free: {}MB", final_status.free_mb);

    // Save memory log
    let log_path = ctx.path("memory_log.json");
    pool.save_log(&log_path)?;
    println!();
    println!("Memory log saved to: {:?}", log_path);

    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GpuMemoryInfo {
    total_mb: u32,
    reserved_mb: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryBlock {
    handle: u32,
    name: String,
    size_mb: u32,
    offloaded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryStatus {
    used_mb: u32,
    free_mb: u32,
    total_mb: u32,
    utilization: f64,
    num_allocations: usize,
    fragmentation: f64,
}

#[derive(Debug)]
struct GpuMemoryPool {
    total_mb: u32,
    blocks: Vec<MemoryBlock>,
    next_handle: u32,
    log: VecDeque<String>,
}

impl GpuMemoryPool {
    fn new(total_mb: u32) -> Self {
        Self {
            total_mb,
            blocks: Vec::new(),
            next_handle: 1,
            log: VecDeque::new(),
        }
    }

    fn allocate(&mut self, name: &str, size_mb: u32) -> Result<u32> {
        let used: u32 = self
            .blocks
            .iter()
            .filter(|b| !b.offloaded)
            .map(|b| b.size_mb)
            .sum();
        let free = self.total_mb - used;

        if size_mb > free {
            return Err(CookbookError::invalid_format(format!(
                "OOM: need {}MB, only {}MB free",
                size_mb, free
            )));
        }

        let handle = self.next_handle;
        self.next_handle += 1;

        self.blocks.push(MemoryBlock {
            handle,
            name: name.to_string(),
            size_mb,
            offloaded: false,
        });

        self.log
            .push_back(format!("ALLOC: {} ({}MB) -> {}", name, size_mb, handle));

        Ok(handle)
    }

    fn free(&mut self, handle: u32) -> Result<()> {
        let idx = self
            .blocks
            .iter()
            .position(|b| b.handle == handle)
            .ok_or_else(|| {
                CookbookError::invalid_format(format!("Invalid handle: {}", handle))
            })?;

        let block = self.blocks.remove(idx);
        self.log
            .push_back(format!("FREE: {} ({}MB)", block.name, block.size_mb));

        Ok(())
    }

    fn offload_to_cpu(&mut self, handle: u32) -> Result<()> {
        let block = self
            .blocks
            .iter_mut()
            .find(|b| b.handle == handle)
            .ok_or_else(|| {
                CookbookError::invalid_format(format!("Invalid handle: {}", handle))
            })?;

        block.offloaded = true;
        self.log.push_back(format!(
            "OFFLOAD: {} ({}MB) -> CPU",
            block.name, block.size_mb
        ));

        Ok(())
    }

    fn find_allocation(&self, name: &str) -> Option<u32> {
        self.blocks
            .iter()
            .find(|b| b.name == name)
            .map(|b| b.handle)
    }

    fn status(&self) -> MemoryStatus {
        let used: u32 = self
            .blocks
            .iter()
            .filter(|b| !b.offloaded)
            .map(|b| b.size_mb)
            .sum();
        let free = self.total_mb - used;
        let utilization = f64::from(used) / f64::from(self.total_mb);

        // Simple fragmentation estimate
        let fragmentation = if self.blocks.len() > 1 {
            0.05 * (self.blocks.len() - 1) as f64
        } else {
            0.0
        };

        MemoryStatus {
            used_mb: used,
            free_mb: free,
            total_mb: self.total_mb,
            utilization,
            num_allocations: self.blocks.len(),
            fragmentation: fragmentation.min(0.5),
        }
    }

    fn save_log(&self, path: &std::path::Path) -> Result<()> {
        #[derive(Serialize)]
        struct Log<'a> {
            operations: &'a VecDeque<String>,
            final_status: MemoryStatus,
        }

        let log = Log {
            operations: &self.log,
            final_status: self.status(),
        };

        let json = serde_json::to_string_pretty(&log)
            .map_err(|e| CookbookError::Serialization(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_creation() {
        let pool = GpuMemoryPool::new(1024);
        assert_eq!(pool.total_mb, 1024);
        assert!(pool.blocks.is_empty());
    }

    #[test]
    fn test_allocate() {
        let mut pool = GpuMemoryPool::new(1024);
        let handle = pool.allocate("test", 256).unwrap();

        assert_eq!(handle, 1);
        assert_eq!(pool.blocks.len(), 1);
    }

    #[test]
    fn test_allocate_oom() {
        let mut pool = GpuMemoryPool::new(100);
        let result = pool.allocate("too_big", 200);

        assert!(result.is_err());
    }

    #[test]
    fn test_free() {
        let mut pool = GpuMemoryPool::new(1024);
        let handle = pool.allocate("test", 256).unwrap();

        pool.free(handle).unwrap();
        assert!(pool.blocks.is_empty());
    }

    #[test]
    fn test_offload() {
        let mut pool = GpuMemoryPool::new(1024);
        let handle = pool.allocate("test", 256).unwrap();

        pool.offload_to_cpu(handle).unwrap();

        let status = pool.status();
        assert_eq!(status.used_mb, 0); // Offloaded doesn't count
    }

    #[test]
    fn test_status() {
        let mut pool = GpuMemoryPool::new(1000);
        pool.allocate("a", 400).unwrap();
        pool.allocate("b", 100).unwrap();

        let status = pool.status();

        assert_eq!(status.used_mb, 500);
        assert_eq!(status.free_mb, 500);
        assert!((status.utilization - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_find_allocation() {
        let mut pool = GpuMemoryPool::new(1024);
        pool.allocate("weights", 256).unwrap();

        let handle = pool.find_allocation("weights");
        assert!(handle.is_some());

        let none = pool.find_allocation("nonexistent");
        assert!(none.is_none());
    }

    #[test]
    fn test_save_log() {
        let ctx = RecipeContext::new("test_memory_log").unwrap();
        let path = ctx.path("log.json");

        let mut pool = GpuMemoryPool::new(1024);
        pool.allocate("test", 256).unwrap();
        pool.save_log(&path).unwrap();

        assert!(path.exists());
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_allocate_within_bounds(total in 100u32..1000, alloc in 1u32..100) {
            let mut pool = GpuMemoryPool::new(total);

            if alloc <= total {
                let result = pool.allocate("test", alloc);
                prop_assert!(result.is_ok());
            }
        }

        #[test]
        fn prop_utilization_bounded(sizes in proptest::collection::vec(10u32..100, 1..5)) {
            let total: u32 = sizes.iter().sum::<u32>() + 100;
            let mut pool = GpuMemoryPool::new(total);

            for (i, size) in sizes.iter().enumerate() {
                let _ = pool.allocate(&format!("block{}", i), *size);
            }

            let status = pool.status();
            prop_assert!(status.utilization >= 0.0);
            prop_assert!(status.utilization <= 1.0);
        }

        #[test]
        fn prop_free_reduces_used(total in 200u32..500, size in 50u32..100) {
            let mut pool = GpuMemoryPool::new(total);
            let handle = pool.allocate("test", size).unwrap();

            let before = pool.status().used_mb;
            pool.free(handle).unwrap();
            let after = pool.status().used_mb;

            prop_assert!(after < before);
        }
    }
}
