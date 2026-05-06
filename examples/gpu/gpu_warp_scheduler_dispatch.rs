//! # GPU Warp-Scheduler Dispatch Heuristic
//!
//! Pick (block_size, grid_size) for kernel launch:
//!   block_size = multiple of 32 (one warp), max 1024 (CUDA limit)
//!   grid_size = ceil(total_threads / block_size)
//!   prefer block_size 128 or 256 for occupancy on most CCs
//!
//! Plus: detect "tail" warps (last block not full).
//!
//! Demonstrates the **GPU.24** recipe for PMAT-140 (gpu round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: CUDA C++ Programming Guide § occupancy.
//!
//! Run with: cargo run --example gpu_warp_scheduler_dispatch
//!
//! Added by PMAT-140 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const WARP_SIZE: u32 = 32;
const MAX_BLOCK: u32 = 1024;
const PREFERRED_BLOCK: u32 = 256;

#[derive(Debug, PartialEq)]
pub enum DispatchVerdict {
    Ok {
        block_size: u32,
        grid_size: u32,
        tail_threads: u32,
    },
    InvalidThreadCount,
}

pub fn dispatch(total_threads: u64) -> DispatchVerdict {
    if total_threads == 0 {
        return DispatchVerdict::InvalidThreadCount;
    }
    let block_size = if total_threads >= u64::from(PREFERRED_BLOCK) {
        PREFERRED_BLOCK
    } else {
        // Round up to multiple of WARP_SIZE.
        let raw = total_threads as u32;
        raw.div_ceil(WARP_SIZE) * WARP_SIZE
    };
    let block_size = block_size.min(MAX_BLOCK);
    let grid_size = total_threads.div_ceil(u64::from(block_size));
    let total_launched = grid_size * u64::from(block_size);
    let tail_threads = (total_launched - total_threads) as u32;
    DispatchVerdict::Ok {
        block_size,
        grid_size: grid_size as u32,
        tail_threads,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_warp_scheduler_dispatch")?;

    for n in [1u64, 31, 256, 1000, 100_000, 0] {
        println!("threads={n}: {:?}", dispatch(n));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_threads_use_preferred_block() {
        let v = dispatch(100_000);
        if let DispatchVerdict::Ok { block_size, .. } = v {
            assert_eq!(block_size, PREFERRED_BLOCK);
        }
    }

    #[test]
    fn tiny_thread_count_rounds_up_to_warp() {
        // 31 threads → block size 32 (single warp).
        let v = dispatch(31);
        if let DispatchVerdict::Ok { block_size, .. } = v {
            assert_eq!(block_size, 32);
        }
    }

    #[test]
    fn single_thread_full_warp() {
        let v = dispatch(1);
        if let DispatchVerdict::Ok { block_size, .. } = v {
            assert_eq!(block_size, 32);
        }
    }

    #[test]
    fn zero_threads_invalid() {
        assert_eq!(dispatch(0), DispatchVerdict::InvalidThreadCount);
    }

    #[test]
    fn tail_threads_for_uneven_workload() {
        // 1000 threads, block 256 → 4 blocks = 1024 threads → 24 tail.
        let v = dispatch(1000);
        if let DispatchVerdict::Ok { tail_threads, .. } = v {
            assert_eq!(tail_threads, 24);
        }
    }

    #[test]
    fn even_workload_zero_tail() {
        // 256 threads, block 256 → 1 block, 0 tail.
        let v = dispatch(256);
        if let DispatchVerdict::Ok { tail_threads, .. } = v {
            assert_eq!(tail_threads, 0);
        }
    }

    #[test]
    fn grid_size_correct() {
        // 1000 threads, block 256 → 4 blocks.
        let v = dispatch(1000);
        if let DispatchVerdict::Ok { grid_size, .. } = v {
            assert_eq!(grid_size, 4);
        }
    }

    #[test]
    fn block_at_warp_multiple() {
        // Block size always multiple of WARP_SIZE.
        for n in [1u64, 31, 256, 1000, 100_000] {
            if let DispatchVerdict::Ok { block_size, .. } = dispatch(n) {
                assert_eq!(block_size % WARP_SIZE, 0, "n={n}");
            }
        }
    }

    #[test]
    fn block_at_or_below_max() {
        let v = dispatch(1_000_000);
        if let DispatchVerdict::Ok { block_size, .. } = v {
            assert!(block_size <= MAX_BLOCK);
        }
    }

    #[test]
    fn small_workload_single_block() {
        // 100 threads → single block of 128 (next warp multiple).
        let v = dispatch(100);
        if let DispatchVerdict::Ok { grid_size, .. } = v {
            assert_eq!(grid_size, 1);
        }
    }
}
