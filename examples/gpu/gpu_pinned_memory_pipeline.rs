//! # GPU Pinned-Memory Pipelined Copy
//!
//! Async H2D copy requires pinned (page-locked) host memory. Pipeline
//! depth = how many in-flight copies you keep:
//!   1 → no overlap (synchronous)
//!   2 → one stage hidden
//!   3-4 → maximum hide ratio with reasonable memory
//!   ≥ 5 → diminishing returns; memory hog
//!
//! Demonstrates the **GPU.37** recipe for PMAT-152 (milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NVIDIA pinned memory + cudaMemcpyAsync docs.
//!
//! Run with: cargo run --example gpu_pinned_memory_pipeline
//!
//! Added by PMAT-152 (catalog crosses 1000 recipes).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PipelineVerdict {
    Ok {
        pipeline_depth: u32,
        memory_overhead_bytes: u64,
        hide_ratio: f64,
    },
    InvalidSize,
    InsufficientMemory,
}

pub fn pick(buffer_size_bytes: u64, available_pinned_mem_bytes: u64) -> PipelineVerdict {
    if buffer_size_bytes == 0 {
        return PipelineVerdict::InvalidSize;
    }
    if available_pinned_mem_bytes < buffer_size_bytes {
        return PipelineVerdict::InsufficientMemory;
    }
    let max_depth = (available_pinned_mem_bytes / buffer_size_bytes).min(8) as u32;
    let pipeline_depth = max_depth.clamp(1, 4);
    let memory_overhead_bytes = u64::from(pipeline_depth) * buffer_size_bytes;
    let hide_ratio = match pipeline_depth {
        1 => 0.0,
        2 => 0.5,
        3 => 0.75,
        _ => 0.85,
    };
    PipelineVerdict::Ok {
        pipeline_depth,
        memory_overhead_bytes,
        hide_ratio,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_pinned_memory_pipeline")?;

    println!("ample memory: {:?}", pick(1024 * 1024, 100 * 1024 * 1024));
    println!(
        "small memory: {:?}",
        pick(50 * 1024 * 1024, 100 * 1024 * 1024)
    );
    println!(
        "exact memory: {:?}",
        pick(100 * 1024 * 1024, 100 * 1024 * 1024)
    );
    println!("insufficient: {:?}", pick(1024, 512));
    println!("invalid: {:?}", pick(0, 1024));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn ample_memory_max_depth() {
        let v = pick(1024 * 1024, 100 * 1024 * 1024);
        if let PipelineVerdict::Ok { pipeline_depth, .. } = v {
            assert_eq!(pipeline_depth, 4);
        }
    }

    #[test]
    fn small_memory_lower_depth() {
        let v = pick(50 * 1024 * 1024, 100 * 1024 * 1024);
        if let PipelineVerdict::Ok { pipeline_depth, .. } = v {
            assert_eq!(pipeline_depth, 2);
        }
    }

    #[test]
    fn exact_memory_depth_one() {
        let v = pick(100 * 1024 * 1024, 100 * 1024 * 1024);
        if let PipelineVerdict::Ok { pipeline_depth, .. } = v {
            assert_eq!(pipeline_depth, 1);
        }
    }

    #[test]
    fn insufficient_memory_rejected() {
        assert_eq!(pick(1024, 512), PipelineVerdict::InsufficientMemory);
    }

    #[test]
    fn invalid_zero_size() {
        assert_eq!(pick(0, 1024), PipelineVerdict::InvalidSize);
    }

    #[test]
    fn deeper_pipeline_hides_more() {
        let depth_1 = pick(50 * 1024 * 1024, 50 * 1024 * 1024);
        let depth_2 = pick(50 * 1024 * 1024, 100 * 1024 * 1024);
        if let (
            PipelineVerdict::Ok { hide_ratio: h1, .. },
            PipelineVerdict::Ok { hide_ratio: h2, .. },
        ) = (depth_1, depth_2)
        {
            assert!(h2 > h1);
        }
    }

    #[test]
    fn memory_overhead_proportional() {
        let v = pick(1024 * 1024, 100 * 1024 * 1024);
        if let PipelineVerdict::Ok {
            pipeline_depth,
            memory_overhead_bytes,
            ..
        } = v
        {
            assert_eq!(
                memory_overhead_bytes,
                u64::from(pipeline_depth) * 1024 * 1024
            );
        }
    }

    #[test]
    fn depth_capped_at_four() {
        // Tons of memory; capped at 4 depth.
        let v = pick(1024, 1024 * 1024 * 1024);
        if let PipelineVerdict::Ok { pipeline_depth, .. } = v {
            assert!(pipeline_depth <= 4);
        }
    }

    #[test]
    fn depth_one_zero_hide() {
        let v = pick(100 * 1024 * 1024, 100 * 1024 * 1024);
        if let PipelineVerdict::Ok { hide_ratio, .. } = v {
            assert_eq!(hide_ratio, 0.0);
        }
    }

    #[test]
    fn very_high_depth_85_pct() {
        let v = pick(1024, 1024 * 1024);
        if let PipelineVerdict::Ok { hide_ratio, .. } = v {
            assert!(hide_ratio >= 0.85);
        }
    }
}
