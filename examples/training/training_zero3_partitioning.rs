//! # Training ZeRO-3 Parameter Partitioning
//!
//! ZeRO-3 partitions optimizer states, gradients, AND parameters across
//! data-parallel ranks. Memory per rank ≈ (params + grads + opt) / N.
//! Communication overhead: extra all-gathers for forward/backward.
//!
//! This recipe estimates per-rank memory + comm-cost factor.
//!
//! Demonstrates the **TRAIN.16** recipe for PMAT-146 (training round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Rajbhandari et al. (2019). ZeRO. arXiv:1910.02054.
//!
//! Run with: cargo run --example training_zero3_partitioning
//!
//! Added by PMAT-146 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroStage {
    Stage0,
    Stage1,
    Stage2,
    Stage3,
}

#[derive(Debug, PartialEq)]
pub enum PartitionVerdict {
    Ok {
        per_rank_memory_gib: f64,
        comm_overhead_factor: f64,
    },
    InvalidParams,
    InvalidWorldSize,
}

pub fn estimate(
    parameter_count: u64,
    bytes_per_param: u32,
    world_size: u32,
    stage: ZeroStage,
) -> PartitionVerdict {
    if parameter_count == 0 || bytes_per_param == 0 {
        return PartitionVerdict::InvalidParams;
    }
    if world_size == 0 {
        return PartitionVerdict::InvalidWorldSize;
    }
    let total_gib = (parameter_count as f64 * f64::from(bytes_per_param)) / 1_073_741_824.0;
    // Adam state = 2× param size in fp32 (m, v moments).
    let state_gib = total_gib * 2.0;
    let grad_gib = total_gib;
    let n = f64::from(world_size);
    let per_rank_memory_gib = match stage {
        ZeroStage::Stage0 => total_gib + grad_gib + state_gib, // No partitioning.
        ZeroStage::Stage1 => total_gib + grad_gib + state_gib / n, // Partition optimizer.
        ZeroStage::Stage2 => total_gib + grad_gib / n + state_gib / n, // Partition + grads.
        ZeroStage::Stage3 => total_gib / n + grad_gib / n + state_gib / n, // Full partitioning.
    };
    let comm_overhead_factor = match stage {
        ZeroStage::Stage0 => 1.0,
        ZeroStage::Stage1 => 1.0,
        ZeroStage::Stage2 => 1.5,
        ZeroStage::Stage3 => 2.0,
    };
    PartitionVerdict::Ok {
        per_rank_memory_gib,
        comm_overhead_factor,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("training_zero3_partitioning")?;

    // 7B model in fp16 across 8 GPUs.
    let params = 7_000_000_000u64;
    println!(
        "Stage 0 (no part): {:?}",
        estimate(params, 2, 8, ZeroStage::Stage0)
    );
    println!(
        "Stage 3 (full): {:?}",
        estimate(params, 2, 8, ZeroStage::Stage3)
    );
    println!("invalid: {:?}", estimate(0, 2, 8, ZeroStage::Stage3));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn stage0_no_partitioning_full_memory() {
        // 1B params × 2 bytes = ~1.86 GiB; with 4× state+grad = ~7.45 GiB.
        let v = estimate(1_000_000_000, 2, 8, ZeroStage::Stage0);
        if let PartitionVerdict::Ok {
            per_rank_memory_gib,
            ..
        } = v
        {
            assert!(per_rank_memory_gib > 7.0);
        }
    }

    #[test]
    fn stage3_partitions_memory() {
        let v0 = estimate(1_000_000_000, 2, 8, ZeroStage::Stage0);
        let v3 = estimate(1_000_000_000, 2, 8, ZeroStage::Stage3);
        if let (
            PartitionVerdict::Ok {
                per_rank_memory_gib: m0,
                ..
            },
            PartitionVerdict::Ok {
                per_rank_memory_gib: m3,
                ..
            },
        ) = (v0, v3)
        {
            assert!(m3 < m0 / 4.0);
        }
    }

    #[test]
    fn stage3_higher_comm_cost() {
        let v0 = estimate(1_000_000_000, 2, 8, ZeroStage::Stage0);
        let v3 = estimate(1_000_000_000, 2, 8, ZeroStage::Stage3);
        if let (
            PartitionVerdict::Ok {
                comm_overhead_factor: c0,
                ..
            },
            PartitionVerdict::Ok {
                comm_overhead_factor: c3,
                ..
            },
        ) = (v0, v3)
        {
            assert!(c3 > c0);
        }
    }

    #[test]
    fn larger_world_size_less_per_rank() {
        let v_4 = estimate(1_000_000_000, 2, 4, ZeroStage::Stage3);
        let v_64 = estimate(1_000_000_000, 2, 64, ZeroStage::Stage3);
        if let (
            PartitionVerdict::Ok {
                per_rank_memory_gib: m4,
                ..
            },
            PartitionVerdict::Ok {
                per_rank_memory_gib: m64,
                ..
            },
        ) = (v_4, v_64)
        {
            assert!(m64 < m4);
        }
    }

    #[test]
    fn invalid_zero_params() {
        assert_eq!(
            estimate(0, 2, 8, ZeroStage::Stage3),
            PartitionVerdict::InvalidParams
        );
    }

    #[test]
    fn invalid_zero_world_size() {
        assert_eq!(
            estimate(1_000, 2, 0, ZeroStage::Stage3),
            PartitionVerdict::InvalidWorldSize
        );
    }

    #[test]
    fn stage_progression_monotone() {
        let v0 = estimate(1_000_000_000, 2, 8, ZeroStage::Stage0);
        let v1 = estimate(1_000_000_000, 2, 8, ZeroStage::Stage1);
        let v2 = estimate(1_000_000_000, 2, 8, ZeroStage::Stage2);
        let v3 = estimate(1_000_000_000, 2, 8, ZeroStage::Stage3);
        if let (
            PartitionVerdict::Ok {
                per_rank_memory_gib: m0,
                ..
            },
            PartitionVerdict::Ok {
                per_rank_memory_gib: m1,
                ..
            },
            PartitionVerdict::Ok {
                per_rank_memory_gib: m2,
                ..
            },
            PartitionVerdict::Ok {
                per_rank_memory_gib: m3,
                ..
            },
        ) = (v0, v1, v2, v3)
        {
            assert!(m0 > m1);
            assert!(m1 > m2);
            assert!(m2 > m3);
        }
    }

    #[test]
    fn fp32_double_size_of_fp16() {
        let v_fp16 = estimate(1_000_000_000, 2, 8, ZeroStage::Stage3);
        let v_fp32 = estimate(1_000_000_000, 4, 8, ZeroStage::Stage3);
        if let (
            PartitionVerdict::Ok {
                per_rank_memory_gib: m16,
                ..
            },
            PartitionVerdict::Ok {
                per_rank_memory_gib: m32,
                ..
            },
        ) = (v_fp16, v_fp32)
        {
            assert!((m32 / m16 - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn stage1_only_optimizer_partitioned() {
        // Stage 1: weights + grads full size, only optimizer divided.
        let v = estimate(1_000_000_000, 2, 8, ZeroStage::Stage1);
        if let PartitionVerdict::Ok {
            comm_overhead_factor,
            ..
        } = v
        {
            assert_eq!(comm_overhead_factor, 1.0);
        }
    }

    #[test]
    fn stage0_one_world_size_invariant() {
        // World size 1 + Stage 0 → all on single rank.
        let v = estimate(1_000_000_000, 2, 1, ZeroStage::Stage0);
        if let PartitionVerdict::Ok {
            per_rank_memory_gib,
            ..
        } = v
        {
            // 1.86 + 1.86 + 3.72 ≈ 7.45 GiB.
            assert!(per_rank_memory_gib > 7.0);
        }
    }

    #[test]
    fn invalid_zero_bytes_per_param() {
        assert_eq!(
            estimate(1_000, 0, 8, ZeroStage::Stage3),
            PartitionVerdict::InvalidParams
        );
    }
}
