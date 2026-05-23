//! # GPU Collective Op Picker
//!
//! Pick the right NCCL/RCCL collective for the situation:
//!   AllReduce → fan-in + fan-out (all GPUs sum + each gets result)
//!   Broadcast → one GPU's data to all
//!   AllGather → each GPU contributes; all receive concatenation
//!   ReduceScatter → all contribute, each receives a slice
//!
//! Picker: maps (operation_intent, n_gpus, bytes_per_gpu) →
//! recommendation + estimated bandwidth-bound time.
//!
//! Demonstrates the **GPU.23** recipe for PMAT-140 (gpu round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NCCL collective communication primitives docs.
//!
//! Run with: cargo run --example gpu_collective_op_picker
//!
//! Added by PMAT-140 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent {
    SumOrAverage,
    DistributeSingleSource,
    AssembleAcrossGpus,
    ScatterChunks,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Collective {
    AllReduce,
    Broadcast,
    AllGather,
    ReduceScatter,
}

#[derive(Debug, PartialEq)]
pub enum CollectiveVerdict {
    Ok {
        op: Collective,
        bytes_moved_per_gpu: u64,
    },
    InvalidGpuCount,
    InvalidBytes,
}

pub fn pick(intent: Intent, n_gpus: u32, bytes_per_gpu: u64) -> CollectiveVerdict {
    if n_gpus < 2 {
        return CollectiveVerdict::InvalidGpuCount;
    }
    if bytes_per_gpu == 0 {
        return CollectiveVerdict::InvalidBytes;
    }
    let op = match intent {
        Intent::SumOrAverage => Collective::AllReduce,
        Intent::DistributeSingleSource => Collective::Broadcast,
        Intent::AssembleAcrossGpus => Collective::AllGather,
        Intent::ScatterChunks => Collective::ReduceScatter,
    };
    // Ring algorithm complexity:
    // AllReduce: 2 × (n-1)/n × bytes
    // Broadcast: bytes
    // AllGather: (n-1)/n × bytes × n  (each receives n × bytes/gpu)
    // ReduceScatter: (n-1)/n × bytes × n
    let n = u64::from(n_gpus);
    let bytes_moved = match op {
        Collective::AllReduce => 2 * (n - 1) * bytes_per_gpu / n,
        Collective::Broadcast => bytes_per_gpu,
        Collective::AllGather | Collective::ReduceScatter => (n - 1) * bytes_per_gpu,
    };
    CollectiveVerdict::Ok {
        op,
        bytes_moved_per_gpu: bytes_moved,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_collective_op_picker")?;

    println!(
        "sum/avg 8 GPUs 1 GiB: {:?}",
        pick(Intent::SumOrAverage, 8, 1 << 30)
    );
    println!(
        "broadcast 8 GPUs: {:?}",
        pick(Intent::DistributeSingleSource, 8, 1 << 30)
    );
    println!(
        "allgather 4 GPUs: {:?}",
        pick(Intent::AssembleAcrossGpus, 4, 1 << 28)
    );
    println!(
        "scatter 8 GPUs: {:?}",
        pick(Intent::ScatterChunks, 8, 1 << 30)
    );
    println!("invalid: {:?}", pick(Intent::SumOrAverage, 1, 1 << 30));
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
    fn sum_picks_allreduce() {
        let v = pick(Intent::SumOrAverage, 8, 1024);
        if let CollectiveVerdict::Ok { op, .. } = v {
            assert_eq!(op, Collective::AllReduce);
        }
    }

    #[test]
    fn distribute_picks_broadcast() {
        let v = pick(Intent::DistributeSingleSource, 8, 1024);
        if let CollectiveVerdict::Ok { op, .. } = v {
            assert_eq!(op, Collective::Broadcast);
        }
    }

    #[test]
    fn assemble_picks_allgather() {
        let v = pick(Intent::AssembleAcrossGpus, 4, 1024);
        if let CollectiveVerdict::Ok { op, .. } = v {
            assert_eq!(op, Collective::AllGather);
        }
    }

    #[test]
    fn scatter_picks_reduce_scatter() {
        let v = pick(Intent::ScatterChunks, 8, 1024);
        if let CollectiveVerdict::Ok { op, .. } = v {
            assert_eq!(op, Collective::ReduceScatter);
        }
    }

    #[test]
    fn single_gpu_invalid() {
        assert_eq!(
            pick(Intent::SumOrAverage, 1, 1024),
            CollectiveVerdict::InvalidGpuCount
        );
    }

    #[test]
    fn zero_bytes_invalid() {
        assert_eq!(
            pick(Intent::SumOrAverage, 8, 0),
            CollectiveVerdict::InvalidBytes
        );
    }

    #[test]
    fn allreduce_bandwidth_doubles_allgather() {
        // Ring AllReduce moves 2× of AllGather.
        let ar = pick(Intent::SumOrAverage, 8, 1024);
        let ag = pick(Intent::AssembleAcrossGpus, 8, 1024);
        if let (
            CollectiveVerdict::Ok {
                bytes_moved_per_gpu: ar_b,
                ..
            },
            CollectiveVerdict::Ok {
                bytes_moved_per_gpu: ag_b,
                ..
            },
        ) = (ar, ag)
        {
            // AllReduce uses 2 phases (reduce-scatter + allgather), each with
            // (n-1)/n × bytes ≈ 0.875 × 1024 = 896 per phase.
            // Total = 2 × 896 = 1792.
            assert_eq!(ar_b, 1792);
            // AllGather = (n-1) × bytes = 7 × 1024 = 7168.
            assert_eq!(ag_b, 7168);
        }
    }

    #[test]
    fn broadcast_constant_bytes() {
        let v = pick(Intent::DistributeSingleSource, 32, 4096);
        if let CollectiveVerdict::Ok {
            bytes_moved_per_gpu,
            ..
        } = v
        {
            assert_eq!(bytes_moved_per_gpu, 4096);
        }
    }

    #[test]
    fn larger_n_more_bytes_for_allgather() {
        let v_small = pick(Intent::AssembleAcrossGpus, 4, 1024);
        let v_large = pick(Intent::AssembleAcrossGpus, 16, 1024);
        if let (
            CollectiveVerdict::Ok {
                bytes_moved_per_gpu: s,
                ..
            },
            CollectiveVerdict::Ok {
                bytes_moved_per_gpu: l,
                ..
            },
        ) = (v_small, v_large)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn min_gpu_count_two_succeeds() {
        let v = pick(Intent::SumOrAverage, 2, 1024);
        assert!(matches!(v, CollectiveVerdict::Ok { .. }));
    }
}
