//! # GPU Warp Specialization Picker
//!
//! Hopper (H100) supports warp specialization: split a block's warps
//! into producer/consumer roles for asynchronous TMA + tensor-core
//! pipelining. Picker rule:
//!   CC ≥ 9.0 + block ≥ 256 + memory-bound kernel → enable
//!   otherwise → general (homogeneous warps)
//!
//! Demonstrates the **GPU.27** recipe for PMAT-143 (gpu round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NVIDIA Hopper architecture WGMMA + warp-specialization paper.
//!
//! Run with: cargo run --example gpu_warp_specialization
//!
//! Added by PMAT-143 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    HomogeneousWarps,
    SpecializedProducerConsumer,
}

#[derive(Debug, PartialEq)]
pub enum WarpVerdict {
    Ok {
        strategy: Strategy,
        producer_warps: u32,
        consumer_warps: u32,
    },
    InvalidBlockSize,
    InvalidComputeCapability,
}

const WARP_SIZE: u32 = 32;
const MIN_BLOCK_FOR_SPECIALIZATION: u32 = 256;

pub fn pick(cc_major: u8, block_size: u32, memory_bound: bool) -> WarpVerdict {
    if cc_major == 0 {
        return WarpVerdict::InvalidComputeCapability;
    }
    if block_size == 0 || block_size % WARP_SIZE != 0 {
        return WarpVerdict::InvalidBlockSize;
    }
    let warps = block_size / WARP_SIZE;
    if cc_major >= 9 && block_size >= MIN_BLOCK_FOR_SPECIALIZATION && memory_bound {
        let producer = warps / 4;
        let consumer = warps - producer;
        return WarpVerdict::Ok {
            strategy: Strategy::SpecializedProducerConsumer,
            producer_warps: producer,
            consumer_warps: consumer,
        };
    }
    WarpVerdict::Ok {
        strategy: Strategy::HomogeneousWarps,
        producer_warps: 0,
        consumer_warps: warps,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_warp_specialization")?;

    println!("Hopper memory-bound: {:?}", pick(9, 512, true));
    println!("Hopper compute-bound: {:?}", pick(9, 512, false));
    println!("Ampere: {:?}", pick(8, 512, true));
    println!("small block: {:?}", pick(9, 128, true));
    println!("invalid: {:?}", pick(0, 256, true));
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
    fn hopper_mem_bound_specializes() {
        let v = pick(9, 512, true);
        if let WarpVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, Strategy::SpecializedProducerConsumer);
        }
    }

    #[test]
    fn hopper_compute_bound_homogeneous() {
        let v = pick(9, 512, false);
        if let WarpVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, Strategy::HomogeneousWarps);
        }
    }

    #[test]
    fn ampere_homogeneous() {
        let v = pick(8, 512, true);
        if let WarpVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, Strategy::HomogeneousWarps);
        }
    }

    #[test]
    fn small_block_homogeneous() {
        let v = pick(9, 128, true);
        if let WarpVerdict::Ok { strategy, .. } = v {
            assert_eq!(strategy, Strategy::HomogeneousWarps);
        }
    }

    #[test]
    fn invalid_zero_cc() {
        assert_eq!(pick(0, 256, true), WarpVerdict::InvalidComputeCapability);
    }

    #[test]
    fn invalid_zero_block() {
        assert_eq!(pick(9, 0, true), WarpVerdict::InvalidBlockSize);
    }

    #[test]
    fn invalid_unaligned_block() {
        assert_eq!(pick(9, 100, true), WarpVerdict::InvalidBlockSize);
    }

    #[test]
    fn producer_quarter_consumer_three_quarter() {
        // 16 warps total; producer = 4, consumer = 12.
        let v = pick(9, 512, true);
        if let WarpVerdict::Ok {
            producer_warps,
            consumer_warps,
            ..
        } = v
        {
            assert_eq!(producer_warps, 4);
            assert_eq!(consumer_warps, 12);
        }
    }

    #[test]
    fn homogeneous_zero_producers() {
        let v = pick(8, 512, true);
        if let WarpVerdict::Ok { producer_warps, .. } = v {
            assert_eq!(producer_warps, 0);
        }
    }

    #[test]
    fn homogeneous_all_consumers() {
        let v = pick(8, 512, true);
        if let WarpVerdict::Ok { consumer_warps, .. } = v {
            assert_eq!(consumer_warps, 16);
        }
    }
}
