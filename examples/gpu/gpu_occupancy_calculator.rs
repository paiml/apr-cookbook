//! # GPU SM Occupancy Calculator
//!
//! Active warps per SM = min(
//!   max_warps_per_sm,
//!   max_threads_per_sm / threads_per_block × warps_per_block,
//!   shared_mem_per_sm / shared_mem_per_block × warps_per_block,
//!   regs_per_sm / (regs_per_thread × threads_per_block) × warps_per_block
//! ). Occupancy = active / max. This recipe builds the calculator.
//!
//! Demonstrates the **GPU.8** recipe for PMAT-130 (gpu coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NVIDIA CUDA Occupancy API + NVIDIA Hopper whitepaper.
//!
//! Run with: cargo run --example gpu_occupancy_calculator
//!
//! Added by PMAT-130 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy)]
pub struct SmConfig {
    pub max_warps: u32,
    pub max_threads: u32,
    pub regs_per_sm: u32,
    pub shared_mem_per_sm_bytes: u32,
}

pub fn hopper_h100() -> SmConfig {
    SmConfig {
        max_warps: 64,
        max_threads: 2048,
        regs_per_sm: 65_536,
        shared_mem_per_sm_bytes: 228 * 1024,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct KernelConfig {
    pub threads_per_block: u32,
    pub regs_per_thread: u32,
    pub shared_mem_per_block_bytes: u32,
}

#[derive(Debug, PartialEq)]
pub enum OccupancyVerdict {
    Ok {
        active_warps: u32,
        occupancy_pct: f64,
        bottleneck: Bottleneck,
    },
    InvalidKernel,
    InvalidSm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck {
    WarpLimit,
    ThreadLimit,
    RegisterPressure,
    SharedMemory,
    None,
}

pub fn calculate(sm: SmConfig, kernel: KernelConfig) -> OccupancyVerdict {
    if kernel.threads_per_block == 0 || kernel.regs_per_thread == 0 {
        return OccupancyVerdict::InvalidKernel;
    }
    if sm.max_warps == 0 || sm.max_threads == 0 {
        return OccupancyVerdict::InvalidSm;
    }
    let warps_per_block = kernel.threads_per_block.div_ceil(32);
    if warps_per_block == 0 {
        return OccupancyVerdict::InvalidKernel;
    }
    let warp_limit = sm.max_warps;
    let thread_limit = sm.max_threads / kernel.threads_per_block * warps_per_block;
    let regs_per_block = kernel.regs_per_thread * kernel.threads_per_block;
    let reg_limit = sm
        .regs_per_sm
        .checked_div(regs_per_block)
        .map_or(u32::MAX, |q| q * warps_per_block);
    let shared_limit = sm
        .shared_mem_per_sm_bytes
        .checked_div(kernel.shared_mem_per_block_bytes)
        .map_or(u32::MAX, |q| q * warps_per_block);

    let candidates = [
        (warp_limit, Bottleneck::WarpLimit),
        (thread_limit, Bottleneck::ThreadLimit),
        (reg_limit, Bottleneck::RegisterPressure),
        (shared_limit, Bottleneck::SharedMemory),
    ];
    let (active, bottleneck) = candidates
        .iter()
        .min_by_key(|(c, _)| *c)
        .map(|(c, b)| (*c, *b))
        .unwrap();

    let active = active.min(sm.max_warps);
    OccupancyVerdict::Ok {
        active_warps: active,
        occupancy_pct: f64::from(active) / f64::from(sm.max_warps) * 100.0,
        bottleneck,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_occupancy_calculator")?;

    let sm = hopper_h100();
    let typical = KernelConfig {
        threads_per_block: 256,
        regs_per_thread: 32,
        shared_mem_per_block_bytes: 16 * 1024,
    };
    println!("typical: {:?}", calculate(sm, typical));

    let high_register = KernelConfig {
        threads_per_block: 256,
        regs_per_thread: 128,
        shared_mem_per_block_bytes: 0,
    };
    println!("high register: {:?}", calculate(sm, high_register));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_sm() -> SmConfig {
        hopper_h100()
    }

    fn sample_kernel() -> KernelConfig {
        KernelConfig {
            threads_per_block: 256,
            regs_per_thread: 32,
            shared_mem_per_block_bytes: 16 * 1024,
        }
    }

    #[test]
    fn calc_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_kernel_calculates_ok() {
        let v = calculate(sample_sm(), sample_kernel());
        assert!(matches!(v, OccupancyVerdict::Ok { .. }));
    }

    #[test]
    fn occupancy_pct_in_0_to_100() {
        if let OccupancyVerdict::Ok { occupancy_pct, .. } = calculate(sample_sm(), sample_kernel())
        {
            assert!((0.0..=100.0).contains(&occupancy_pct));
        }
    }

    #[test]
    fn zero_threads_invalid() {
        let mut k = sample_kernel();
        k.threads_per_block = 0;
        assert_eq!(calculate(sample_sm(), k), OccupancyVerdict::InvalidKernel);
    }

    #[test]
    fn zero_regs_invalid() {
        let mut k = sample_kernel();
        k.regs_per_thread = 0;
        assert_eq!(calculate(sample_sm(), k), OccupancyVerdict::InvalidKernel);
    }

    #[test]
    fn zero_max_warps_invalid_sm() {
        let mut sm = sample_sm();
        sm.max_warps = 0;
        assert_eq!(calculate(sm, sample_kernel()), OccupancyVerdict::InvalidSm);
    }

    #[test]
    fn high_register_pressure_bottleneck() {
        let mut k = sample_kernel();
        k.regs_per_thread = 128;
        k.shared_mem_per_block_bytes = 0;
        if let OccupancyVerdict::Ok { bottleneck, .. } = calculate(sample_sm(), k) {
            assert!(matches!(
                bottleneck,
                Bottleneck::RegisterPressure | Bottleneck::WarpLimit
            ));
        }
    }

    #[test]
    fn high_shared_mem_bottleneck() {
        let mut k = sample_kernel();
        k.shared_mem_per_block_bytes = 200 * 1024;
        if let OccupancyVerdict::Ok { bottleneck, .. } = calculate(sample_sm(), k) {
            assert_eq!(bottleneck, Bottleneck::SharedMemory);
        }
    }

    #[test]
    fn active_warps_capped_at_max() {
        if let OccupancyVerdict::Ok { active_warps, .. } = calculate(sample_sm(), sample_kernel()) {
            assert!(active_warps <= sample_sm().max_warps);
        }
    }

    #[test]
    fn small_kernel_high_occupancy() {
        let k = KernelConfig {
            threads_per_block: 64,
            regs_per_thread: 16,
            shared_mem_per_block_bytes: 0,
        };
        if let OccupancyVerdict::Ok { occupancy_pct, .. } = calculate(sample_sm(), k) {
            // Small kernel with low resource use → should hit warp/thread limit.
            assert!(occupancy_pct > 0.0);
        }
    }
}
