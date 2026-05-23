//! # apr ptx — Register Pressure Threshold Detector
//!
//! `apr ptx <FILE>` analyses CUDA PTX for register pressure issues.
//! The threshold for "high pressure" depends on SM architecture: SM 7.0
//! has 64K registers / SM, SM 8.0 has 64K, SM 9.0 has 64K. Per-thread
//! cap is 255 — using > 64 typically forces register spills. This recipe
//! builds the classifier.
//!
//! Demonstrates the **PTX.6** recipe for PMAT-111 (apr ptx coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender PTX-001 + NVIDIA CUDA Programming Guide
//!
//! Run with: cargo run --example cli_ptx_register_pressure_threshold
//!
//! Added by PMAT-111 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum PressureLevel {
    Healthy,                 // <= 32 regs/thread
    Moderate,                // 33-64 regs
    SpillRisk { regs: u32 }, // 65-128 regs
    HardCap { regs: u32 },   // 129-255 regs (often spills)
    InvalidCount,            // > 255 (impossible)
}

const HEALTHY_CAP: u32 = 32;
const MODERATE_CAP: u32 = 64;
const SPILL_CAP: u32 = 128;
const HARD_CAP: u32 = 255;

pub fn classify(regs: u32) -> PressureLevel {
    match regs {
        0..=HEALTHY_CAP => PressureLevel::Healthy,
        n if n <= MODERATE_CAP => PressureLevel::Moderate,
        n if n <= SPILL_CAP => PressureLevel::SpillRisk { regs: n },
        n if n <= HARD_CAP => PressureLevel::HardCap { regs: n },
        _ => PressureLevel::InvalidCount,
    }
}

pub fn occupancy_estimate(regs_per_thread: u32, threads_per_block: u32) -> Option<u32> {
    if regs_per_thread == 0 || threads_per_block == 0 {
        return None;
    }
    // Simple model: blocks per SM = 64K / (regs × threads), capped at 16.
    let regs_per_block = regs_per_thread * threads_per_block;
    if regs_per_block == 0 {
        return None;
    }
    let blocks = 65536 / regs_per_block;
    Some(blocks.min(16))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_ptx_register_pressure_threshold")?;

    for regs in [16u32, 32, 48, 64, 96, 128, 200, 255, 300] {
        let level = classify(regs);
        let occ = occupancy_estimate(regs, 256);
        println!("regs={regs:>3} → {level:?}  est blocks/SM (256 thr) = {occ:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn under_32_regs_healthy() {
        assert_eq!(classify(16), PressureLevel::Healthy);
        assert_eq!(classify(32), PressureLevel::Healthy);
    }

    #[test]
    fn moderate_range_33_to_64() {
        assert_eq!(classify(33), PressureLevel::Moderate);
        assert_eq!(classify(64), PressureLevel::Moderate);
    }

    #[test]
    fn spill_risk_range_65_to_128() {
        assert!(matches!(classify(65), PressureLevel::SpillRisk { .. }));
        assert!(matches!(classify(128), PressureLevel::SpillRisk { .. }));
    }

    #[test]
    fn hard_cap_range_129_to_255() {
        assert!(matches!(classify(129), PressureLevel::HardCap { .. }));
        assert!(matches!(classify(255), PressureLevel::HardCap { .. }));
    }

    #[test]
    fn over_255_invalid() {
        assert_eq!(classify(256), PressureLevel::InvalidCount);
        assert_eq!(classify(1000), PressureLevel::InvalidCount);
    }

    #[test]
    fn occupancy_zero_regs_returns_none() {
        // Avoid divide-by-zero.
        assert!(occupancy_estimate(0, 256).is_none());
        assert!(occupancy_estimate(32, 0).is_none());
    }

    #[test]
    fn occupancy_caps_at_16() {
        // SM has at most 16 active blocks.
        let blocks = occupancy_estimate(8, 32);
        assert!(blocks.unwrap() <= 16);
    }

    #[test]
    fn higher_register_pressure_lowers_occupancy() {
        let low = occupancy_estimate(16, 256).unwrap();
        let high = occupancy_estimate(128, 256).unwrap();
        assert!(high <= low, "high pressure should reduce occupancy");
    }
}
