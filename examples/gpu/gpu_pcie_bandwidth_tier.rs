//! # GPU PCIe Bandwidth Tier Classifier
//!
//! Bidirectional PCIe peak (theoretical, single-direction):
//! - Gen3 x16 ≈ 16 GB/s, x8 ≈ 8 GB/s
//! - Gen4 x16 ≈ 32 GB/s, x8 ≈ 16 GB/s
//! - Gen5 x16 ≈ 64 GB/s, x8 ≈ 32 GB/s
//!
//! Useful to predict host-to-device transfer cost for model loading.
//! This recipe classifies a (gen, lanes) pair and reports peak GB/s
//! plus tier (Slow/Medium/Fast/VeryFast).
//!
//! Demonstrates the **GPU.22** recipe for PMAT-137 (gpu coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: PCI-SIG PCIe Base Specification 4.0, 5.0.
//!
//! Run with: cargo run --example gpu_pcie_bandwidth_tier
//!
//! Added by PMAT-137 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcieGen {
    Gen3,
    Gen4,
    Gen5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BandwidthTier {
    Slow,
    Medium,
    Fast,
    VeryFast,
}

#[derive(Debug, PartialEq)]
pub enum PcieVerdict {
    Ok { peak_gbps: u32, tier: BandwidthTier },
    InvalidLanes,
}

pub fn classify(gen: PcieGen, lanes: u8) -> PcieVerdict {
    if !matches!(lanes, 1 | 2 | 4 | 8 | 16) {
        return PcieVerdict::InvalidLanes;
    }
    let per_lane_gbps = match gen {
        PcieGen::Gen3 => 1,
        PcieGen::Gen4 => 2,
        PcieGen::Gen5 => 4,
    };
    let peak_gbps = per_lane_gbps * u32::from(lanes);
    let tier = match peak_gbps {
        0..=4 => BandwidthTier::Slow,
        5..=15 => BandwidthTier::Medium,
        16..=31 => BandwidthTier::Fast,
        _ => BandwidthTier::VeryFast,
    };
    PcieVerdict::Ok { peak_gbps, tier }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_pcie_bandwidth_tier")?;

    let cases = [
        (PcieGen::Gen3, 16),
        (PcieGen::Gen4, 16),
        (PcieGen::Gen5, 16),
        (PcieGen::Gen3, 8),
        (PcieGen::Gen5, 4),
    ];
    for (g, l) in cases {
        println!("{g:?} x{l}: {:?}", classify(g, l));
    }
    println!("invalid 5 lanes: {:?}", classify(PcieGen::Gen4, 5));
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
    fn gen3_x16_fast_tier() {
        let v = classify(PcieGen::Gen3, 16);
        assert_eq!(
            v,
            PcieVerdict::Ok {
                peak_gbps: 16,
                tier: BandwidthTier::Fast
            }
        );
    }

    #[test]
    fn gen4_x16_very_fast() {
        let v = classify(PcieGen::Gen4, 16);
        assert_eq!(
            v,
            PcieVerdict::Ok {
                peak_gbps: 32,
                tier: BandwidthTier::VeryFast
            }
        );
    }

    #[test]
    fn gen5_x16_very_fast_higher() {
        let v = classify(PcieGen::Gen5, 16);
        if let PcieVerdict::Ok { peak_gbps, tier } = v {
            assert_eq!(peak_gbps, 64);
            assert_eq!(tier, BandwidthTier::VeryFast);
        }
    }

    #[test]
    fn gen3_x8_medium() {
        let v = classify(PcieGen::Gen3, 8);
        assert_eq!(
            v,
            PcieVerdict::Ok {
                peak_gbps: 8,
                tier: BandwidthTier::Medium
            }
        );
    }

    #[test]
    fn gen3_x1_slow() {
        let v = classify(PcieGen::Gen3, 1);
        assert_eq!(
            v,
            PcieVerdict::Ok {
                peak_gbps: 1,
                tier: BandwidthTier::Slow
            }
        );
    }

    #[test]
    fn invalid_3_lanes_rejected() {
        assert_eq!(classify(PcieGen::Gen4, 3), PcieVerdict::InvalidLanes);
    }

    #[test]
    fn invalid_zero_lanes_rejected() {
        assert_eq!(classify(PcieGen::Gen4, 0), PcieVerdict::InvalidLanes);
    }

    #[test]
    fn gen5_higher_than_gen4_at_same_lanes() {
        let g4 = classify(PcieGen::Gen4, 16);
        let g5 = classify(PcieGen::Gen5, 16);
        if let (PcieVerdict::Ok { peak_gbps: b4, .. }, PcieVerdict::Ok { peak_gbps: b5, .. }) =
            (g4, g5)
        {
            assert!(b5 > b4);
        }
    }

    #[test]
    fn x8_half_x16_at_same_gen() {
        let x16 = classify(PcieGen::Gen5, 16);
        let x8 = classify(PcieGen::Gen5, 8);
        if let (PcieVerdict::Ok { peak_gbps: b16, .. }, PcieVerdict::Ok { peak_gbps: b8, .. }) =
            (x16, x8)
        {
            assert_eq!(b16, b8 * 2);
        }
    }

    #[test]
    fn gen5_x4_fast_tier() {
        let v = classify(PcieGen::Gen5, 4);
        assert_eq!(
            v,
            PcieVerdict::Ok {
                peak_gbps: 16,
                tier: BandwidthTier::Fast
            }
        );
    }
}
