//! # GPU PCIe / NVLink P2P Topology Picker
//!
//! Multi-GPU communication links:
//!   PCIeGen3:  ~16 GB/s
//!   PCIeGen4:  ~32 GB/s
//!   PCIeGen5:  ~64 GB/s
//!   NVLink3:   ~150 GB/s
//!   NVLink4:   ~225 GB/s
//!   NVSwitch4: ~450 GB/s (full bisection)
//!
//! Picker maps (gpu_count, target_bandwidth) → recommended interconnect.
//!
//! Demonstrates the **GPU.29** recipe for PMAT-146 (gpu round 5).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NVIDIA NVLink/NVSwitch architecture whitepapers.
//!
//! Run with: cargo run --example gpu_pcie_p2p_topology
//!
//! Added by PMAT-146 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interconnect {
    PcieGen3,
    PcieGen4,
    PcieGen5,
    NvLink3,
    NvLink4,
    NvSwitch4,
}

#[derive(Debug, PartialEq)]
pub enum TopologyVerdict {
    Ok {
        interconnect: Interconnect,
        peak_bandwidth_gbps: u32,
    },
    InvalidGpuCount,
    InsufficientForTarget {
        max_available: u32,
    },
}

pub fn pick(gpu_count: u32, target_bandwidth_gbps: u32) -> TopologyVerdict {
    if gpu_count < 2 {
        return TopologyVerdict::InvalidGpuCount;
    }
    let candidates = [
        (Interconnect::PcieGen3, 16),
        (Interconnect::PcieGen4, 32),
        (Interconnect::PcieGen5, 64),
        (Interconnect::NvLink3, 150),
        (Interconnect::NvLink4, 225),
        (Interconnect::NvSwitch4, 450),
    ];
    for &(interconnect, gbps) in &candidates {
        if gbps >= target_bandwidth_gbps {
            // NVSwitch4 needs gpu_count >= 4 for full bisection.
            if interconnect == Interconnect::NvSwitch4 && gpu_count < 4 {
                continue;
            }
            return TopologyVerdict::Ok {
                interconnect,
                peak_bandwidth_gbps: gbps,
            };
        }
    }
    TopologyVerdict::InsufficientForTarget { max_available: 450 }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_pcie_p2p_topology")?;

    println!("2 GPUs, 30 Gbps: {:?}", pick(2, 30));
    println!("8 GPUs, 200 Gbps: {:?}", pick(8, 200));
    println!("16 GPUs, 400 Gbps: {:?}", pick(16, 400));
    println!("16 GPUs, 1000 Gbps: {:?}", pick(16, 1000));
    println!("invalid 1 GPU: {:?}", pick(1, 30));
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
    fn two_gpu_low_bw_picks_pcie3() {
        let v = pick(2, 16);
        if let TopologyVerdict::Ok { interconnect, .. } = v {
            assert_eq!(interconnect, Interconnect::PcieGen3);
        }
    }

    #[test]
    fn medium_bw_picks_pcie5() {
        let v = pick(2, 50);
        if let TopologyVerdict::Ok { interconnect, .. } = v {
            assert_eq!(interconnect, Interconnect::PcieGen5);
        }
    }

    #[test]
    fn high_bw_picks_nvlink() {
        let v = pick(2, 100);
        if let TopologyVerdict::Ok { interconnect, .. } = v {
            assert!(matches!(
                interconnect,
                Interconnect::NvLink3 | Interconnect::NvLink4 | Interconnect::NvSwitch4
            ));
        }
    }

    #[test]
    fn very_high_bw_picks_nvswitch4() {
        let v = pick(8, 400);
        if let TopologyVerdict::Ok { interconnect, .. } = v {
            assert_eq!(interconnect, Interconnect::NvSwitch4);
        }
    }

    #[test]
    fn impossible_bw_rejects() {
        let v = pick(8, 1000);
        assert!(matches!(v, TopologyVerdict::InsufficientForTarget { .. }));
    }

    #[test]
    fn invalid_one_gpu_rejected() {
        assert_eq!(pick(1, 50), TopologyVerdict::InvalidGpuCount);
    }

    #[test]
    fn invalid_zero_gpu_rejected() {
        assert_eq!(pick(0, 50), TopologyVerdict::InvalidGpuCount);
    }

    #[test]
    fn nvswitch_requires_4_gpus() {
        // 2 GPUs + 400 Gbps → can't pick NvSwitch4 (needs ≥ 4); rejects.
        let v = pick(2, 400);
        assert!(matches!(v, TopologyVerdict::InsufficientForTarget { .. }));
    }

    #[test]
    fn higher_target_picks_higher_tier() {
        let v_low = pick(8, 30);
        let v_high = pick(8, 200);
        if let (
            TopologyVerdict::Ok {
                peak_bandwidth_gbps: low,
                ..
            },
            TopologyVerdict::Ok {
                peak_bandwidth_gbps: high,
                ..
            },
        ) = (v_low, v_high)
        {
            assert!(high > low);
        }
    }

    #[test]
    fn target_zero_picks_lowest() {
        let v = pick(8, 0);
        if let TopologyVerdict::Ok {
            peak_bandwidth_gbps,
            ..
        } = v
        {
            assert_eq!(peak_bandwidth_gbps, 16);
        }
    }
}
