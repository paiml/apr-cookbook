//! # GPU L1 / Shared-Memory Partition Picker
//!
//! On Volta+ (CC ≥ 7.0), L1 cache and shared memory are unified
//! 128 KiB carveout. Picker:
//!   - kernel uses shared mem heavily → "PreferShared"
//!   - kernel is L1-cache-bound (lots of random loads) → "PreferL1"
//!   - balanced → "Equal"
//!
//! For Ampere (CC ≥ 8.0), 164 KiB unified.
//!
//! Demonstrates the **GPU.25** recipe for PMAT-140 (gpu round 3).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: NVIDIA Volta/Ampere architecture whitepapers § L1/Shared.
//!
//! Run with: cargo run --example gpu_l1_shared_mem_partition
//!
//! Added by PMAT-140 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Carveout {
    PreferShared,
    PreferL1,
    Equal,
}

#[derive(Debug, PartialEq)]
pub enum PartitionVerdict {
    Ok {
        carveout: Carveout,
        l1_kib: u32,
        shared_kib: u32,
    },
    UnsupportedComputeCapability {
        major: u8,
        minor: u8,
    },
}

pub fn pick(
    cc_major: u8,
    cc_minor: u8,
    shared_mem_per_block_kib: u32,
    l1_intensity: f64,
) -> PartitionVerdict {
    if cc_major < 7 {
        return PartitionVerdict::UnsupportedComputeCapability {
            major: cc_major,
            minor: cc_minor,
        };
    }
    let total_kib = if cc_major >= 8 { 164 } else { 128 };
    let carveout = if shared_mem_per_block_kib >= total_kib / 2 {
        Carveout::PreferShared
    } else if l1_intensity > 0.6 {
        Carveout::PreferL1
    } else {
        Carveout::Equal
    };
    let (l1_kib, shared_kib) = match carveout {
        Carveout::PreferShared => (total_kib / 4, total_kib * 3 / 4),
        Carveout::PreferL1 => (total_kib * 3 / 4, total_kib / 4),
        Carveout::Equal => (total_kib / 2, total_kib / 2),
    };
    PartitionVerdict::Ok {
        carveout,
        l1_kib,
        shared_kib,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_l1_shared_mem_partition")?;

    println!("Volta heavy shared: {:?}", pick(7, 0, 100, 0.2));
    println!("Ampere heavy L1: {:?}", pick(8, 0, 8, 0.8));
    println!("Hopper balanced: {:?}", pick(9, 0, 32, 0.4));
    println!("Pre-Volta unsupported: {:?}", pick(6, 1, 16, 0.5));
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
    fn pre_volta_unsupported() {
        let v = pick(6, 1, 16, 0.5);
        assert!(matches!(
            v,
            PartitionVerdict::UnsupportedComputeCapability { .. }
        ));
    }

    #[test]
    fn volta_heavy_shared_picks_prefer_shared() {
        // Block uses 100 KiB of shared (≥ 64 = 128/2).
        let v = pick(7, 0, 100, 0.2);
        if let PartitionVerdict::Ok { carveout, .. } = v {
            assert_eq!(carveout, Carveout::PreferShared);
        }
    }

    #[test]
    fn high_l1_intensity_picks_prefer_l1() {
        let v = pick(8, 0, 8, 0.8);
        if let PartitionVerdict::Ok { carveout, .. } = v {
            assert_eq!(carveout, Carveout::PreferL1);
        }
    }

    #[test]
    fn balanced_picks_equal() {
        let v = pick(7, 0, 16, 0.4);
        if let PartitionVerdict::Ok { carveout, .. } = v {
            assert_eq!(carveout, Carveout::Equal);
        }
    }

    #[test]
    fn ampere_total_164() {
        // Ampere has 164 KiB total, equal partition = 82 each.
        let v = pick(8, 0, 16, 0.4);
        if let PartitionVerdict::Ok {
            l1_kib, shared_kib, ..
        } = v
        {
            assert_eq!(l1_kib + shared_kib, 164);
            assert_eq!(l1_kib, 82);
        }
    }

    #[test]
    fn volta_total_128() {
        let v = pick(7, 0, 16, 0.4);
        if let PartitionVerdict::Ok {
            l1_kib, shared_kib, ..
        } = v
        {
            assert_eq!(l1_kib + shared_kib, 128);
        }
    }

    #[test]
    fn prefer_shared_assigns_more_to_shared() {
        let v = pick(7, 0, 100, 0.2);
        if let PartitionVerdict::Ok {
            l1_kib, shared_kib, ..
        } = v
        {
            assert!(shared_kib > l1_kib);
        }
    }

    #[test]
    fn prefer_l1_assigns_more_to_l1() {
        let v = pick(7, 0, 8, 0.8);
        if let PartitionVerdict::Ok {
            l1_kib, shared_kib, ..
        } = v
        {
            assert!(l1_kib > shared_kib);
        }
    }

    #[test]
    fn boundary_exactly_half_picks_shared() {
        // shared_mem_per_block = total/2 → still triggers PreferShared.
        let v = pick(7, 0, 64, 0.5);
        if let PartitionVerdict::Ok { carveout, .. } = v {
            assert_eq!(carveout, Carveout::PreferShared);
        }
    }

    #[test]
    fn cc_5_unsupported() {
        let v = pick(5, 2, 16, 0.5);
        assert!(matches!(
            v,
            PartitionVerdict::UnsupportedComputeCapability { major: 5, .. }
        ));
    }
}
