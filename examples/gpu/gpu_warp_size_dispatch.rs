//! # GPU Warp-Size Runtime Dispatcher
//!
//! NVIDIA GPUs have 32-thread warps; AMD GPUs have 64-thread "wave64"
//! (or 32-thread "wave32" on RDNA3+). Kernel block size must be
//! warp-size aligned for full occupancy.
//!
//! Picker maps (vendor, generation) → warp size + recommended block.
//!
//! Demonstrates the **GPU.34** recipe for PMAT-150 (gpu round 6).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: AMD CDNA/RDNA wavefront semantics + NVIDIA warp docs.
//!
//! Run with: cargo run --example gpu_warp_size_dispatch
//!
//! Added by PMAT-150 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vendor {
    Nvidia,
    AmdGcn,
    AmdRdna1,
    AmdRdna3,
}

#[derive(Debug, PartialEq)]
pub enum DispatchVerdict {
    Ok {
        warp_size: u32,
        recommended_block_size: u32,
    },
    InvalidVendor,
}

pub fn pick(vendor: Vendor) -> DispatchVerdict {
    let warp_size = match vendor {
        Vendor::Nvidia | Vendor::AmdRdna3 => 32,
        Vendor::AmdGcn | Vendor::AmdRdna1 => 64,
    };
    // Recommended block size = 8 warps (typical sweet spot).
    let recommended_block_size = warp_size * 8;
    DispatchVerdict::Ok {
        warp_size,
        recommended_block_size,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_warp_size_dispatch")?;

    for v in [
        Vendor::Nvidia,
        Vendor::AmdGcn,
        Vendor::AmdRdna1,
        Vendor::AmdRdna3,
    ] {
        println!("{v:?}: {:?}", pick(v));
    }
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
    fn nvidia_warp_32() {
        let v = pick(Vendor::Nvidia);
        if let DispatchVerdict::Ok { warp_size, .. } = v {
            assert_eq!(warp_size, 32);
        }
    }

    #[test]
    fn amd_gcn_wave64() {
        let v = pick(Vendor::AmdGcn);
        if let DispatchVerdict::Ok { warp_size, .. } = v {
            assert_eq!(warp_size, 64);
        }
    }

    #[test]
    fn amd_rdna1_wave64() {
        // RDNA1 still defaults to wave64 in compute.
        let v = pick(Vendor::AmdRdna1);
        if let DispatchVerdict::Ok { warp_size, .. } = v {
            assert_eq!(warp_size, 64);
        }
    }

    #[test]
    fn amd_rdna3_wave32() {
        // RDNA3 introduces wave32 by default for compute.
        let v = pick(Vendor::AmdRdna3);
        if let DispatchVerdict::Ok { warp_size, .. } = v {
            assert_eq!(warp_size, 32);
        }
    }

    #[test]
    fn block_size_eight_warps() {
        let v = pick(Vendor::Nvidia);
        if let DispatchVerdict::Ok {
            warp_size,
            recommended_block_size,
            ..
        } = v
        {
            assert_eq!(recommended_block_size, warp_size * 8);
        }
    }

    #[test]
    fn block_size_at_least_warp_size() {
        for vendor in [
            Vendor::Nvidia,
            Vendor::AmdGcn,
            Vendor::AmdRdna1,
            Vendor::AmdRdna3,
        ] {
            if let DispatchVerdict::Ok {
                warp_size,
                recommended_block_size,
                ..
            } = pick(vendor)
            {
                assert!(recommended_block_size >= warp_size);
            }
        }
    }

    #[test]
    fn block_size_multiple_of_warp() {
        for vendor in [
            Vendor::Nvidia,
            Vendor::AmdGcn,
            Vendor::AmdRdna1,
            Vendor::AmdRdna3,
        ] {
            if let DispatchVerdict::Ok {
                warp_size,
                recommended_block_size,
                ..
            } = pick(vendor)
            {
                assert_eq!(recommended_block_size % warp_size, 0);
            }
        }
    }

    #[test]
    fn block_size_within_max_1024() {
        for vendor in [
            Vendor::Nvidia,
            Vendor::AmdGcn,
            Vendor::AmdRdna1,
            Vendor::AmdRdna3,
        ] {
            if let DispatchVerdict::Ok {
                recommended_block_size,
                ..
            } = pick(vendor)
            {
                assert!(recommended_block_size <= 1024);
            }
        }
    }

    #[test]
    fn nvidia_block_256() {
        if let DispatchVerdict::Ok {
            recommended_block_size,
            ..
        } = pick(Vendor::Nvidia)
        {
            assert_eq!(recommended_block_size, 256);
        }
    }

    #[test]
    fn amd_gcn_block_512() {
        if let DispatchVerdict::Ok {
            recommended_block_size,
            ..
        } = pick(Vendor::AmdGcn)
        {
            assert_eq!(recommended_block_size, 512);
        }
    }
}
