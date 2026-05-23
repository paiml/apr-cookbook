//! # apr gpu — Device Capability Filter
//!
//! `apr gpu --json` reports each device's compute capability. This recipe
//! builds the filter for "which devices can run kernel X": e.g.,
//! FlashAttention 3 needs SM 9.0+; BF16 dense kernels need SM 8.0+;
//! INT8 tensor cores need SM 7.5+. The filter classifies the cluster
//! into capable and skipped devices.
//!
//! Demonstrates the **GPU.11** recipe for PMAT-107 (apr gpu coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender GPU-CAPABILITY-001 + NVIDIA SM version table
//!
//! Run with: cargo run --example cli_gpu_device_capability_filter
//!
//! Added by PMAT-107 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDevice {
    pub name: String,
    pub sm_major: u32,
    pub sm_minor: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelClass {
    FlashAttention3, // SM 9.0+ (Hopper+)
    Bf16Dense,       // SM 8.0+ (Ampere+)
    Int8TensorCore,  // SM 7.5+ (Turing+)
    Fp16TensorCore,  // SM 7.0+ (Volta+)
    Generic,         // any GPU
}

impl KernelClass {
    pub fn min_sm(self) -> (u32, u32) {
        match self {
            KernelClass::FlashAttention3 => (9, 0),
            KernelClass::Bf16Dense => (8, 0),
            KernelClass::Int8TensorCore => (7, 5),
            KernelClass::Fp16TensorCore => (7, 0),
            KernelClass::Generic => (0, 0),
        }
    }
}

pub fn supports(device: &GpuDevice, kernel: KernelClass) -> bool {
    let (min_major, min_minor) = kernel.min_sm();
    let dev = device.sm_major * 10 + device.sm_minor;
    let min = min_major * 10 + min_minor;
    dev >= min
}

pub fn filter_devices(
    devices: &[GpuDevice],
    kernel: KernelClass,
) -> (Vec<&GpuDevice>, Vec<&GpuDevice>) {
    let (capable, skipped): (Vec<_>, Vec<_>) = devices.iter().partition(|d| supports(d, kernel));
    (capable, skipped)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_gpu_device_capability_filter")?;

    let cluster = vec![
        GpuDevice {
            name: "H100".into(),
            sm_major: 9,
            sm_minor: 0,
        },
        GpuDevice {
            name: "A100".into(),
            sm_major: 8,
            sm_minor: 0,
        },
        GpuDevice {
            name: "RTX 4090".into(),
            sm_major: 8,
            sm_minor: 9,
        },
        GpuDevice {
            name: "V100".into(),
            sm_major: 7,
            sm_minor: 0,
        },
        GpuDevice {
            name: "K80".into(),
            sm_major: 3,
            sm_minor: 7,
        },
    ];

    for kernel in [
        KernelClass::FlashAttention3,
        KernelClass::Bf16Dense,
        KernelClass::Fp16TensorCore,
        KernelClass::Generic,
    ] {
        let (cap, skip) = filter_devices(&cluster, kernel);
        println!("{kernel:?}: capable={} skipped={}", cap.len(), skip.len());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h100() -> GpuDevice {
        GpuDevice {
            name: "H100".into(),
            sm_major: 9,
            sm_minor: 0,
        }
    }
    fn a100() -> GpuDevice {
        GpuDevice {
            name: "A100".into(),
            sm_major: 8,
            sm_minor: 0,
        }
    }
    fn v100() -> GpuDevice {
        GpuDevice {
            name: "V100".into(),
            sm_major: 7,
            sm_minor: 0,
        }
    }
    fn k80() -> GpuDevice {
        GpuDevice {
            name: "K80".into(),
            sm_major: 3,
            sm_minor: 7,
        }
    }

    #[test]
    fn filter_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn h100_supports_everything() {
        for k in [
            KernelClass::FlashAttention3,
            KernelClass::Bf16Dense,
            KernelClass::Int8TensorCore,
            KernelClass::Fp16TensorCore,
            KernelClass::Generic,
        ] {
            assert!(supports(&h100(), k));
        }
    }

    #[test]
    fn a100_lacks_fa3() {
        // A100 is SM 8.0; FA3 needs SM 9.0+.
        assert!(!supports(&a100(), KernelClass::FlashAttention3));
        // But supports BF16 and below.
        assert!(supports(&a100(), KernelClass::Bf16Dense));
        assert!(supports(&a100(), KernelClass::Fp16TensorCore));
    }

    #[test]
    fn v100_lacks_bf16_dense() {
        // V100 is SM 7.0; BF16 dense needs SM 8.0+.
        assert!(!supports(&v100(), KernelClass::Bf16Dense));
        // But supports FP16.
        assert!(supports(&v100(), KernelClass::Fp16TensorCore));
    }

    #[test]
    fn k80_only_supports_generic() {
        // K80 is SM 3.7 — no tensor cores.
        assert!(!supports(&k80(), KernelClass::Fp16TensorCore));
        assert!(supports(&k80(), KernelClass::Generic));
    }

    #[test]
    fn filter_partitions_cluster() {
        let cluster = vec![h100(), a100(), v100(), k80()];
        let (cap, skip) = filter_devices(&cluster, KernelClass::Bf16Dense);
        // H100 + A100 capable (SM ≥ 8.0); V100 + K80 skipped.
        assert_eq!(cap.len(), 2);
        assert_eq!(skip.len(), 2);
    }

    #[test]
    fn empty_cluster_yields_empty_partitions() {
        let (cap, skip) = filter_devices(&[], KernelClass::Bf16Dense);
        assert!(cap.is_empty());
        assert!(skip.is_empty());
    }

    #[test]
    fn boundary_at_min_sm_passes() {
        // RTX 4090 = SM 8.9 — exactly at FA3's required floor minus 1.
        let rtx4090 = GpuDevice {
            name: "RTX 4090".into(),
            sm_major: 8,
            sm_minor: 9,
        };
        // SM 8.9 < SM 9.0 = no FA3.
        assert!(!supports(&rtx4090, KernelClass::FlashAttention3));
    }
}
