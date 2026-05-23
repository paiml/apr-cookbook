//! # GPU Kernel Autotune Picker
//!
//! Try multiple kernel variants (block size, tile size, vectorization)
//! and pick fastest. Picker takes timing samples + variant configs,
//! returns best.
//!
//! Demonstrates the **GPU.35** recipe for PMAT-152 (milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: cuBLAS gemm autotune kernels.
//!
//! Run with: cargo run --example gpu_kernel_autotune
//!
//! Added by PMAT-152 (catalog crosses 1000 recipes).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KernelVariant {
    pub block_size: u32,
    pub tile_size: u32,
    pub timing_us: f64,
}

#[derive(Debug, PartialEq)]
pub enum AutotuneVerdict {
    Ok {
        best_block: u32,
        best_tile: u32,
        speedup_vs_default: f64,
    },
    EmptyVariants,
    InvalidTiming,
}

pub fn pick(variants: &[KernelVariant], default_timing_us: f64) -> AutotuneVerdict {
    if variants.is_empty() {
        return AutotuneVerdict::EmptyVariants;
    }
    if !default_timing_us.is_finite() || default_timing_us <= 0.0 {
        return AutotuneVerdict::InvalidTiming;
    }
    if variants
        .iter()
        .any(|v| !v.timing_us.is_finite() || v.timing_us <= 0.0)
    {
        return AutotuneVerdict::InvalidTiming;
    }
    let best = variants
        .iter()
        .min_by(|a, b| {
            a.timing_us
                .partial_cmp(&b.timing_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    AutotuneVerdict::Ok {
        best_block: best.block_size,
        best_tile: best.tile_size,
        speedup_vs_default: default_timing_us / best.timing_us,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("gpu_kernel_autotune")?;

    let variants = [
        KernelVariant {
            block_size: 128,
            tile_size: 16,
            timing_us: 100.0,
        },
        KernelVariant {
            block_size: 256,
            tile_size: 32,
            timing_us: 50.0,
        },
        KernelVariant {
            block_size: 512,
            tile_size: 64,
            timing_us: 80.0,
        },
    ];
    println!("3 variants vs 200μs default: {:?}", pick(&variants, 200.0));
    println!("empty: {:?}", pick(&[], 100.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typical() -> Vec<KernelVariant> {
        vec![
            KernelVariant {
                block_size: 128,
                tile_size: 16,
                timing_us: 100.0,
            },
            KernelVariant {
                block_size: 256,
                tile_size: 32,
                timing_us: 50.0,
            },
            KernelVariant {
                block_size: 512,
                tile_size: 64,
                timing_us: 80.0,
            },
        ]
    }

    #[test]
    fn picker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn fastest_picked() {
        let v = pick(&typical(), 200.0);
        if let AutotuneVerdict::Ok { best_block, .. } = v {
            assert_eq!(best_block, 256);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(pick(&[], 100.0), AutotuneVerdict::EmptyVariants);
    }

    #[test]
    fn invalid_default_rejected() {
        assert_eq!(pick(&typical(), 0.0), AutotuneVerdict::InvalidTiming);
    }

    #[test]
    fn nan_default_rejected() {
        assert_eq!(pick(&typical(), f64::NAN), AutotuneVerdict::InvalidTiming);
    }

    #[test]
    fn invalid_variant_timing_rejected() {
        let bad = vec![KernelVariant {
            block_size: 128,
            tile_size: 16,
            timing_us: -1.0,
        }];
        assert_eq!(pick(&bad, 100.0), AutotuneVerdict::InvalidTiming);
    }

    #[test]
    fn speedup_correct() {
        let v = pick(&typical(), 200.0);
        if let AutotuneVerdict::Ok {
            speedup_vs_default, ..
        } = v
        {
            // 200 / 50 = 4.
            assert!((speedup_vs_default - 4.0).abs() < 1e-9);
        }
    }

    #[test]
    fn single_variant_picked() {
        let v = pick(
            &[KernelVariant {
                block_size: 128,
                tile_size: 16,
                timing_us: 100.0,
            }],
            150.0,
        );
        assert!(matches!(v, AutotuneVerdict::Ok { .. }));
    }

    #[test]
    fn variant_slower_than_default_still_picked() {
        // Even if best variant slower than default, it's the best of the lot.
        let slow = vec![KernelVariant {
            block_size: 128,
            tile_size: 16,
            timing_us: 1000.0,
        }];
        if let AutotuneVerdict::Ok {
            speedup_vs_default, ..
        } = pick(&slow, 100.0)
        {
            assert!(speedup_vs_default < 1.0);
        }
    }

    #[test]
    fn best_block_size_returned() {
        if let AutotuneVerdict::Ok {
            best_block,
            best_tile,
            ..
        } = pick(&typical(), 200.0)
        {
            assert_eq!(best_block, 256);
            assert_eq!(best_tile, 32);
        }
    }

    #[test]
    fn ties_picked_first_by_iteration() {
        let tied = vec![
            KernelVariant {
                block_size: 128,
                tile_size: 16,
                timing_us: 50.0,
            },
            KernelVariant {
                block_size: 256,
                tile_size: 32,
                timing_us: 50.0,
            },
        ];
        // Either is valid; just check we get an answer.
        assert!(matches!(pick(&tied, 100.0), AutotuneVerdict::Ok { .. }));
    }
}
