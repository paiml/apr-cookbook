//! # Conversion Quantization Calibration Sample Sizer
//!
//! Static quantization needs calibration runs to set scale/zero_point.
//! Sample size:
//!   small model (< 1B): 128 samples
//!   medium (1B-10B): 512 samples
//!   large (10B-100B): 2048 samples
//!   xlarge (≥ 100B): 8192 samples
//!
//! Plus distribution check: input batch must span vocab/feature range.
//!
//! Demonstrates the **CONV.21** recipe for PMAT-153 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: TensorRT post-training quantization calibrator docs.
//!
//! Run with: cargo run --example convert_quant_calibration
//!
//! Added by PMAT-153 (catalog 1000→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CalibVerdict {
    Ok {
        samples: u32,
        diversity_required: bool,
    },
    InvalidParams,
}

pub fn pick(parameter_count: u64) -> CalibVerdict {
    if parameter_count == 0 {
        return CalibVerdict::InvalidParams;
    }
    let samples = if parameter_count < 1_000_000_000 {
        128
    } else if parameter_count < 10_000_000_000 {
        512
    } else if parameter_count < 100_000_000_000 {
        2048
    } else {
        8192
    };
    CalibVerdict::Ok {
        samples,
        diversity_required: parameter_count >= 10_000_000_000,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("convert_quant_calibration")?;

    println!("100M: {:?}", pick(100_000_000));
    println!("7B: {:?}", pick(7_000_000_000));
    println!("70B: {:?}", pick(70_000_000_000));
    println!("400B: {:?}", pick(400_000_000_000));
    println!("invalid: {:?}", pick(0));
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
    fn small_128_samples() {
        let v = pick(100_000_000);
        if let CalibVerdict::Ok { samples, .. } = v {
            assert_eq!(samples, 128);
        }
    }

    #[test]
    fn medium_512_samples() {
        let v = pick(7_000_000_000);
        if let CalibVerdict::Ok { samples, .. } = v {
            assert_eq!(samples, 512);
        }
    }

    #[test]
    fn large_2048_samples() {
        let v = pick(70_000_000_000);
        if let CalibVerdict::Ok { samples, .. } = v {
            assert_eq!(samples, 2048);
        }
    }

    #[test]
    fn xlarge_8192_samples() {
        let v = pick(400_000_000_000);
        if let CalibVerdict::Ok { samples, .. } = v {
            assert_eq!(samples, 8192);
        }
    }

    #[test]
    fn invalid_zero_params() {
        assert_eq!(pick(0), CalibVerdict::InvalidParams);
    }

    #[test]
    fn diversity_for_large_models() {
        let small = pick(100_000_000);
        let large = pick(70_000_000_000);
        if let (
            CalibVerdict::Ok {
                diversity_required: ds,
                ..
            },
            CalibVerdict::Ok {
                diversity_required: dl,
                ..
            },
        ) = (small, large)
        {
            assert!(!ds);
            assert!(dl);
        }
    }

    #[test]
    fn samples_increase_with_size() {
        let v_small = pick(100_000_000);
        let v_med = pick(7_000_000_000);
        let v_lg = pick(70_000_000_000);
        let v_xl = pick(400_000_000_000);
        if let (
            CalibVerdict::Ok { samples: s, .. },
            CalibVerdict::Ok { samples: m, .. },
            CalibVerdict::Ok { samples: l, .. },
            CalibVerdict::Ok { samples: x, .. },
        ) = (v_small, v_med, v_lg, v_xl)
        {
            assert!(s < m);
            assert!(m < l);
            assert!(l < x);
        }
    }

    #[test]
    fn boundary_at_1b_picks_512() {
        let v = pick(1_000_000_000);
        if let CalibVerdict::Ok { samples, .. } = v {
            assert_eq!(samples, 512);
        }
    }

    #[test]
    fn boundary_at_10b_picks_2048() {
        let v = pick(10_000_000_000);
        if let CalibVerdict::Ok { samples, .. } = v {
            assert_eq!(samples, 2048);
        }
    }

    #[test]
    fn boundary_at_100b_picks_8192() {
        let v = pick(100_000_000_000);
        if let CalibVerdict::Ok { samples, .. } = v {
            assert_eq!(samples, 8192);
        }
    }

    #[test]
    fn samples_powers_of_two() {
        for params in [
            100_000_000u64,
            5_000_000_000,
            50_000_000_000,
            500_000_000_000,
        ] {
            if let CalibVerdict::Ok { samples, .. } = pick(params) {
                assert!(samples.is_power_of_two());
            }
        }
    }
}
