//! # Training Optimizer-State Memory Estimator
//!
//! Optimizer state size relative to model weights:
//!   SGD:        0 (no state) or 1× (with momentum)
//!   AdamW:      2× (m + v moments)
//!   Lion:       1× (single momentum)
//!
//! Plus mixed-precision: master weights add 1× in fp32 if model in
//! fp16/bf16. Total memory = weights + grads + optimizer_state +
//! optional master.
//!
//! Demonstrates the **TRAIN.14** recipe for PMAT-144 (training round 4).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: ZeRO paper (Rajbhandari et al., 2019) optimizer-state breakdown.
//!
//! Run with: cargo run --example training_optimizer_state_memory
//!
//! Added by PMAT-144 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Optimizer {
    SgdNoMomentum,
    SgdWithMomentum,
    AdamW,
    Lion,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Fp32,
    Fp16,
    Bf16,
    Mixed,
}

#[derive(Debug, PartialEq)]
pub enum MemoryVerdict {
    Ok {
        weights_gib: f64,
        grads_gib: f64,
        optimizer_gib: f64,
        master_gib: f64,
        total_gib: f64,
    },
    InvalidParameterCount,
}

pub fn estimate(parameter_count: u64, optimizer: Optimizer, precision: Precision) -> MemoryVerdict {
    if parameter_count == 0 {
        return MemoryVerdict::InvalidParameterCount;
    }
    let bytes_per_param = match precision {
        Precision::Fp32 => 4.0,
        Precision::Fp16 | Precision::Bf16 | Precision::Mixed => 2.0,
    };
    let weights_gib = (parameter_count as f64 * bytes_per_param) / 1_073_741_824.0;
    let grads_gib = weights_gib;
    let optimizer_factor = match optimizer {
        Optimizer::SgdNoMomentum => 0.0,
        Optimizer::SgdWithMomentum | Optimizer::Lion => 1.0,
        Optimizer::AdamW => 2.0,
    };
    let optimizer_gib = weights_gib * optimizer_factor;
    let master_gib = if matches!(precision, Precision::Mixed) {
        // Master copy in fp32 = 4 bytes × params / 1 GiB.
        (parameter_count as f64 * 4.0) / 1_073_741_824.0
    } else {
        0.0
    };
    let total_gib = weights_gib + grads_gib + optimizer_gib + master_gib;
    MemoryVerdict::Ok {
        weights_gib,
        grads_gib,
        optimizer_gib,
        master_gib,
        total_gib,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("training_optimizer_state_memory")?;

    // 7B params in fp16 + AdamW + mixed precision.
    println!(
        "7B AdamW mixed: {:?}",
        estimate(7_000_000_000, Optimizer::AdamW, Precision::Mixed)
    );

    // 1B params in fp32 + SGD.
    println!(
        "1B SGD fp32: {:?}",
        estimate(1_000_000_000, Optimizer::SgdNoMomentum, Precision::Fp32)
    );

    println!(
        "invalid: {:?}",
        estimate(0, Optimizer::AdamW, Precision::Fp32)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn sgd_no_momentum_no_optimizer_state() {
        let v = estimate(1_000_000_000, Optimizer::SgdNoMomentum, Precision::Fp32);
        if let MemoryVerdict::Ok { optimizer_gib, .. } = v {
            assert!(optimizer_gib.abs() < 1e-9);
        }
    }

    #[test]
    fn adamw_two_x_weights() {
        let v = estimate(1_000_000_000, Optimizer::AdamW, Precision::Fp32);
        if let MemoryVerdict::Ok {
            weights_gib,
            optimizer_gib,
            ..
        } = v
        {
            assert!((optimizer_gib - 2.0 * weights_gib).abs() < 1e-9);
        }
    }

    #[test]
    fn lion_one_x_weights() {
        let v = estimate(1_000_000_000, Optimizer::Lion, Precision::Fp32);
        if let MemoryVerdict::Ok {
            weights_gib,
            optimizer_gib,
            ..
        } = v
        {
            assert!((optimizer_gib - weights_gib).abs() < 1e-9);
        }
    }

    #[test]
    fn fp16_half_size_of_fp32() {
        let v_fp32 = estimate(1_000_000_000, Optimizer::AdamW, Precision::Fp32);
        let v_fp16 = estimate(1_000_000_000, Optimizer::AdamW, Precision::Fp16);
        if let (
            MemoryVerdict::Ok {
                weights_gib: f32_w, ..
            },
            MemoryVerdict::Ok {
                weights_gib: f16_w, ..
            },
        ) = (v_fp32, v_fp16)
        {
            assert!((f32_w / f16_w - 2.0).abs() < 1e-6);
        }
    }

    #[test]
    fn mixed_precision_adds_master() {
        let v_fp16 = estimate(1_000_000_000, Optimizer::AdamW, Precision::Fp16);
        let v_mixed = estimate(1_000_000_000, Optimizer::AdamW, Precision::Mixed);
        if let (
            MemoryVerdict::Ok {
                master_gib: f16_m, ..
            },
            MemoryVerdict::Ok {
                master_gib: mixed_m,
                ..
            },
        ) = (v_fp16, v_mixed)
        {
            assert!(f16_m.abs() < 1e-9);
            assert!(mixed_m > 3.0); // 4 bytes × 1B / 1 GiB ≈ 3.7
        }
    }

    #[test]
    fn invalid_zero_params_rejected() {
        assert_eq!(
            estimate(0, Optimizer::AdamW, Precision::Fp32),
            MemoryVerdict::InvalidParameterCount
        );
    }

    #[test]
    fn total_includes_all_components() {
        if let MemoryVerdict::Ok {
            weights_gib,
            grads_gib,
            optimizer_gib,
            master_gib,
            total_gib,
        } = estimate(1_000_000_000, Optimizer::AdamW, Precision::Mixed)
        {
            let sum = weights_gib + grads_gib + optimizer_gib + master_gib;
            assert!((total_gib - sum).abs() < 1e-6);
        }
    }

    #[test]
    fn larger_model_more_memory() {
        let v_small = estimate(1_000_000_000, Optimizer::AdamW, Precision::Fp32);
        let v_large = estimate(7_000_000_000, Optimizer::AdamW, Precision::Fp32);
        if let (
            MemoryVerdict::Ok {
                total_gib: small, ..
            },
            MemoryVerdict::Ok {
                total_gib: large, ..
            },
        ) = (v_small, v_large)
        {
            assert!(large > small * 6.0);
        }
    }

    #[test]
    fn sgd_with_momentum_one_x() {
        let v = estimate(1_000_000_000, Optimizer::SgdWithMomentum, Precision::Fp32);
        if let MemoryVerdict::Ok {
            weights_gib,
            optimizer_gib,
            ..
        } = v
        {
            assert!((optimizer_gib - weights_gib).abs() < 1e-9);
        }
    }

    #[test]
    fn grads_match_weights_size() {
        if let MemoryVerdict::Ok {
            weights_gib,
            grads_gib,
            ..
        } = estimate(1_000_000_000, Optimizer::AdamW, Precision::Fp32)
        {
            assert!((weights_gib - grads_gib).abs() < 1e-9);
        }
    }
}
