//! # Training Attention Dropout Rate Picker
//!
//! Attention dropout regularizes training. Picker by model size:
//!   small (< 100M params) → 0.10 (high; small models overfit easily)
//!   medium (100M - 1B) → 0.05
//!   large (1B - 10B) → 0.02
//!   xlarge (≥ 10B) → 0.0 (data dominates regularization)
//!
//! Demonstrates the **TRAIN.19** recipe for PMAT-151 (training round 6).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GPT-3 paper (Brown et al., 2020) hyperparameter table.
//!
//! Run with: cargo run --example training_attention_dropout
//!
//! Added by PMAT-151 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DropoutVerdict {
    Ok { dropout_rate: f64 },
    InvalidModelSize,
}

pub fn pick(parameter_count: u64) -> DropoutVerdict {
    if parameter_count == 0 {
        return DropoutVerdict::InvalidModelSize;
    }
    let dropout_rate = if parameter_count < 100_000_000 {
        0.10
    } else if parameter_count < 1_000_000_000 {
        0.05
    } else if parameter_count < 10_000_000_000 {
        0.02
    } else {
        0.0
    };
    DropoutVerdict::Ok { dropout_rate }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("training_attention_dropout")?;

    println!("50M params: {:?}", pick(50_000_000));
    println!("500M params: {:?}", pick(500_000_000));
    println!("7B params: {:?}", pick(7_000_000_000));
    println!("70B params: {:?}", pick(70_000_000_000));
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
    fn small_high_dropout() {
        let v = pick(50_000_000);
        if let DropoutVerdict::Ok { dropout_rate } = v {
            assert!((dropout_rate - 0.10).abs() < 1e-9);
        }
    }

    #[test]
    fn medium_moderate_dropout() {
        let v = pick(500_000_000);
        if let DropoutVerdict::Ok { dropout_rate } = v {
            assert!((dropout_rate - 0.05).abs() < 1e-9);
        }
    }

    #[test]
    fn large_low_dropout() {
        let v = pick(7_000_000_000);
        if let DropoutVerdict::Ok { dropout_rate } = v {
            assert!((dropout_rate - 0.02).abs() < 1e-9);
        }
    }

    #[test]
    fn xlarge_zero_dropout() {
        let v = pick(70_000_000_000);
        if let DropoutVerdict::Ok { dropout_rate } = v {
            assert!(dropout_rate.abs() < 1e-9);
        }
    }

    #[test]
    fn invalid_zero_params() {
        assert_eq!(pick(0), DropoutVerdict::InvalidModelSize);
    }

    #[test]
    fn dropout_decreases_with_size() {
        let small = pick(50_000_000);
        let med = pick(500_000_000);
        let lg = pick(7_000_000_000);
        let xl = pick(70_000_000_000);
        if let (
            DropoutVerdict::Ok { dropout_rate: s },
            DropoutVerdict::Ok { dropout_rate: m },
            DropoutVerdict::Ok { dropout_rate: l },
            DropoutVerdict::Ok { dropout_rate: x },
        ) = (small, med, lg, xl)
        {
            assert!(s > m);
            assert!(m > l);
            assert!(l > x);
        }
    }

    #[test]
    fn boundary_at_100m() {
        let v = pick(100_000_000);
        if let DropoutVerdict::Ok { dropout_rate } = v {
            assert!((dropout_rate - 0.05).abs() < 1e-9);
        }
    }

    #[test]
    fn boundary_at_1b() {
        let v = pick(1_000_000_000);
        if let DropoutVerdict::Ok { dropout_rate } = v {
            assert!((dropout_rate - 0.02).abs() < 1e-9);
        }
    }

    #[test]
    fn boundary_at_10b() {
        let v = pick(10_000_000_000);
        if let DropoutVerdict::Ok { dropout_rate } = v {
            assert!(dropout_rate.abs() < 1e-9);
        }
    }

    #[test]
    fn dropout_at_most_one() {
        for params in [
            10_000_000u64,
            1_000_000_000,
            10_000_000_000,
            100_000_000_000,
        ] {
            if let DropoutVerdict::Ok { dropout_rate } = pick(params) {
                assert!(dropout_rate <= 1.0);
                assert!(dropout_rate >= 0.0);
            }
        }
    }
}
