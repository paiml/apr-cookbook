//! # Training Warmup Steps Picker
//!
//! Optimizer LR warmup ramps from 0 to peak. Picker by model size:
//!   small (< 100M params) → 500 steps (rapid stabilization)
//:   medium (100M - 1B) → 1000 steps
//!   large (1B - 10B) → 2000 steps
//!   xlarge (≥ 10B) → 5000 steps (slow convergence)
//!
//! Demonstrates the **TRAIN.21** recipe for PMAT-151 (training round 6).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GoyaIyer et al (2017). Linear warmup + 1cycle policy.
//!
//! Run with: cargo run --example training_warmup_steps
//!
//! Added by PMAT-151 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WarmupVerdict {
    Ok {
        warmup_steps: u32,
        warmup_pct_of_train: f64,
    },
    InvalidParams,
    InvalidTotalSteps,
}

pub fn pick(parameter_count: u64, total_train_steps: u32) -> WarmupVerdict {
    if parameter_count == 0 {
        return WarmupVerdict::InvalidParams;
    }
    if total_train_steps == 0 {
        return WarmupVerdict::InvalidTotalSteps;
    }
    let warmup_steps = if parameter_count < 100_000_000 {
        500u32
    } else if parameter_count < 1_000_000_000 {
        1000
    } else if parameter_count < 10_000_000_000 {
        2000
    } else {
        5000
    };
    let warmup_steps = warmup_steps.min(total_train_steps);
    let warmup_pct_of_train = f64::from(warmup_steps) / f64::from(total_train_steps) * 100.0;
    WarmupVerdict::Ok {
        warmup_steps,
        warmup_pct_of_train,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("training_warmup_steps")?;

    println!("50M, 100k steps: {:?}", pick(50_000_000, 100_000));
    println!("7B, 200k steps: {:?}", pick(7_000_000_000, 200_000));
    println!("70B, 500k steps: {:?}", pick(70_000_000_000, 500_000));
    println!("invalid: {:?}", pick(0, 100_000));
    println!("zero total: {:?}", pick(50_000_000, 0));
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
    fn small_500_warmup() {
        let v = pick(50_000_000, 100_000);
        if let WarmupVerdict::Ok { warmup_steps, .. } = v {
            assert_eq!(warmup_steps, 500);
        }
    }

    #[test]
    fn medium_1000_warmup() {
        let v = pick(500_000_000, 100_000);
        if let WarmupVerdict::Ok { warmup_steps, .. } = v {
            assert_eq!(warmup_steps, 1000);
        }
    }

    #[test]
    fn large_2000_warmup() {
        let v = pick(7_000_000_000, 100_000);
        if let WarmupVerdict::Ok { warmup_steps, .. } = v {
            assert_eq!(warmup_steps, 2000);
        }
    }

    #[test]
    fn xlarge_5000_warmup() {
        let v = pick(70_000_000_000, 500_000);
        if let WarmupVerdict::Ok { warmup_steps, .. } = v {
            assert_eq!(warmup_steps, 5000);
        }
    }

    #[test]
    fn invalid_zero_params() {
        assert_eq!(pick(0, 100_000), WarmupVerdict::InvalidParams);
    }

    #[test]
    fn invalid_zero_steps() {
        assert_eq!(pick(50_000_000, 0), WarmupVerdict::InvalidTotalSteps);
    }

    #[test]
    fn warmup_cap_at_total_steps() {
        // Tiny train run: 100 steps total. Even xlarge model warmup → 100.
        let v = pick(70_000_000_000, 100);
        if let WarmupVerdict::Ok { warmup_steps, .. } = v {
            assert_eq!(warmup_steps, 100);
        }
    }

    #[test]
    fn warmup_pct_correct() {
        // 1000 / 100k = 1%.
        let v = pick(500_000_000, 100_000);
        if let WarmupVerdict::Ok {
            warmup_pct_of_train,
            ..
        } = v
        {
            assert!((warmup_pct_of_train - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn larger_model_more_warmup() {
        let small = pick(50_000_000, 100_000);
        let xl = pick(70_000_000_000, 100_000);
        if let (
            WarmupVerdict::Ok {
                warmup_steps: s, ..
            },
            WarmupVerdict::Ok {
                warmup_steps: x, ..
            },
        ) = (small, xl)
        {
            assert!(x > s);
        }
    }

    #[test]
    fn boundary_at_100m_picks_1000() {
        let v = pick(100_000_000, 100_000);
        if let WarmupVerdict::Ok { warmup_steps, .. } = v {
            assert_eq!(warmup_steps, 1000);
        }
    }

    #[test]
    fn boundary_at_10b_picks_5000() {
        let v = pick(10_000_000_000, 100_000);
        if let WarmupVerdict::Ok { warmup_steps, .. } = v {
            assert_eq!(warmup_steps, 5000);
        }
    }
}
