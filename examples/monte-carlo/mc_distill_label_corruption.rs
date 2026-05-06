//! # Monte-Carlo Label-Corruption Sensitivity in Distillation
//!
//! Sim distillation accuracy as a function of teacher label noise:
//! `corruption_pct` of teacher labels are flipped randomly. Returns
//! mean student accuracy and decay rate.
//!
//! Demonstrates the **MC.59** recipe for PMAT-177 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: noisy-label distillation studies (Patrini et al. 2017).
//!
//! Run with: cargo run --example mc_distill_label_corruption
//!
//! Added by PMAT-177 (catalog 1216→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum CorruptionVerdict {
    Ok {
        mean_accuracy: f64,
        decay_per_pct_corruption: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    base_accuracy: f64,
    sensitivity: f64,
    corruption_pct: f64,
    samples: u32,
    seed: u64,
) -> CorruptionVerdict {
    if !base_accuracy.is_finite()
        || !(0.0..=1.0).contains(&base_accuracy)
        || !sensitivity.is_finite()
        || sensitivity < 0.0
        || !corruption_pct.is_finite()
        || !(0.0..=1.0).contains(&corruption_pct)
        || samples == 0
    {
        return CorruptionVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut sum = 0.0;
    for _ in 0..samples {
        let noise = (unit(&mut rng_state) - 0.5) * 0.1;
        let observed_corruption = (corruption_pct + noise).clamp(0.0, 1.0);
        let acc = (base_accuracy - sensitivity * observed_corruption).clamp(0.0, 1.0);
        sum += acc;
    }
    let mean_accuracy = sum / f64::from(samples);
    let decay_per_pct_corruption = if corruption_pct > 0.0 {
        (base_accuracy - mean_accuracy) / corruption_pct
    } else {
        0.0
    };
    CorruptionVerdict::Ok {
        mean_accuracy,
        decay_per_pct_corruption,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_distill_label_corruption")?;

    println!("clean: {:?}", simulate(0.95, 0.5, 0.0, 1000, 42));
    println!("light: {:?}", simulate(0.95, 0.5, 0.05, 1000, 42));
    println!("heavy: {:?}", simulate(0.95, 0.5, 0.30, 1000, 42));
    println!("invalid: {:?}", simulate(2.0, 0.5, 0.05, 1000, 42));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn no_corruption_near_base() {
        let v = simulate(0.95, 0.5, 0.0, 1000, 42);
        if let CorruptionVerdict::Ok { mean_accuracy, .. } = v {
            assert!((mean_accuracy - 0.95).abs() < 0.01);
        }
    }

    #[test]
    fn higher_corruption_lower_accuracy() {
        let lo = simulate(0.95, 0.5, 0.05, 1000, 42);
        let hi = simulate(0.95, 0.5, 0.40, 1000, 42);
        if let (
            CorruptionVerdict::Ok {
                mean_accuracy: l, ..
            },
            CorruptionVerdict::Ok {
                mean_accuracy: h, ..
            },
        ) = (lo, hi)
        {
            assert!(l > h);
        }
    }

    #[test]
    fn invalid_acc_over_one() {
        assert_eq!(
            simulate(2.0, 0.5, 0.1, 1000, 42),
            CorruptionVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_neg_sensitivity() {
        assert_eq!(
            simulate(0.95, -0.1, 0.1, 1000, 42),
            CorruptionVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_corruption_over_one() {
        assert_eq!(
            simulate(0.95, 0.5, 1.5, 1000, 42),
            CorruptionVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(
            simulate(0.95, 0.5, 0.1, 0, 42),
            CorruptionVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(f64::NAN, 0.5, 0.1, 1000, 42),
            CorruptionVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(0.95, 0.5, 0.1, 1000, 42);
        let b = simulate(0.95, 0.5, 0.1, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn accuracy_clamped() {
        let v = simulate(0.10, 5.0, 0.9, 1000, 42);
        if let CorruptionVerdict::Ok { mean_accuracy, .. } = v {
            assert!(mean_accuracy >= 0.0);
        }
    }

    #[test]
    fn zero_corruption_zero_decay() {
        let v = simulate(0.95, 0.5, 0.0, 1000, 42);
        if let CorruptionVerdict::Ok {
            decay_per_pct_corruption,
            ..
        } = v
        {
            assert!((decay_per_pct_corruption - 0.0).abs() < 1e-9);
        }
    }
}
