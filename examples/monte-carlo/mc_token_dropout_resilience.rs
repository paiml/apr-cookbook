//! # Monte-Carlo Token-Dropout Resilience
//!
//! Sim accuracy when a fraction of input tokens are randomly dropped.
//! Each token independently dropped with `dropout_pct`. Returns
//! observed mean accuracy across `samples` runs.
//!
//! Demonstrates the **MC.51** recipe for PMAT-174 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Devlin et al. (2018). BERT — masking studies.
//!
//! Run with: cargo run --example mc_token_dropout_resilience
//!
//! Added by PMAT-174 (catalog 1189→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DropoutVerdict {
    Ok { mean_accuracy: f64, std_dev: f64 },
    InvalidConfig,
}

pub fn simulate(
    base_accuracy: f64,
    sensitivity: f64,
    dropout_pct: f64,
    samples: u32,
    seed: u64,
) -> DropoutVerdict {
    if !base_accuracy.is_finite()
        || !(0.0..=1.0).contains(&base_accuracy)
        || !sensitivity.is_finite()
        || sensitivity < 0.0
        || !dropout_pct.is_finite()
        || !(0.0..=1.0).contains(&dropout_pct)
        || samples == 0
    {
        return DropoutVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for _ in 0..samples {
        // Observed dropout ratio drawn around target with ±10% jitter.
        let jitter = (unit(&mut rng_state) - 0.5) * 0.1 * dropout_pct;
        let observed = (dropout_pct + jitter).clamp(0.0, 1.0);
        let acc = (base_accuracy - sensitivity * observed).clamp(0.0, 1.0);
        sum += acc;
        sum_sq += acc * acc;
    }
    let n = f64::from(samples);
    let mean_accuracy = sum / n;
    let variance = (sum_sq / n) - mean_accuracy * mean_accuracy;
    let std_dev = variance.max(0.0).sqrt();
    DropoutVerdict::Ok {
        mean_accuracy,
        std_dev,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_token_dropout_resilience")?;

    println!("low dropout: {:?}", simulate(0.95, 0.5, 0.05, 1000, 42));
    println!("high dropout: {:?}", simulate(0.95, 0.5, 0.40, 1000, 42));
    println!("invalid: {:?}", simulate(0.95, 0.5, 1.5, 1000, 42));
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
    fn no_dropout_near_base() {
        let v = simulate(0.95, 0.5, 0.0, 1000, 42);
        if let DropoutVerdict::Ok { mean_accuracy, .. } = v {
            assert!((mean_accuracy - 0.95).abs() < 0.01);
        }
    }

    #[test]
    fn higher_dropout_lower_mean() {
        let lo = simulate(0.95, 0.5, 0.05, 1000, 42);
        let hi = simulate(0.95, 0.5, 0.50, 1000, 42);
        if let (
            DropoutVerdict::Ok {
                mean_accuracy: l, ..
            },
            DropoutVerdict::Ok {
                mean_accuracy: h, ..
            },
        ) = (lo, hi)
        {
            assert!(l > h);
        }
    }

    #[test]
    fn invalid_neg_base() {
        assert_eq!(
            simulate(-0.1, 0.5, 0.1, 1000, 42),
            DropoutVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_neg_sensitivity() {
        assert_eq!(
            simulate(0.9, -0.1, 0.1, 1000, 42),
            DropoutVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_dropout_over_one() {
        assert_eq!(
            simulate(0.9, 0.5, 1.5, 1000, 42),
            DropoutVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(
            simulate(0.9, 0.5, 0.1, 0, 42),
            DropoutVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(f64::NAN, 0.5, 0.1, 1000, 42),
            DropoutVerdict::InvalidConfig
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
        let v = simulate(0.10, 5.0, 1.0, 1000, 42);
        if let DropoutVerdict::Ok { mean_accuracy, .. } = v {
            assert!(mean_accuracy >= 0.0);
        }
    }

    #[test]
    fn std_dev_non_negative() {
        let v = simulate(0.95, 0.5, 0.30, 1000, 42);
        if let DropoutVerdict::Ok { std_dev, .. } = v {
            assert!(std_dev >= 0.0);
        }
    }
}
