//! # Monte-Carlo Dropout Resilience
//!
//! Simulate accuracy variance under random feature dropout. Each
//! sample drops K of N features with probability p; observed accuracy
//! degrades as a function of dropout rate.
//!
//! Demonstrates the **MC.22** recipe for PMAT-165 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Srivastava et al. (2014). Dropout regularization.
//!
//! Run with: cargo run --example mc_dropout_resilience
//!
//! Added by PMAT-165 (catalog 1108→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DropoutVerdict {
    Ok {
        mean_accuracy: f64,
        min_accuracy: f64,
        max_accuracy: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    base_accuracy: f64,
    sensitivity: f64,
    dropout_rate: f64,
    runs: u32,
    seed: u64,
) -> DropoutVerdict {
    if !base_accuracy.is_finite()
        || !(0.0..=1.0).contains(&base_accuracy)
        || !sensitivity.is_finite()
        || sensitivity < 0.0
        || !dropout_rate.is_finite()
        || !(0.0..=1.0).contains(&dropout_rate)
        || runs == 0
    {
        return DropoutVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut sum = 0.0;
    let mut min_acc = f64::INFINITY;
    let mut max_acc = f64::NEG_INFINITY;
    for _ in 0..runs {
        // Draw a binomial-ish dropout count via uniform sampling.
        let observed_rate = unit(&mut rng_state) * dropout_rate * 2.0;
        let acc = (base_accuracy - sensitivity * observed_rate).clamp(0.0, 1.0);
        sum += acc;
        if acc < min_acc {
            min_acc = acc;
        }
        if acc > max_acc {
            max_acc = acc;
        }
    }
    let mean_accuracy = sum / f64::from(runs);
    DropoutVerdict::Ok {
        mean_accuracy,
        min_accuracy: min_acc,
        max_accuracy: max_acc,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_dropout_resilience")?;

    println!("low dropout: {:?}", simulate(0.95, 0.5, 0.05, 1000, 42));
    println!("high dropout: {:?}", simulate(0.95, 0.5, 0.30, 1000, 42));
    println!("invalid: {:?}", simulate(2.0, 0.5, 0.1, 1000, 42));
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
        let v = simulate(0.95, 0.5, 0.0, 100, 42);
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
    fn invalid_base_over_one() {
        assert_eq!(
            simulate(2.0, 0.5, 0.1, 100, 42),
            DropoutVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_negative_sensitivity() {
        assert_eq!(
            simulate(0.9, -0.5, 0.1, 100, 42),
            DropoutVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_dropout_rate_negative() {
        assert_eq!(
            simulate(0.9, 0.5, -0.1, 100, 42),
            DropoutVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_runs() {
        assert_eq!(
            simulate(0.9, 0.5, 0.1, 0, 42),
            DropoutVerdict::InvalidConfig
        );
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(f64::NAN, 0.5, 0.1, 100, 42),
            DropoutVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(0.95, 0.5, 0.1, 100, 42);
        let b = simulate(0.95, 0.5, 0.1, 100, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn min_below_max() {
        let v = simulate(0.95, 0.5, 0.30, 1000, 42);
        if let DropoutVerdict::Ok {
            min_accuracy,
            max_accuracy,
            ..
        } = v
        {
            assert!(min_accuracy <= max_accuracy);
        }
    }

    #[test]
    fn accuracy_clamped_in_range() {
        let v = simulate(0.10, 5.0, 1.0, 100, 42);
        if let DropoutVerdict::Ok {
            min_accuracy,
            max_accuracy,
            ..
        } = v
        {
            assert!(min_accuracy >= 0.0);
            assert!(max_accuracy <= 1.0);
        }
    }
}
