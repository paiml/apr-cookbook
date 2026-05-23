//! # Monte-Carlo Warmup-to-Target
//!
//! Sim incremental learning: starting accuracy `acc0`, gain per
//! sample shrinks as accuracy approaches the target (asymptotic).
//! Returns expected number of samples to reach `target_accuracy`.
//!
//! Demonstrates the **MC.41** recipe for PMAT-171 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: PAC learning sample-complexity bounds.
//!
//! Run with: cargo run --example mc_warmup_to_target
//!
//! Added by PMAT-171 (catalog 1162→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WarmupVerdict {
    Ok {
        samples_needed: u32,
        final_accuracy: f64,
    },
    NeverReaches {
        final_accuracy: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    start_accuracy: f64,
    target_accuracy: f64,
    learning_rate: f64,
    max_samples: u32,
) -> WarmupVerdict {
    if !start_accuracy.is_finite()
        || !target_accuracy.is_finite()
        || !learning_rate.is_finite()
        || !(0.0..=1.0).contains(&start_accuracy)
        || !(0.0..=1.0).contains(&target_accuracy)
        || target_accuracy <= start_accuracy
        || learning_rate <= 0.0
        || max_samples == 0
    {
        return WarmupVerdict::InvalidConfig;
    }
    let mut accuracy = start_accuracy;
    for step in 0..max_samples {
        // Asymptotic learning curve: gain shrinks as we approach 1.0.
        let gain = learning_rate * (1.0 - accuracy);
        accuracy = (accuracy + gain).min(1.0);
        if accuracy >= target_accuracy {
            return WarmupVerdict::Ok {
                samples_needed: step + 1,
                final_accuracy: accuracy,
            };
        }
    }
    WarmupVerdict::NeverReaches {
        final_accuracy: accuracy,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_warmup_to_target")?;

    println!("fast: {:?}", simulate(0.5, 0.90, 0.1, 1000));
    println!("slow: {:?}", simulate(0.5, 0.99, 0.001, 1000));
    println!("never: {:?}", simulate(0.5, 0.99, 0.0001, 100));
    println!("invalid: {:?}", simulate(0.5, 0.5, 0.1, 100));
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
    fn fast_learning_quick_target() {
        let v = simulate(0.5, 0.90, 0.5, 1000);
        if let WarmupVerdict::Ok { samples_needed, .. } = v {
            assert!(samples_needed < 20);
        }
    }

    #[test]
    fn slow_learning_more_samples() {
        let v = simulate(0.5, 0.90, 0.001, 100_000);
        if let WarmupVerdict::Ok { samples_needed, .. } = v {
            assert!(samples_needed > 1000);
        }
    }

    #[test]
    fn never_reaches_target() {
        let v = simulate(0.5, 0.99, 0.0001, 100);
        assert!(matches!(v, WarmupVerdict::NeverReaches { .. }));
    }

    #[test]
    fn invalid_target_below_start() {
        assert_eq!(simulate(0.5, 0.5, 0.1, 100), WarmupVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_lr() {
        assert_eq!(simulate(0.5, 0.9, 0.0, 100), WarmupVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_neg_lr() {
        assert_eq!(simulate(0.5, 0.9, -0.1, 100), WarmupVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_max_samples() {
        assert_eq!(simulate(0.5, 0.9, 0.1, 0), WarmupVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_accuracy_over_one() {
        assert_eq!(simulate(0.5, 1.5, 0.1, 100), WarmupVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            simulate(f64::NAN, 0.9, 0.1, 100),
            WarmupVerdict::InvalidConfig
        );
    }

    #[test]
    fn final_accuracy_at_least_target() {
        let v = simulate(0.5, 0.85, 0.1, 1000);
        if let WarmupVerdict::Ok { final_accuracy, .. } = v {
            assert!(final_accuracy >= 0.85);
        }
    }

    #[test]
    fn deterministic() {
        let a = simulate(0.5, 0.85, 0.1, 1000);
        let b = simulate(0.5, 0.85, 0.1, 1000);
        assert_eq!(a, b);
    }
}
