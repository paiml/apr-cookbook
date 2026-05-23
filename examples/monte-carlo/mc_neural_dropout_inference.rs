//! # Monte-Carlo Neural Dropout Inference
//!
//! Sim Monte-Carlo dropout: run N forward passes with random masks
//! over a synthetic linear model. Reports mean prediction + variance
//! (uncertainty estimate, Gal & Ghahramani 2016).
//!
//! Demonstrates the **MC.115** recipe for PMAT-197 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Gal & Ghahramani, "Dropout as a Bayesian Approximation"
//!  (ICML 2016).
//!
//! Run with: cargo run --example mc_neural_dropout_inference
//!
//! Added by PMAT-197 (catalog 1396→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DropoutVerdict {
    Ok {
        mean_prediction: f64,
        variance: f64,
        std_dev: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    forward_passes: u32,
    weights: &[f64],
    inputs: &[f64],
    dropout_prob: f64,
    seed: u64,
) -> DropoutVerdict {
    if forward_passes == 0
        || weights.len() != inputs.len()
        || weights.is_empty()
        || !(0.0..=1.0).contains(&dropout_prob)
    {
        return DropoutVerdict::InvalidConfig;
    }
    let mut predictions: Vec<f64> = Vec::with_capacity(forward_passes as usize);
    let mut rng_state = seed | 1;
    for _ in 0..forward_passes {
        let mut sum = 0.0;
        for (w, x) in weights.iter().zip(inputs.iter()) {
            let r = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
            if r >= dropout_prob {
                sum += w * x;
            }
        }
        predictions.push(sum);
    }
    let n = predictions.len() as f64;
    let mean: f64 = predictions.iter().sum::<f64>() / n;
    let variance: f64 = predictions.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n;
    DropoutVerdict::Ok {
        mean_prediction: mean,
        variance,
        std_dev: variance.sqrt(),
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_neural_dropout_inference")?;

    let weights = [0.5, 0.5, 1.0];
    let inputs = [1.0, 2.0, 3.0];
    println!(
        "low dropout: {:?}",
        simulate(100, &weights, &inputs, 0.1, 42)
    );
    println!(
        "high dropout: {:?}",
        simulate(100, &weights, &inputs, 0.5, 42)
    );
    println!("invalid: {:?}", simulate(0, &weights, &inputs, 0.1, 42));
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
    fn no_dropout_zero_variance() {
        let w = [1.0, 1.0];
        let x = [1.0, 1.0];
        let v = simulate(100, &w, &x, 0.0, 42);
        if let DropoutVerdict::Ok { variance, .. } = v {
            assert!(variance < 1e-9);
        }
    }

    #[test]
    fn higher_dropout_more_variance() {
        let w = [1.0, 1.0, 1.0];
        let x = [1.0, 1.0, 1.0];
        let lo = simulate(2000, &w, &x, 0.05, 42);
        let hi = simulate(2000, &w, &x, 0.5, 42);
        if let (DropoutVerdict::Ok { variance: l, .. }, DropoutVerdict::Ok { variance: h, .. }) =
            (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn invalid_zero_passes() {
        let w = [1.0];
        let x = [1.0];
        assert_eq!(simulate(0, &w, &x, 0.1, 42), DropoutVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_mismatch_dims() {
        let w = [1.0, 1.0];
        let x = [1.0];
        assert_eq!(
            simulate(100, &w, &x, 0.1, 42),
            DropoutVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_empty_weights() {
        assert_eq!(
            simulate(100, &[], &[], 0.1, 42),
            DropoutVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_dropout_prob() {
        let w = [1.0];
        let x = [1.0];
        assert_eq!(
            simulate(100, &w, &x, 1.5, 42),
            DropoutVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let w = [1.0];
        let x = [1.0];
        let a = simulate(100, &w, &x, 0.1, 42);
        let b = simulate(100, &w, &x, 0.1, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn variance_nonneg() {
        let w = [1.0, 1.0];
        let x = [1.0, 1.0];
        let v = simulate(100, &w, &x, 0.3, 42);
        if let DropoutVerdict::Ok { variance, .. } = v {
            assert!(variance >= 0.0);
        }
    }

    #[test]
    fn std_dev_sqrt_variance() {
        let w = [1.0, 1.0];
        let x = [1.0, 1.0];
        let v = simulate(100, &w, &x, 0.3, 42);
        if let DropoutVerdict::Ok {
            variance, std_dev, ..
        } = v
        {
            assert!((std_dev - variance.sqrt()).abs() < 1e-9);
        }
    }

    #[test]
    fn many_passes_handled() {
        let w = [1.0, 1.0];
        let x = [1.0, 1.0];
        let v = simulate(10_000, &w, &x, 0.3, 42);
        assert!(matches!(v, DropoutVerdict::Ok { .. }));
    }

    #[test]
    fn outputs_finite() {
        let w = [1.0, 2.0];
        let x = [3.0, 4.0];
        let v = simulate(100, &w, &x, 0.3, 42);
        if let DropoutVerdict::Ok {
            mean_prediction,
            variance,
            std_dev,
        } = v
        {
            assert!(mean_prediction.is_finite());
            assert!(variance.is_finite());
            assert!(std_dev.is_finite());
        }
    }
}
