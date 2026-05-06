//! # Monte-Carlo Quantization Loss Estimator
//!
//! Estimate accuracy loss from quantization (FP32 → INT8 etc.) by
//! sampling perturbations proportional to quantization noise. Returns
//! mean loss and 95th percentile.
//!
//! Demonstrates the **MC.32** recipe for PMAT-168 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Quantization-aware training analysis (Jacob et al. 2018).
//!
//! Run with: cargo run --example mc_quantization_loss_estimator
//!
//! Added by PMAT-168 (catalog 1135→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum QuantVerdict {
    Ok {
        mean_loss: f64,
        p95_loss: f64,
        max_loss: f64,
    },
    InvalidConfig,
}

pub fn estimate(
    base_accuracy: f64,
    quantization_bits: u32,
    samples: u32,
    seed: u64,
) -> QuantVerdict {
    if !base_accuracy.is_finite()
        || !(0.0..=1.0).contains(&base_accuracy)
        || quantization_bits == 0
        || quantization_bits > 32
        || samples == 0
    {
        return QuantVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    // Approximate quantization noise: ~ 1 / 2^bits.
    let noise_scale = 1.0 / 2.0_f64.powi(quantization_bits as i32);
    let mut losses: Vec<f64> = Vec::with_capacity(samples as usize);
    for _ in 0..samples {
        // Loss = base_accuracy × noise × random factor in [0, 2].
        let factor = unit(&mut rng_state) * 2.0;
        let loss = (base_accuracy * noise_scale * factor).clamp(0.0, base_accuracy);
        losses.push(loss);
    }
    losses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean_loss = losses.iter().sum::<f64>() / f64::from(samples);
    let p95_loss = losses[(samples as f64 * 0.95) as usize];
    let max_loss = *losses.last().unwrap_or(&0.0);
    QuantVerdict::Ok {
        mean_loss,
        p95_loss,
        max_loss,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_quantization_loss_estimator")?;

    println!("INT8: {:?}", estimate(0.95, 8, 10_000, 42));
    println!("INT4: {:?}", estimate(0.95, 4, 10_000, 42));
    println!("INT16: {:?}", estimate(0.95, 16, 10_000, 42));
    println!("invalid: {:?}", estimate(0.95, 0, 100, 42));
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
    fn fewer_bits_higher_loss() {
        let v_int4 = estimate(0.95, 4, 10_000, 42);
        let v_int8 = estimate(0.95, 8, 10_000, 42);
        if let (QuantVerdict::Ok { mean_loss: l4, .. }, QuantVerdict::Ok { mean_loss: l8, .. }) =
            (v_int4, v_int8)
        {
            assert!(l4 > l8);
        }
    }

    #[test]
    fn p95_above_mean() {
        let v = estimate(0.95, 8, 10_000, 42);
        if let QuantVerdict::Ok {
            mean_loss,
            p95_loss,
            ..
        } = v
        {
            assert!(p95_loss >= mean_loss);
        }
    }

    #[test]
    fn max_above_p95() {
        let v = estimate(0.95, 8, 10_000, 42);
        if let QuantVerdict::Ok {
            p95_loss, max_loss, ..
        } = v
        {
            assert!(max_loss >= p95_loss);
        }
    }

    #[test]
    fn invalid_zero_bits() {
        assert_eq!(estimate(0.95, 0, 100, 42), QuantVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_many_bits() {
        assert_eq!(estimate(0.95, 33, 100, 42), QuantVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(estimate(0.95, 8, 0, 42), QuantVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_accuracy_over_one() {
        assert_eq!(estimate(1.5, 8, 100, 42), QuantVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(estimate(f64::NAN, 8, 100, 42), QuantVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = estimate(0.95, 8, 1000, 42);
        let b = estimate(0.95, 8, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn loss_bounded_by_accuracy() {
        let v = estimate(0.5, 4, 1000, 42);
        if let QuantVerdict::Ok { max_loss, .. } = v {
            assert!(max_loss <= 0.5 + 1e-9);
        }
    }
}
