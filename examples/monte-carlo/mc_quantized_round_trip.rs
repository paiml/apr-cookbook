//! # Monte-Carlo Quantization Round-Trip Loss
//!
//! Sim FP32 → INT_K → FP32 round-trip MSE on N samples drawn from
//! uniform [-1, 1]. Returns mean squared error and max absolute error.
//!
//! Demonstrates the **MC.43** recipe for PMAT-172 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: integer quantization round-trip analysis (Jacob et al. 2018).
//!
//! Run with: cargo run --example mc_quantized_round_trip
//!
//! Added by PMAT-172 (catalog 1171→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum RoundTripVerdict {
    Ok { mse: f64, max_abs_error: f64 },
    InvalidConfig,
}

pub fn simulate(bits: u32, samples: u32, seed: u64) -> RoundTripVerdict {
    if bits == 0 || bits > 16 || samples == 0 {
        return RoundTripVerdict::InvalidConfig;
    }
    let levels = (1u64 << bits) - 1;
    let mut rng_state = seed | 1;
    let mut sum_sq = 0.0;
    let mut max_err: f64 = 0.0;
    for _ in 0..samples {
        let original = unit(&mut rng_state) * 2.0 - 1.0;
        // Quantize to levels in [0, levels].
        let normalized = (original + 1.0) / 2.0;
        let q = (normalized * levels as f64).round() as u64;
        let dequantized = (q as f64 / levels as f64) * 2.0 - 1.0;
        let err = original - dequantized;
        sum_sq += err * err;
        if err.abs() > max_err {
            max_err = err.abs();
        }
    }
    let mse = sum_sq / f64::from(samples);
    RoundTripVerdict::Ok {
        mse,
        max_abs_error: max_err,
    }
}

fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_quantized_round_trip")?;

    println!("INT8: {:?}", simulate(8, 10_000, 42));
    println!("INT4: {:?}", simulate(4, 10_000, 42));
    println!("INT16: {:?}", simulate(16, 10_000, 42));
    println!("invalid: {:?}", simulate(0, 100, 42));
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
    fn fewer_bits_higher_mse() {
        let v_int4 = simulate(4, 10_000, 42);
        let v_int8 = simulate(8, 10_000, 42);
        if let (RoundTripVerdict::Ok { mse: m4, .. }, RoundTripVerdict::Ok { mse: m8, .. }) =
            (v_int4, v_int8)
        {
            assert!(m4 > m8);
        }
    }

    #[test]
    fn more_bits_lower_max_err() {
        let v_int4 = simulate(4, 10_000, 42);
        let v_int16 = simulate(16, 10_000, 42);
        if let (
            RoundTripVerdict::Ok {
                max_abs_error: e4, ..
            },
            RoundTripVerdict::Ok {
                max_abs_error: e16, ..
            },
        ) = (v_int4, v_int16)
        {
            assert!(e4 > e16);
        }
    }

    #[test]
    fn invalid_zero_bits() {
        assert_eq!(simulate(0, 100, 42), RoundTripVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_many_bits() {
        assert_eq!(simulate(17, 100, 42), RoundTripVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(simulate(8, 0, 42), RoundTripVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(8, 1000, 42);
        let b = simulate(8, 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn mse_non_negative() {
        let v = simulate(8, 1000, 42);
        if let RoundTripVerdict::Ok { mse, .. } = v {
            assert!(mse >= 0.0);
        }
    }

    #[test]
    fn max_err_bounded() {
        // Round-trip max error ≤ half the level spacing = 1/levels.
        let v = simulate(8, 10_000, 42);
        if let RoundTripVerdict::Ok { max_abs_error, .. } = v {
            assert!(max_abs_error <= 1.0 / 255.0 + 1e-6);
        }
    }

    #[test]
    fn small_n_works() {
        let v = simulate(8, 5, 42);
        assert!(matches!(v, RoundTripVerdict::Ok { .. }));
    }

    #[test]
    fn high_bits_low_mse() {
        let v = simulate(16, 10_000, 42);
        if let RoundTripVerdict::Ok { mse, .. } = v {
            assert!(mse < 1e-8);
        }
    }
}
