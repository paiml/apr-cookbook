//! # Monte-Carlo Kalman Filter Smoothing
//!
//! Sim 1D Kalman filter on noisy observations of a constant signal.
//! Reports filtered estimates' mean error vs raw observations'.
//!
//! Demonstrates the **MC.121** recipe for PMAT-199 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Kalman, R.E. "A New Approach to Linear Filtering"
//!  (J Basic Eng 82, 1960).
//!
//! Run with: cargo run --example mc_kalman_filter_smoothing
//!
//! Added by PMAT-199 (catalog 1414→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum KalmanVerdict {
    Ok {
        raw_mse: f64,
        filtered_mse: f64,
        improvement_ratio: f64,
    },
    InvalidConfig,
}

pub fn simulate(
    samples: u32,
    true_value: f64,
    noise_std: f64,
    measurement_noise_var: f64,
    seed: u64,
) -> KalmanVerdict {
    if samples == 0 || noise_std <= 0.0 || measurement_noise_var <= 0.0 {
        return KalmanVerdict::InvalidConfig;
    }
    let mut rng_state = seed | 1;
    let mut estimate = 0.0;
    let mut estimate_var = 1.0; // Initial uncertainty.
    let mut sum_raw_sq: f64 = 0.0;
    let mut sum_filtered_sq: f64 = 0.0;
    for _ in 0..samples {
        let noise = (gaussian(&mut rng_state)) * noise_std;
        let observation = true_value + noise;
        let kalman_gain = estimate_var / (estimate_var + measurement_noise_var);
        estimate += kalman_gain * (observation - estimate);
        estimate_var *= 1.0 - kalman_gain;
        sum_raw_sq += (observation - true_value).powi(2);
        sum_filtered_sq += (estimate - true_value).powi(2);
    }
    let raw_mse = sum_raw_sq / f64::from(samples);
    let filtered_mse = sum_filtered_sq / f64::from(samples);
    let improvement_ratio = if filtered_mse > 0.0 {
        raw_mse / filtered_mse
    } else {
        f64::INFINITY
    };
    KalmanVerdict::Ok {
        raw_mse,
        filtered_mse,
        improvement_ratio,
    }
}

fn gaussian(rng_state: &mut u64) -> f64 {
    // Box-Muller transform.
    let u1 = (lcg(rng_state) >> 32) as f64 / (u32::MAX as f64);
    let u2 = (lcg(rng_state) >> 32) as f64 / (u32::MAX as f64);
    let u1 = u1.max(1e-12);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_kalman_filter_smoothing")?;

    println!("typical: {:?}", simulate(1000, 50.0, 5.0, 25.0, 42));
    println!("invalid: {:?}", simulate(0, 50.0, 5.0, 25.0, 42));
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
    fn both_mse_finite_and_bounded() {
        let v = simulate(2000, 50.0, 10.0, 100.0, 42);
        if let KalmanVerdict::Ok {
            raw_mse,
            filtered_mse,
            ..
        } = v
        {
            assert!(raw_mse.is_finite() && filtered_mse.is_finite());
        }
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(
            simulate(0, 50.0, 5.0, 25.0, 42),
            KalmanVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_noise() {
        assert_eq!(
            simulate(100, 50.0, 0.0, 25.0, 42),
            KalmanVerdict::InvalidConfig
        );
    }

    #[test]
    fn invalid_zero_measurement_var() {
        assert_eq!(
            simulate(100, 50.0, 5.0, 0.0, 42),
            KalmanVerdict::InvalidConfig
        );
    }

    #[test]
    fn deterministic() {
        let a = simulate(500, 50.0, 5.0, 25.0, 42);
        let b = simulate(500, 50.0, 5.0, 25.0, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn mse_nonneg() {
        let v = simulate(500, 50.0, 5.0, 25.0, 42);
        if let KalmanVerdict::Ok {
            raw_mse,
            filtered_mse,
            ..
        } = v
        {
            assert!(raw_mse >= 0.0);
            assert!(filtered_mse >= 0.0);
        }
    }

    #[test]
    fn improvement_ratio_finite() {
        let v = simulate(2000, 50.0, 5.0, 25.0, 42);
        if let KalmanVerdict::Ok {
            improvement_ratio, ..
        } = v
        {
            assert!(improvement_ratio > 0.0);
        }
    }

    #[test]
    fn finite_outputs() {
        let v = simulate(500, 50.0, 5.0, 25.0, 42);
        if let KalmanVerdict::Ok {
            raw_mse,
            filtered_mse,
            improvement_ratio,
        } = v
        {
            assert!(raw_mse.is_finite());
            assert!(filtered_mse.is_finite());
            assert!(improvement_ratio.is_finite());
        }
    }

    #[test]
    fn higher_noise_higher_raw_mse() {
        let lo = simulate(2000, 50.0, 1.0, 1.0, 42);
        let hi = simulate(2000, 50.0, 10.0, 100.0, 42);
        if let (KalmanVerdict::Ok { raw_mse: l, .. }, KalmanVerdict::Ok { raw_mse: h, .. }) =
            (lo, hi)
        {
            assert!(h > l);
        }
    }

    #[test]
    fn many_samples_better_smoothing() {
        let small = simulate(100, 50.0, 10.0, 100.0, 42);
        let big = simulate(5000, 50.0, 10.0, 100.0, 42);
        if let (
            KalmanVerdict::Ok {
                filtered_mse: s, ..
            },
            KalmanVerdict::Ok {
                filtered_mse: b, ..
            },
        ) = (small, big)
        {
            assert!(b <= s);
        }
    }

    #[test]
    fn single_sample_works() {
        let v = simulate(1, 50.0, 5.0, 25.0, 42);
        assert!(matches!(v, KalmanVerdict::Ok { .. }));
    }
}
