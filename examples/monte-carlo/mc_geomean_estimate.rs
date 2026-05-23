//! # Monte-Carlo Geometric Mean Estimate
//!
//! Estimate the geometric mean of a discrete positive distribution by
//! sampling N values. Returns the GM estimate (×1000 for u32 fixed
//! point) and the sample variance of log values.
//!
//! Demonstrates the **MC.138** recipe for PMAT-204 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Aitchison, The Statistical Analysis of Compositional
//!  Data (1986); GM is exp(mean(log x)).
//!
//! Run with: cargo run --example mc_geomean_estimate
//!
//! Added by PMAT-204 (catalog 1459→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum GmVerdict {
    Ok {
        gm_x1000: u32,
        log_variance_x1000: u32,
    },
    InvalidConfig,
}

pub fn estimate(values: &[u32], samples: u32, seed: u64) -> GmVerdict {
    if values.is_empty() || samples == 0 || values.contains(&0) {
        return GmVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut log_sum = 0.0f64;
    let mut log_sq_sum = 0.0f64;
    for _ in 0..samples {
        let idx = (lcg(&mut state) as usize) % values.len();
        let log_v = (values[idx] as f64).ln();
        log_sum += log_v;
        log_sq_sum += log_v * log_v;
    }
    let mean_log = log_sum / samples as f64;
    let var_log = (log_sq_sum / samples as f64) - mean_log * mean_log;
    let gm = mean_log.exp();
    GmVerdict::Ok {
        gm_x1000: (gm * 1000.0) as u32,
        log_variance_x1000: (var_log.max(0.0) * 1000.0) as u32,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_geomean_estimate")?;

    let values = [1u32, 2, 4, 8, 16];
    println!("dyadic: {:?}", estimate(&values, 10_000, 42));
    println!("invalid: {:?}", estimate(&[], 1000, 42));
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
    fn invalid_empty_values() {
        assert_eq!(estimate(&[], 1000, 42), GmVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_samples() {
        assert_eq!(estimate(&[1], 0, 42), GmVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_value() {
        assert_eq!(estimate(&[1, 0, 2], 1000, 42), GmVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = estimate(&[1, 2, 4], 1000, 42);
        let b = estimate(&[1, 2, 4], 1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn dyadic_gm_near_4() {
        // GM(1,2,4,8,16) = (1*2*4*8*16)^(1/5) = 1024^0.2 = 4
        let v = estimate(&[1, 2, 4, 8, 16], 50_000, 42);
        if let GmVerdict::Ok { gm_x1000, .. } = v {
            // Allow ±10% bands at this sample size.
            assert!((3600..=4400).contains(&gm_x1000));
        }
    }

    #[test]
    fn constant_values_zero_variance() {
        let v = estimate(&[5, 5, 5], 1000, 42);
        if let GmVerdict::Ok {
            log_variance_x1000, ..
        } = v
        {
            assert_eq!(log_variance_x1000, 0);
        }
    }

    #[test]
    fn constant_values_gm_equals_value() {
        let v = estimate(&[7, 7, 7], 1000, 42);
        if let GmVerdict::Ok { gm_x1000, .. } = v {
            assert!((6900..=7100).contains(&gm_x1000));
        }
    }

    #[test]
    fn single_value_handled() {
        let v = estimate(&[5], 1000, 42);
        if let GmVerdict::Ok { gm_x1000, .. } = v {
            assert!((4900..=5100).contains(&gm_x1000));
        }
    }

    #[test]
    fn variance_non_negative() {
        let v = estimate(&[1, 2, 4, 8], 1000, 42);
        if let GmVerdict::Ok {
            log_variance_x1000, ..
        } = v
        {
            // u32 is naturally non-negative; sanity check value is finite.
            assert!(log_variance_x1000 < u32::MAX);
        }
    }

    #[test]
    fn large_value_handled() {
        let v = estimate(&[1_000_000], 1000, 42);
        if let GmVerdict::Ok { gm_x1000, .. } = v {
            assert!(gm_x1000 > 999_000_000);
        }
    }

    #[test]
    fn many_samples_handled() {
        let v = estimate(&[1, 2, 4], 100_000, 42);
        assert!(matches!(v, GmVerdict::Ok { .. }));
    }

    #[test]
    fn gm_le_arithmetic_mean() {
        // GM ≤ AM for positive values (AM-GM inequality).
        let values = [1u32, 4, 9];
        let v = estimate(&values, 50_000, 42);
        if let GmVerdict::Ok { gm_x1000, .. } = v {
            // AM=14/3≈4.667 → AM*1000 ≈ 4667
            assert!(gm_x1000 < 4700);
        }
    }
}
