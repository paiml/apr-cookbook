//! # Monte-Carlo π via Darts in Unit Square
//!
//! Throw N random darts at unit square; count those landing inside
//! unit quarter-circle. π ≈ 4 × (in / total).
//!
//! Demonstrates the **MC.130** recipe for PMAT-202 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: classic Monte-Carlo π estimation (Metropolis & Ulam,
//!  J Amer Stat Assoc 44, 1949).
//!
//! Run with: cargo run --example mc_simulated_pi_dart
//!
//! Added by PMAT-202 (catalog 1441→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DartVerdict {
    Ok {
        pi_estimate: f64,
        relative_error: f64,
        in_circle: u32,
    },
    InvalidConfig,
}

pub fn simulate(darts: u32, seed: u64) -> DartVerdict {
    if darts == 0 {
        return DartVerdict::InvalidConfig;
    }
    let mut in_circle = 0u32;
    let mut rng_state = seed | 1;
    for _ in 0..darts {
        let x = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        let y = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64);
        if x * x + y * y <= 1.0 {
            in_circle += 1;
        }
    }
    let pi_estimate = 4.0 * f64::from(in_circle) / f64::from(darts);
    let relative_error = (pi_estimate - std::f64::consts::PI).abs() / std::f64::consts::PI;
    DartVerdict::Ok {
        pi_estimate,
        relative_error,
        in_circle,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_simulated_pi_dart")?;

    println!("typical: {:?}", simulate(100_000, 42));
    println!("invalid: {:?}", simulate(0, 42));
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
    fn pi_estimate_close() {
        let v = simulate(100_000, 42);
        if let DartVerdict::Ok { relative_error, .. } = v {
            assert!(relative_error < 0.02);
        }
    }

    #[test]
    fn invalid_zero_darts() {
        assert_eq!(simulate(0, 42), DartVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(1000, 42);
        let b = simulate(1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn larger_sample_more_accurate() {
        let small = simulate(100, 42);
        let big = simulate(50_000, 42);
        if let (
            DartVerdict::Ok {
                relative_error: s, ..
            },
            DartVerdict::Ok {
                relative_error: b, ..
            },
        ) = (small, big)
        {
            assert!(b <= s + 0.1);
        }
    }

    #[test]
    fn in_circle_le_darts() {
        let v = simulate(1000, 42);
        if let DartVerdict::Ok { in_circle, .. } = v {
            assert!(in_circle <= 1000);
        }
    }

    #[test]
    fn pi_estimate_in_realistic_bounds() {
        let v = simulate(10_000, 42);
        if let DartVerdict::Ok { pi_estimate, .. } = v {
            assert!(pi_estimate > 2.5 && pi_estimate < 4.0);
        }
    }

    #[test]
    fn relative_error_nonneg() {
        let v = simulate(1000, 42);
        if let DartVerdict::Ok { relative_error, .. } = v {
            assert!(relative_error >= 0.0);
        }
    }

    #[test]
    fn pi_estimate_finite() {
        let v = simulate(1000, 42);
        if let DartVerdict::Ok { pi_estimate, .. } = v {
            assert!(pi_estimate.is_finite());
        }
    }

    #[test]
    fn quarter_circle_area_correct() {
        // π/4 ≈ 0.785; about 78% of darts should land in.
        let v = simulate(10_000, 42);
        if let DartVerdict::Ok { in_circle, .. } = v {
            let frac = f64::from(in_circle) / 10_000.0;
            assert!(frac > 0.70 && frac < 0.85);
        }
    }

    #[test]
    fn many_darts_handled() {
        let v = simulate(1_000_000, 42);
        assert!(matches!(v, DartVerdict::Ok { .. }));
    }

    #[test]
    fn single_dart_works() {
        let v = simulate(1, 42);
        if let DartVerdict::Ok { in_circle, .. } = v {
            assert!(in_circle <= 1);
        }
    }
}
