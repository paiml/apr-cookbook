//! # Monte-Carlo Hit-or-Miss Integration
//!
//! Estimate ∫f(x)dx on [0,1] where 0 ≤ f(x) ≤ 1 using the hit-or-miss
//! method: throw N points into the unit square; the ratio of hits
//! (under the curve) to total estimates the area. Returns estimate
//! ×1000 and hit count.
//!
//! Demonstrates the **MC.142** recipe for PMAT-206 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hammersley & Handscomb, Monte Carlo Methods (1964)
//!  ch. 3.2; Buffon's needle precursor (1733).
//!
//! Run with: cargo run --example mc_hit_or_miss_integration
//!
//! Added by PMAT-206 (catalog 1477→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum HitMissVerdict {
    Ok { integral_x1000: u32, hits: u32 },
    InvalidConfig,
}

/// Estimate ∫f over [0,1] where f(x) = x^p / max_y; max_y is the
/// scaling factor so f(x) <= 1. We hardcode f(x) = x^2, max_y = 1.
pub fn estimate(samples: u32, seed: u64) -> HitMissVerdict {
    if samples < 100 {
        return HitMissVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let mut hits = 0u32;
    for _ in 0..samples {
        let x = (lcg(&mut state) as f64) / (u32::MAX as f64);
        let y = (lcg(&mut state) as f64) / (u32::MAX as f64);
        if y < x * x {
            hits += 1;
        }
    }
    let integral = hits as f64 / samples as f64;
    HitMissVerdict::Ok {
        integral_x1000: (integral * 1000.0) as u32,
        hits,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_hit_or_miss_integration")?;

    println!("estimate: {:?}", estimate(100_000, 42));
    println!("invalid: {:?}", estimate(10, 42));
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
    fn invalid_too_few_samples() {
        assert_eq!(estimate(50, 42), HitMissVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = estimate(500, 42);
        let b = estimate(500, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn integral_x_squared_near_third() {
        // ∫₀¹ x² dx = 1/3 → 333 (×1000).
        let v = estimate(100_000, 42);
        if let HitMissVerdict::Ok { integral_x1000, .. } = v {
            assert!((300..=370).contains(&integral_x1000));
        }
    }

    #[test]
    fn integral_in_zero_to_one_range() {
        let v = estimate(500, 42);
        if let HitMissVerdict::Ok { integral_x1000, .. } = v {
            assert!(integral_x1000 <= 1000);
        }
    }

    #[test]
    fn hits_le_samples() {
        let v = estimate(500, 42);
        if let HitMissVerdict::Ok { hits, .. } = v {
            assert!(hits <= 500);
        }
    }

    #[test]
    fn more_samples_tighter_estimate() {
        // SE ∝ 1/√N → larger N gives estimate nearer to true 1/3.
        let small = estimate(500, 42);
        let large = estimate(100_000, 42);
        if let (
            HitMissVerdict::Ok {
                integral_x1000: s, ..
            },
            HitMissVerdict::Ok {
                integral_x1000: l, ..
            },
        ) = (small, large)
        {
            let s_err = (s as i32 - 333).abs();
            let l_err = (l as i32 - 333).abs();
            assert!(l_err <= s_err);
        }
    }

    #[test]
    fn different_seeds_different_estimates() {
        let a = estimate(500, 42);
        let b = estimate(500, 999);
        assert!(a != b);
    }

    #[test]
    fn minimum_samples_accepted() {
        let v = estimate(100, 42);
        assert!(matches!(v, HitMissVerdict::Ok { .. }));
    }

    #[test]
    fn very_large_samples_handled() {
        let v = estimate(1_000_000, 42);
        assert!(matches!(v, HitMissVerdict::Ok { .. }));
    }

    #[test]
    fn hits_consistent_with_integral() {
        let v = estimate(1000, 42);
        if let HitMissVerdict::Ok {
            integral_x1000,
            hits,
        } = v
        {
            let ratio = (hits as f64 / 1000.0 * 1000.0) as u32;
            assert_eq!(ratio, integral_x1000);
        }
    }

    #[test]
    fn hits_zero_only_at_low_n_unlikely() {
        let v = estimate(1000, 42);
        if let HitMissVerdict::Ok { hits, .. } = v {
            // 1000 samples with E[hits]=333 → unlikely to be zero.
            assert!(hits > 0);
        }
    }
}
