//! # Monte-Carlo Buffon's Needle π Estimation
//!
//! Estimate π via Buffon's needle: drop short needles (length ≤ line
//! spacing) on a plane of parallel lines; count crossings. π ≈ 2L·N
//! / (d·C) where L=needle length, d=line spacing, N=trials, C=crossings.
//! Returns π estimate (×1000).
//!
//! Demonstrates the **MC.187** recipe for PMAT-221 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Buffon, "Essai d'arithmétique morale" (1777); Laplace
//!  refinement (1812).
//!
//! Run with: cargo run --example mc_buffon_needle_pi
//!
//! Added by PMAT-221 (catalog 1612→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BuffonVerdict {
    Ok {
        pi_estimate_x1000: u32,
        crossings: u32,
    },
    InvalidConfig,
}

pub fn simulate(trials: u32, seed: u64) -> BuffonVerdict {
    if trials < 1000 {
        return BuffonVerdict::InvalidConfig;
    }
    let mut state = seed | 1;
    let needle_len = 1.0f64;
    let line_spacing = 2.0f64; // L < d for short-needle case
    let mut crossings = 0u32;
    for _ in 0..trials {
        // y = distance from needle's center to nearest line in [0, d/2)
        let y = (lcg(&mut state) as f64) / (u32::MAX as f64) * line_spacing / 2.0;
        // theta = angle in [0, π/2)
        let theta = (lcg(&mut state) as f64) / (u32::MAX as f64) * std::f64::consts::PI / 2.0;
        if y < (needle_len / 2.0) * theta.sin() {
            crossings += 1;
        }
    }
    if crossings == 0 {
        return BuffonVerdict::Ok {
            pi_estimate_x1000: 0,
            crossings: 0,
        };
    }
    let pi_est = (2.0 * needle_len * trials as f64) / (line_spacing * crossings as f64);
    BuffonVerdict::Ok {
        pi_estimate_x1000: (pi_est * 1000.0) as u32,
        crossings,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_buffon_needle_pi")?;

    println!("100k trials: {:?}", simulate(100_000, 42));
    println!("invalid: {:?}", simulate(50, 42));
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
    fn invalid_too_few_trials() {
        assert_eq!(simulate(50, 42), BuffonVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(1000, 42);
        let b = simulate(1000, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn pi_estimate_near_3142() {
        // π × 1000 = 3142. Allow ±10% with 100k trials.
        let v = simulate(100_000, 42);
        if let BuffonVerdict::Ok {
            pi_estimate_x1000, ..
        } = v
        {
            assert!((2800..=3500).contains(&pi_estimate_x1000));
        }
    }

    #[test]
    fn crossings_le_trials() {
        let v = simulate(1000, 42);
        if let BuffonVerdict::Ok { crossings, .. } = v {
            assert!(crossings <= 1000);
        }
    }

    #[test]
    fn crossings_at_least_one() {
        // With 1000 trials, almost certainly some crossings.
        let v = simulate(1000, 42);
        if let BuffonVerdict::Ok { crossings, .. } = v {
            assert!(crossings >= 1);
        }
    }

    #[test]
    fn more_trials_better_estimate() {
        let small = simulate(1000, 42);
        let large = simulate(100_000, 42);
        if let (
            BuffonVerdict::Ok {
                pi_estimate_x1000: s,
                ..
            },
            BuffonVerdict::Ok {
                pi_estimate_x1000: l,
                ..
            },
        ) = (small, large)
        {
            let s_err = (s as i32 - 3142).abs();
            let l_err = (l as i32 - 3142).abs();
            assert!(l_err <= s_err);
        }
    }

    #[test]
    fn min_trials_accepted() {
        let v = simulate(1000, 42);
        assert!(matches!(v, BuffonVerdict::Ok { .. }));
    }

    #[test]
    fn many_trials_handled() {
        let v = simulate(1_000_000, 42);
        assert!(matches!(v, BuffonVerdict::Ok { .. }));
    }

    #[test]
    fn different_seeds_different_estimates() {
        let a = simulate(1000, 42);
        let b = simulate(1000, 999);
        assert!(a != b);
    }

    #[test]
    fn pi_finite() {
        let v = simulate(1000, 42);
        if let BuffonVerdict::Ok {
            pi_estimate_x1000, ..
        } = v
        {
            assert!(pi_estimate_x1000 < u32::MAX);
        }
    }
}
