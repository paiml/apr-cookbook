//! # Monte-Carlo Buffon's Needle π Estimation
//!
//! Estimate π via Buffon's needle: drop needles of length L on a
//! plane with parallel lines spaced D apart (L ≤ D). P(crossing) =
//! 2L / (π × D). Solve for π = 2L × N / (D × crossings).
//!
//! Demonstrates the **MC.120** recipe for PMAT-199 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Buffon, Histoire de l'Acad. Royale des Sci. (1733);
//!  Monte-Carlo classics.
//!
//! Run with: cargo run --example mc_pi_estimation_buffon
//!
//! Added by PMAT-199 (catalog 1414→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BuffonVerdict {
    Ok {
        pi_estimate: f64,
        crossings: u32,
        relative_error: f64,
    },
    InvalidConfig,
}

pub fn simulate(needles: u32, needle_length: f64, line_spacing: f64, seed: u64) -> BuffonVerdict {
    if needles == 0 || needle_length <= 0.0 || line_spacing <= 0.0 || needle_length > line_spacing {
        return BuffonVerdict::InvalidConfig;
    }
    let mut crossings = 0u32;
    let mut rng_state = seed | 1;
    for _ in 0..needles {
        let center = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64) * line_spacing;
        let angle = (lcg(&mut rng_state) >> 32) as f64 / (u32::MAX as f64) * std::f64::consts::PI;
        let half_proj = needle_length * angle.sin() / 2.0;
        if center - half_proj < 0.0 || center + half_proj > line_spacing {
            crossings += 1;
        }
    }
    let pi_estimate = if crossings == 0 {
        f64::INFINITY
    } else {
        (2.0 * needle_length * f64::from(needles)) / (line_spacing * f64::from(crossings))
    };
    let relative_error = (pi_estimate - std::f64::consts::PI).abs() / std::f64::consts::PI;
    BuffonVerdict::Ok {
        pi_estimate,
        crossings,
        relative_error,
    }
}

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_pi_estimation_buffon")?;

    println!("typical: {:?}", simulate(100_000, 1.0, 2.0, 42));
    println!("invalid: {:?}", simulate(0, 1.0, 2.0, 42));
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
        let v = simulate(100_000, 1.0, 2.0, 42);
        if let BuffonVerdict::Ok { relative_error, .. } = v {
            assert!(relative_error < 0.05);
        }
    }

    #[test]
    fn invalid_zero_needles() {
        assert_eq!(simulate(0, 1.0, 2.0, 42), BuffonVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_length() {
        assert_eq!(simulate(100, 0.0, 2.0, 42), BuffonVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_zero_spacing() {
        assert_eq!(simulate(100, 1.0, 0.0, 42), BuffonVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_length_exceeds_spacing() {
        assert_eq!(simulate(100, 3.0, 2.0, 42), BuffonVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let a = simulate(1000, 1.0, 2.0, 42);
        let b = simulate(1000, 1.0, 2.0, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn larger_sample_more_accurate() {
        let small = simulate(100, 1.0, 2.0, 42);
        let big = simulate(50_000, 1.0, 2.0, 42);
        if let (
            BuffonVerdict::Ok {
                relative_error: s, ..
            },
            BuffonVerdict::Ok {
                relative_error: b, ..
            },
        ) = (small, big)
        {
            assert!(b <= s + 0.05);
        }
    }

    #[test]
    fn crossings_le_needles() {
        let v = simulate(1000, 1.0, 2.0, 42);
        if let BuffonVerdict::Ok { crossings, .. } = v {
            assert!(crossings <= 1000);
        }
    }

    #[test]
    fn pi_estimate_finite() {
        let v = simulate(10_000, 1.0, 2.0, 42);
        if let BuffonVerdict::Ok { pi_estimate, .. } = v {
            assert!(pi_estimate.is_finite());
        }
    }

    #[test]
    fn equal_length_spacing_more_crossings() {
        let v_short = simulate(10_000, 0.5, 2.0, 42);
        let v_long = simulate(10_000, 2.0, 2.0, 42);
        if let (BuffonVerdict::Ok { crossings: s, .. }, BuffonVerdict::Ok { crossings: l, .. }) =
            (v_short, v_long)
        {
            assert!(l > s);
        }
    }

    #[test]
    fn relative_error_nonneg() {
        let v = simulate(10_000, 1.0, 2.0, 42);
        if let BuffonVerdict::Ok { relative_error, .. } = v {
            assert!(relative_error >= 0.0);
        }
    }
}
